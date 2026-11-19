import json,re,numpy as np,os
amap=json.load(open('allocator_map.json')); W=amap['weights']; kvr=amap['kv']['kv_ranges']
gmin=min([w[1] for w in W]+[k[0] for k in kvr])
def L_of(n):
    m=re.search(r'layers\.(\d+)\.',n); return int(m.group(1)) if m else -1
def typ(n):
    n=n.lower()
    if 'embed' in n: return 'embed'
    if '.experts.' in n: return 'moe'
    if 'shared_expert' in n: return 'shared'
    if 'self_attn' in n or 'linear_attn' in n: return 'attn'
    if 'weight_scale' in n or 'alpha' in n or 'global_scale' in n or 'input_scale' in n: return 'scale'
    if 'lm_head' in n: return 'lmhead'
    return 'other'
rec=sorted([(p,p+nb,L_of(nm),typ(nm)) for nm,p,nb,sh,dt in W])
SA=np.array([r[0] for r in rec],np.int64);EA=np.array([r[1] for r in rec],np.int64);LA=np.array([r[2] for r in rec],np.int64);TA=np.array([r[3] for r in rec],object)
RES=16*1024*1024
nsyn=np.maximum(1,np.ceil((EA-SA)/RES).astype(np.int64)); base=np.concatenate([[0],np.cumsum(nsyn)[:-1]]).astype(np.int64)
total_rows=int(nsyn.sum())
owner_of_row=np.zeros(total_rows,np.int64)
for i in range(len(rec)): owner_of_row[base[i]:base[i]+nsyn[i]]=LA[i]
def locate(a):
    j=np.searchsorted(SA,a,'right')-1; jc=np.clip(j,0,len(SA)-1)
    ok=(j>=0)&(a<EA[jc])&(a>=SA[jc]); sub=np.clip((a-SA[jc])//RES,0,nsyn[jc]-1); row=base[jc]+sub
    return np.where(ok,row,-1), np.where(ok,LA[jc],-9)
H=np.zeros((48,total_rows)); dropped=np.zeros(48,np.int64)
dt=np.dtype([('grid','<u8'),('cta','<u4'),('warp','<u2'),('opcode','<u2'),('addr','<u8'),('size','<u4'),('layer','<u4')])
REG="v5_trace.51233.bin"; n=os.path.getsize(REG)//32; mm=np.memmap(REG,dtype=dt,mode='r'); CH=20_000_000
for s in range(0,n,CH):
    c=mm[s:s+CH]; lay=c['layer'].astype(np.int64); a=c['addr'].astype(np.int64)
    ok=(lay>=0)&(lay<48); row,rown=locate(a[ok]); al=lay[ok]
    v=row>=0; ownr=np.where(v,owner_of_row[np.clip(row,0,total_rows-1)],-9)
    keep=v&(ownr<0)  # row>=0 但 ownr 一定>=0; 用 owner==al 判定
    keep=v&(ownr==al)
    bad=v&(ownr>=0)&(ownr!=al)
    H+=np.bincount(al[keep]*total_rows+row[keep],minlength=48*total_rows).reshape(48,total_rows)
    for L in np.unique(al[bad]): dropped[L]+=int((al[bad]==L).sum())
    if (s//CH)%40==0: print("reg %.0f%%"%(s/n*100),flush=True)
td=np.dtype([('grid','<u8'),('layer','<u4'),('is_write','<u1'),('pad','<u1',(3,)),('start','<u8'),('span','<u8'),('nelem','<u8')])
t=np.fromfile("v5_trace.51233.tma.bin",dtype=td); lay=t['layer'].astype(np.int64); a=t['start'].astype(np.int64); ne=t['nelem'].astype(np.float64)
ok=(lay>=0)&(lay<48); row,rown=locate(a[ok]); al=lay[ok]; v=row>=0; ownr=np.where(v,owner_of_row[np.clip(row,0,total_rows-1)],-9)
keep=v&(ownr==al); bad=v&(ownr>=0)&(ownr!=al)
H+=np.bincount(al[keep]*total_rows+row[keep],weights=ne[ok][keep],minlength=48*total_rows).reshape(48,total_rows)
tdrop=np.zeros(48)
for L in np.unique(al[bad]): tdrop[L]+=ne[ok][bad][al[bad]==L].sum()
print("\n被剔除的'跨层权重'记录(判为解析假象):")
for L in range(48):
    if dropped[L]>0 or tdrop[L]>0: print("  L%-2d regular剔=%d, TMA剔元素=%.3e"%(L,dropped[L],tdrop[L]))
tot=H.sum(); print("保留总访问=%.4e ; 剔除总量(记录+元素)=%d / %.3e"%(tot,dropped.sum(),tdrop.sum()))
np.savez_compressed("v5_aligned_clean.npz",H=H,total_rows=total_rows,SA=SA,EA=EA,LA=LA,TA=TA,nsyn=nsyn,base=base,RES=RES,gmin=gmin)
print("saved v5_aligned_clean.npz")

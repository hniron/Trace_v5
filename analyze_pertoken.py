#!/usr/bin/env python
"""按 token 切分的访存直方图：只累加选中的 6 个 token。
Regular: 以 200->100 边界切 token；TMA: 以 47->0 回退切 token。两边索引对齐。"""
import numpy as np, json, os
REG="v5_trace.51233.bin"; TMA="v5_trace.51233.tma.bin"
SEL=[3,13,23,33,43,53]
CH=20_000_000
m=np.load("v5_meta.npz"); ymin=int(m['ymin']); ymax=int(m['ymax']); bw=float(m['bw']); NB_X=int(m['NB_X']); NB_Y=int(m['NB_Y'])
EMB_W=2; OUTX=EMB_W+48
amap=json.load(open("allocator_map.json")); W=amap['weights']; kvr=amap['kv']['kv_ranges']
cats=['embed','moe','attn','lmhead','weight','kv','act']; cidx={c:i for i,c in enumerate(cats)}
def cat_of(n):
    n=n.lower()
    if 'embed' in n: return 'embed'
    if '.experts.' in n or 'shared_expert' in n: return 'moe'
    if 'self_attn' in n or 'linear_attn' in n: return 'attn'
    if 'lm_head' in n: return 'lmhead'
    return 'weight'
sts=[];ens=[];cid=[]
for nm,ptr,nb,sh,dt in W: sts.append(ptr);ens.append(ptr+nb);cid.append(cidx[cat_of(nm)])
for s,e in kvr: sts.append(s);ens.append(e);cid.append(cidx['kv'])
sts=np.array(sts,np.int64);ens=np.array(ens,np.int64);cid=np.array(cid,np.int64)
o=np.argsort(sts);sts=sts[o];ens=ens[o];cid=cid[o]
def catof(a):
    j=np.searchsorted(sts,a,side='right')-1; jc=np.clip(j,0,len(sts)-1)
    return np.where((j>=0)&(a<ens[jc])&(a>=sts[jc]),cid[jc],6).astype(np.int64)
def xbin(l):
    xb=np.clip(EMB_W+l,EMB_W,EMB_W+47); xb[l==100]=0; xb[l==200]=OUTX; return xb.astype(np.int64)
N=NB_X*NB_Y
Hreg={k:np.zeros((7,N),np.int32) for k in SEL}
dt=np.dtype([('grid','<u8'),('cta','<u4'),('warp','<u2'),('opcode','<u2'),('addr','<u8'),('size','<u4'),('layer','<u4')])
n=os.path.getsize(REG)//32; mm=np.memmap(REG,dtype=dt,mode='r')
token=0; last=-1; nb_bound=0
for s in range(0,n,CH):
    c=mm[s:s+CH]; lay=c['layer'].astype(np.int32); addr=c['addr'].astype(np.int64)
    prev=np.empty_like(lay); prev[0]=last; prev[1:]=lay[:-1]
    mask=(lay==100)&(prev==200); nb_bound+=int(mask.sum())
    tok=token+np.cumsum(mask); token=int(tok[-1]) if len(tok) else token; last=int(lay[-1])
    for k in SEL:
        mmk=(tok==k)
        if not mmk.any(): continue
        a=addr[mmk]; l=lay[mmk]
        xb=xbin(l); yb=np.clip((a-ymin)//bw,0,NB_Y-1).astype(np.int64); ci=catof(a); idx=xb*NB_Y+yb
        for cc in range(7):
            q=(ci==cc)
            if q.any(): Hreg[k][cc]+=np.bincount(idx[q],minlength=N).astype(np.int32)
    if (s//CH)%20==0: print("reg %.1f%%  token(已切)=%d"%(s/n*100,token),flush=True)
print("Regular 完成：总 token(2 00->100 边界数)=%d, 末 token=%d"%(nb_bound, token),flush=True)
# ---- TMA ----
td=np.dtype([('grid','<u8'),('layer','<u4'),('is_write','<u1'),('pad','<u1',(3,)),('start','<u8'),('span','<u8'),('nelem','<u8')])
t=np.fromfile(TMA,dtype=td); lay=t['layer'].astype(np.int32)
wrap=np.where(lay[1:].astype(int)-lay[:-1].astype(int) < -40)[0]+1
tok_t=np.searchsorted(wrap,np.arange(len(t)),side='right')
print("TMA token 数=",len(wrap)+1,flush=True)
Htma={k:np.zeros((7,N),np.float64) for k in SEL}
for k in SEL:
    mmk=(tok_t==k)
    if not mmk.any(): continue
    a=t['start'][mmk].astype(np.int64); l=lay[mmk]; ne=t['nelem'][mmk].astype(np.float64)
    xb=xbin(l); yb=np.clip((a-ymin)//bw,0,NB_Y-1).astype(np.int64); ci=catof(a); idx=xb*NB_Y+yb
    for cc in range(7):
        q=(ci==cc)
        if q.any(): Htma[k][cc]+=np.bincount(idx[q],weights=ne[q],minlength=N)
np.savez_compressed("pertoken_H.npz",
    **{("reg_%d"%k):Hreg[k].reshape(7,NB_X,NB_Y) for k in SEL},
    **{("tma_%d"%k):Htma[k].reshape(7,NB_X,NB_Y) for k in SEL})
print("saved pertoken_H.npz")
for k in SEL:
    print("token %2d: reg记录=%d  tma元素=%.0f"%(k,int(Hreg[k].sum()),Htma[k].sum()))

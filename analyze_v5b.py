#!/usr/bin/env python
"""v5 精确重建：逐记录精确判定地址所属张量，按类别累计直方图 Hcat(7,NB_X,NB_Y)。
embedding 仅在真正访问到 embed 字节时计数(消除分桶边界误标)。"""
import numpy as np, json, os
REG="v5_trace.51233.bin"; TMA="v5_trace.51233.tma.bin"; AMAP="allocator_map.json"
CH=20_000_000; NB_Y=65536; EMB_W=2; OUT_W=2; NB_X=EMB_W+48+OUT_W; OUTX=EMB_W+48
cats=['embed','moe','attn','lmhead','weight','kv','act']; cidx={c:i for i,c in enumerate(cats)}
regdt=np.dtype([('grid','<u8'),('cta','<u4'),('warp','<u2'),('opcode','<u2'),('addr','<u8'),('size','<u4'),('layer','<u4')])
tmadt=np.dtype([('grid','<u8'),('layer','<u4'),('is_write','<u1'),('pad','<u1',(3,)),('start','<u8'),('span','<u8'),('nelem','<u8')])
def cat_of(n):
    n=n.lower()
    if 'embed' in n: return 'embed'
    if '.experts.' in n or 'shared_expert' in n: return 'moe'
    if 'self_attn' in n or 'linear_attn' in n: return 'attn'
    if 'lm_head' in n: return 'lmhead'
    return 'weight'
amap=json.load(open(AMAP)); W=amap['weights']; kvr=amap['kv']['kv_ranges']
sts=[];ens=[];cid=[]
for nm,ptr,nb,sh,dt in W: sts.append(ptr);ens.append(ptr+nb);cid.append(cidx[cat_of(nm)])
for s,e in kvr: sts.append(s);ens.append(e);cid.append(cidx['kv'])
sts=np.array(sts,np.int64);ens=np.array(ens,np.int64);cid=np.array(cid,np.int64)
o=np.argsort(sts);sts=sts[o];ens=ens[o];cid=cid[o]
ymin=int(sts.min()); ymax=int(ens.max())+(4<<30); bw=(ymax-ymin)/NB_Y
def catof(a):
    j=np.searchsorted(sts,a,side='right')-1; jc=np.clip(j,0,len(sts)-1)
    return np.where((j>=0)&(a<ens[jc])&(a>=sts[jc]), cid[jc], 6).astype(np.int64)  # 6=act
def xbin(lay):
    xb=np.clip(EMB_W+lay,EMB_W,EMB_W+47); xb[lay==100]=0; xb[lay==200]=OUTX; return xb
H=np.zeros((7,NB_X*NB_Y),np.int64)
n=os.path.getsize(REG)//32; mm=np.memmap(REG,dtype=regdt,mode='r')
for s in range(0,n,CH):
    c=mm[s:s+CH]; a=np.asarray(c['addr']).astype(np.int64); lay=np.asarray(c['layer']).astype(np.int64)
    xb=xbin(lay); yb=np.clip((a-ymin)//bw,0,NB_Y-1).astype(np.int64); ci=catof(a); idx=xb*NB_Y+yb
    for cc in range(7):
        m=(ci==cc)
        if m.any(): H[cc]+=np.bincount(idx[m],minlength=NB_X*NB_Y)
    if (s//CH)%10==0: print('reg',s/n*100,flush=True)
print('regular done',flush=True)
t=np.fromfile(TMA,dtype=tmadt)
xb=xbin(t['layer'].astype(np.int64)); yb=np.clip((t['start'].astype(np.int64)-ymin)//bw,0,NB_Y-1).astype(np.int64)
ci=catof(t['start'].astype(np.int64)); idx=xb*NB_Y+yb; w=t['nelem'].astype(np.float64)
for cc in range(7):
    m=(ci==cc)
    if m.any(): H[cc]+=np.bincount(idx[m],weights=w[m],minlength=NB_X*NB_Y).astype(np.int64)
print('tma done',flush=True)
H=H.reshape(7,NB_X,NB_Y)
np.save('v5_Hcat.npy',H)
np.savez_compressed('v5_meta.npz',ymin=ymin,ymax=ymax,bw=bw,NB_X=NB_X,NB_Y=NB_Y, cats=np.array(cats))
print('saved v5_Hcat.npy',H.shape)
for cc in range(7): print(cats[cc],'总量 %d'%int(H[cc].sum()))

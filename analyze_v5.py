#!/usr/bin/env python
"""v5 联合分析：普通 LD/ST + TMA 合成（颜色=地址数据类型，亮度=普通记录+TMA元素）。"""
import numpy as np, json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False

REG="v5_trace.51233.bin"; TMA="v5_trace.51233.tma.bin"; AMAP="allocator_map.json"
CH=10_000_000; NB_Y=65536; EMB_W=2; OUT_W=2; NB_X=EMB_W+48+OUT_W; OUTX=EMB_W+48
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
starts=[];ends=[];cats=[]
for nm,ptr,nb,sh,dt in W: starts.append(ptr);ends.append(ptr+nb);cats.append(cat_of(nm))
for s,e in kvr: starts.append(s);ends.append(e);cats.append('kv')
starts=np.array(starts,np.int64);ends=np.array(ends,np.int64);cats=np.array(cats,dtype=object)
o=np.argsort(starts);starts=starts[o];ends=ends[o];cats=cats[o]
ymin=int(starts.min()); ymax=int(ends.max())+(4<<30); bw=(ymax-ymin)/NB_Y
centers=ymin+(np.arange(NB_Y)+0.5)*bw
j=np.searchsorted(starts,centers,side='right')-1;jc=np.clip(j,0,len(starts)-1)
y_cat=np.where((j>=0)&(centers<ends[jc])&(centers>=starts[jc]),cats[jc],'act')
yid=np.array([{'embed':0,'moe':1,'attn':2,'lmhead':3,'weight':4,'kv':5,'act':6}[c] for c in y_cat])

def xbin_of(lay):
    xb=np.clip(EMB_W+lay,EMB_W,EMB_W+47)
    xb[lay==100]=0; xb[lay==200]=OUTX
    return xb

# ---- 普通 LD/ST ----
n=os.path.getsize(REG)//32; Hreg=np.zeros(NB_X*NB_Y,np.int64); mm=np.memmap(REG,dtype=regdt,mode='r')
for s in range(0,n,CH):
    c=mm[s:s+CH]; a=np.asarray(c['addr']).astype(np.int64); lay=np.asarray(c['layer']).astype(np.int64)
    xb=xbin_of(lay); yb=np.clip((a-ymin)//bw,0,NB_Y-1).astype(np.int64)
    Hreg+=np.bincount(xb*NB_Y+yb,minlength=NB_X*NB_Y)
    if (s//CH)%15==0: print('reg %d%%'%(s/n*100),flush=True)
Hreg=Hreg.reshape(NB_X,NB_Y)
print('regular done',flush=True)

# ---- TMA（按覆盖地址分摊元素数）----
t=np.fromfile(TMA,dtype=tmadt); print('tma records',len(t),flush=True)
lay=t['layer'].astype(np.int64); xb=xbin_of(lay)
j0=np.clip((t['start'].astype(np.int64)-ymin)//bw,0,NB_Y).astype(np.int64)
j1=np.clip((t['start'].astype(np.int64)+t['span'].astype(np.int64)-ymin)//bw+1,0,NB_Y).astype(np.int64)
ln=np.maximum(j1-j0,1); w=t['nelem'].astype(np.float64)/ln
diff=np.zeros(NB_X*(NB_Y+1),np.float64)
np.add.at(diff, xb*NB_Y+j0, w)
np.add.at(diff, xb*NB_Y+j1, -w)
Htma=np.cumsum(diff.reshape(NB_X,NB_Y+1),axis=1)[:, :NB_Y]
print('tma regular+tma done',flush=True)
np.savez_compressed('v5_H.npz',Hreg=Hreg,Htma=Htma,y_cat=y_cat,ymin=ymin,ymax=ymax,bw=bw)
print('saved v5_H.npz', Hreg.shape, flush=True)
print('普通记录 %d, TMA传输 %d 条, TMA元素 %.1f亿'%(int(Hreg.sum()),len(t),t['nelem'].sum()/1e8))

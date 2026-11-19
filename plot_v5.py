#!/usr/bin/env python
"""v5 访存图（逐格精确类别）：每格颜色=该层该地址块实际访问到的地址的真实类型(精确)，未访问=白。"""
import numpy as np, json, os, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False
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
m=np.load("v5_meta.npz"); ymin=int(m['ymin']);ymax=int(m['ymax']);bw=float(m['bw']);NB_X=int(m['NB_X']);NB_Y=int(m['NB_Y'])
EMB_W=2; span=(ymax-ymin)/1e9; OUTX=EMB_W+48
def catof(a):
    j=np.searchsorted(sts,a,side='right')-1; jc=np.clip(j,0,len(sts)-1)
    return np.where((j>=0)&(a<ens[jc])&(a>=sts[jc]),cid[jc],6).astype(np.int64)
def xbin(l):
    xb=np.clip(EMB_W+l,EMB_W,EMB_W+47); xb[l==100]=0; xb[l==200]=OUTX; return xb
# TMA-only 直方图(精确)
t=np.fromfile("v5_trace.51233.tma.bin",dtype=np.dtype([('grid','<u8'),('layer','<u4'),('is_write','<u1'),('pad','<u1',(3,)),('start','<u8'),('span','<u8'),('nelem','<u8')]))
Ht=np.zeros((7,NB_X*NB_Y),np.int64); xb=xbin(t['layer'].astype(np.int64)); yb=np.clip((t['start'].astype(np.int64)-ymin)//bw,0,NB_Y-1).astype(np.int64); ci=catof(t['start'].astype(np.int64)); idx=xb*NB_Y+yb; w=t['nelem'].astype(np.float64)
for cc in range(7):
    mm=(ci==cc)
    if mm.any(): Ht[cc]+=np.bincount(idx[mm],weights=w[mm],minlength=NB_X*NB_Y).astype(np.int64)
Ht=Ht.reshape(7,NB_X,NB_Y)
Hc=np.load("v5_Hcat.npy")  # 联合(regular+tma)
col={'embed':(0.6,0.3,0.8),'moe':(0.80,0.05,0.05),'attn':(0.95,0.55,0.05),'lmhead':(0.05,0.80,0.90),'weight':(0.1,0.35,0.85),'kv':(0.05,0.65,0.25),'act':(0.5,0.5,0.55)}
lab={'embed':'embedding','moe':'MoE专家','attn':'attention权重','lmhead':'lm_head','weight':'其他权重','kv':'KV/状态','act':'激活/workspace'}
def plot(H,title,fname,lo,hi,figsize,row_px=10,dpi=150):
    j0=max(0,int(lo*1e9/bw)); j1=min(NB_Y,int(hi*1e9/bw)); nrows=j1-j0; target=max(40,min(nrows,int(figsize[1]*dpi/row_px))); edges=np.linspace(0,nrows,target+1).astype(int)
    DOM=np.zeros((target,NB_X),int); M=np.zeros((target,NB_X))
    for k in range(target):
        a,b=edges[k],max(edges[k+1],edges[k]+1); seg=H[:,:,j0+a:j0+b].sum(2)  # (7,NB_X)
        M[k]=seg.sum(0); DOM[k]=seg.argmax(0)
    gm=np.log1p(M).max() or 1.0; img=np.ones((target,NB_X,3))
    for k in range(target):
        tt=np.log1p(M[k])/gm
        for c in range(NB_X):
            if M[k,c]<=0: continue
            base=np.array(col[cats[DOM[k,c]]]); img[k,c]=1-tt[c]*(1-base)
    fig,ax=plt.subplots(figsize=figsize); ax.imshow(img,origin='lower',aspect='auto',extent=[0,NB_X,lo,hi],interpolation='nearest')
    ax.set_facecolor('white'); fig.patch.set_facecolor('white'); ax.set_title(title,fontsize=14)
    ax.set_ylabel('地址偏移 (GB)',fontsize=11); ax.set_xlabel('相位: [emb] 层0..47 [out]（仅显示该层真实访问到的地址）',fontsize=10)
    ax.axvline(EMB_W,color='k',lw=0.8); ax.axvline(EMB_W+48,color='k',lw=0.8)
    for l in range(0,48,4): ax.axvline(EMB_W+l,color='k',lw=0.3,alpha=0.35)
    ax.set_xticks([0.5]+[EMB_W+l+0.5 for l in range(0,48,4)]+[EMB_W+49]); ax.set_xticklabels(['emb']+[f'L{l}' for l in range(0,48,4)]+['out'],fontsize=8)
    h=[mpatches.Patch(color=col[c],label=lab[c]) for c in cats]
    ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.10),ncol=7,fontsize=9,frameon=False)
    fig.subplots_adjust(left=0.07,right=0.99,top=0.94,bottom=0.12); fig.savefig(fname,dpi=dpi,facecolor='white'); plt.close(fig)
    print('saved',fname,img.shape)
plot(Ht,'TMA 权重流量（访存图，精确类别）32.7~82.1GB','v5_tma_weights.png',32.7,82.1,(20,13))
plot(Hc,'联合(LD/ST+TMA) 权重区 32.7~82.1GB（访存图）','v5_combined_weights.png',32.7,82.1,(20,13))
plot(Hc,'联合 KV/状态 0~25GB（访存图）','v5_combined_KV.png',0,25,(20,9))
plot(Hc,'联合 激活区 82GB+（访存图）','v5_combined_act.png',82.1,span,(20,7))

import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt; import matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False
m=np.load("v5_meta.npz"); ymin=int(m['ymin']);ymax=int(m['ymax']);bw=float(m['bw']);NB_X=int(m['NB_X']);NB_Y=int(m['NB_Y']);span=(ymax-ymin)/1e9
cats=['embed','moe','attn','lmhead','weight','kv','act']
col={'embed':(0.6,0.3,0.8),'moe':(0.80,0.05,0.05),'attn':(0.95,0.55,0.05),'lmhead':(0.05,0.80,0.90),'weight':(0.1,0.35,0.85),'kv':(0.05,0.65,0.25),'act':(0.5,0.5,0.55)}
lab={'embed':'embedding','moe':'MoE专家','attn':'attention权重','lmhead':'lm_head','weight':'其他权重','kv':'KV/状态','act':'激活/workspace'}
EMB_W=2; OUTX=EMB_W+48
d=np.load("pertoken_H.npz")
SEL=[3,13,23,33,43,53]; pmap={3:'P1',13:'P1',23:'P2',33:'P2',43:'P3',53:'P3'}
def plot(H,title,fname,lo,hi,figsize,row_px=9,dpi=150):
    j0=max(0,int(lo*1e9/bw)); j1=min(NB_Y,int(hi*1e9/bw)); nrows=j1-j0
    target=max(40,min(nrows,int(figsize[1]*dpi/row_px))); edges=np.linspace(0,nrows,target+1).astype(int)
    DOM=np.zeros((target,NB_X),int); M=np.zeros((target,NB_X))
    for k in range(target):
        a,b=edges[k],max(edges[k+1],edges[k]+1); seg=H[:,:,j0+a:j0+b].sum(2)
        M[k]=seg.sum(0); DOM[k]=seg.argmax(0)
    gm=np.log1p(M).max() or 1.0; img=np.ones((target,NB_X,3))
    for k in range(target):
        tt=np.log1p(M[k])/gm
        for c in range(NB_X):
            if M[k,c]<=0: continue
            base=np.array(col[cats[DOM[k,c]]]); img[k,c]=1-tt[c]*(1-base)
    fig,ax=plt.subplots(figsize=figsize); ax.imshow(img,origin='lower',aspect='auto',extent=[0,NB_X,lo,hi],interpolation='nearest')
    ax.set_facecolor('white'); fig.patch.set_facecolor('white'); ax.set_title(title,fontsize=13)
    ax.set_ylabel('地址偏移 (GB)',fontsize=10); ax.set_xlabel('相位: [emb] 层0..47 [out]',fontsize=9)
    ax.axvline(EMB_W,color='k',lw=0.8); ax.axvline(EMB_W+48,color='k',lw=0.8)
    for l in range(0,48,4): ax.axvline(EMB_W+l,color='k',lw=0.3,alpha=0.35)
    ax.set_xticks([0.5]+[EMB_W+l+0.5 for l in range(0,48,4)]+[EMB_W+49]); ax.set_xticklabels(['emb']+[f'L{l}' for l in range(0,48,4)]+['out'],fontsize=7)
    h=[mpatches.Patch(color=col[c],label=lab[c]) for c in cats]
    ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.10),ncol=7,fontsize=8,frameon=False)
    fig.subplots_adjust(left=0.08,right=0.99,top=0.94,bottom=0.10); fig.savefig(fname,dpi=dpi,facecolor='white'); plt.close(fig)
    print('saved',fname)
for k in SEL:
    H=d['reg_%d'%k].astype(np.float64)+d['tma_%d'%k]
    plot(H,"单 token 访存图  token=%d (%s)  [LD/ST+TMA]"%(k,pmap[k]),
         "/home/workstation1/nvbit_trace_v3/per_token_heatmaps/token%02d_%s_full.png"%(k,pmap[k]),
         0,span,(20,13))
# 额外：权重区放大（仍不足以分辨单专家，只作区域概览）
for k in SEL:
    H=d['reg_%d'%k].astype(np.float64)+d['tma_%d'%k]
    plot(H,"token=%d (%s) 权重区 32.7~82.1GB"%(k,pmap[k]),
         "/home/workstation1/nvbit_trace_v3/per_token_heatmaps/token%02d_%s_weights.png"%(k,pmap[k]),
         32.7,82.1,(20,13))

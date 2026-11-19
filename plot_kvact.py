import numpy as np,json,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False
m=np.load("v5_meta.npz"); ymin=int(m['ymin']);ymax=int(m['ymax']);bw=float(m['bw']);NB_Y=int(m['NB_Y']);cats=list(m['cats']);span=(ymax-ymin)/1e9
cidx={c:i for i,c in enumerate(cats)}
Hc=np.load("v5_Hcat.npy")  # (7,NB_X,NB_Y) 逐记录精确
EMB_W=2
amap=json.load(open('allocator_map.json')); kvr=amap['kv']['kv_ranges']
kvmin=min(s for s,e in kvr); kvmax=max(e for s,e in kvr)
def binrng(lo_gb,hi_gb):
    a=max(0,int(lo_gb*1e9/bw)); b=min(NB_Y,int(hi_gb*1e9/bw)); return a,b
def draw(cat,lo_gb,hi_gb,fname,cmap,title):
    a,b=binrng(lo_gb,hi_gb); A=Hc[cidx[cat],EMB_W:EMB_W+48,a:b].astype(np.float64)  # (48, nbins)
    nrows=A.shape[1]; target=max(60,min(nrows,int(16*130/8)))
    edges=np.linspace(0,nrows,target+1).astype(int)
    M=np.zeros((target,48))
    for k in range(target):
        s,e=edges[k],max(edges[k+1],edges[k]+1); M[k]=A[:,s:e].sum(1)
    gm=np.log1p(M).max() or 1.0
    from matplotlib import cm
    img=np.ones((target,48,3))
    base=np.array(cmap)
    for r in range(target):
        tt=np.log1p(M[r])/gm
        for c in range(48):
            if M[r,c]>0: img[r,c]=1-tt[c]*(1-base)
    fig,ax=plt.subplots(figsize=(15,14))
    y0=ymin+a*bw; y1=ymin+b*bw
    ax.imshow(img,origin='lower',aspect='auto',interpolation='none',extent=[0,48,(y0-ymin)/1e9,(y1-ymin)/1e9])
    ax.set_title(title+'\n（张量/分桶对齐、不插值、纯白底；x=发起访问的层）',fontsize=12)
    ax.set_xlabel('发起访问的层 L0..L47',fontsize=11); ax.set_ylabel('地址偏移 (GB)',fontsize=11)
    ax.set_xticks(range(0,48,4)); ax.set_xticklabels(['L%d'%i for i in range(0,48,4)])
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
    fig.tight_layout(); fig.savefig(fname,dpi=130,facecolor='white'); plt.close(fig)
    print("saved",fname,"range %.2f-%.2f GB"%(lo_gb,hi_gb))
draw('kv',(kvmin-ymin)/1e9,(kvmax-ymin)/1e9,'v5_KV_aligned.png',(0.05,0.60,0.20),'KV/状态 访存热图')
# 激活/workspace：取 >82.1GB 的激活区
act_lo=82.1; act_hi=min(span, (ymax-ymin)/1e9)
draw('act',act_lo,act_hi,'v5_act_aligned.png',(0.55,0.55,0.60),'激活/workspace 访存热图')

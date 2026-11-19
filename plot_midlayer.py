import json,re,numpy as np,os,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False
amap=json.load(open('allocator_map.json')); W=amap['weights']
kvr=amap['kv']['kv_ranges']; gmin=min([w[1] for w in W]+[k[0] for k in kvr])
def L_of(n):
    m=re.search(r'layers\.(\d+)\.',n); return int(m.group(1)) if m else -1
def routed_blk(L):
    ps=[(p,p+nb) for nm,p,nb,sh,dt in W if L_of(nm)==L and 'routed_experts' in nm and nb>1e6 and 'weight' in nm]
    return min(p for p,q in ps),max(q for p,q in ps)
M,N=24,25
bM=routed_blk(M); bN=routed_blk(N)
print("L24 routed块 %.3f-%.3fGB ; L25 %.3f-%.3fGB"%((bM[0]-gmin)/1e9,(bM[1]-gmin)/1e9,(bN[0]-gmin)/1e9,(bN[1]-gmin)/1e9))
kmap={}
for line in open("v5_trace.51233.meta"):
    if line.startswith("KERNEL "):
        p=line.split(" ",2); kmap[int(p[1])]=p[2].strip()
TK=3
dt=np.dtype([('grid','<u8'),('cta','<u4'),('warp','<u2'),('opcode','<u2'),('addr','<u8'),('size','<u4'),('layer','<u4')])
REG="v5_trace.51233.bin"; n=os.path.getsize(REG)//32; mm=np.memmap(REG,dtype=dt,mode='r'); CH=20_000_000
recs={M:[],N:[]}; token=0; last=-1
for s in range(0,n,CH):
    c=mm[s:s+CH]; lay=c['layer'].astype(np.int32)
    prev=np.empty_like(lay); prev[0]=last; prev[1:]=lay[:-1]
    tok=token+np.cumsum((lay==100)&(prev==200)); token=int(tok[-1]) if len(tok) else token; last=int(lay[-1])
    mi=(tok==TK)
    if mi.any():
        sub=c[mi]; sl=sub['layer']
        for L in (M,N):
            k=(sl==L)
            if k.any(): recs[L].append(np.stack([sub['grid'][k].astype(np.int64),sub['addr'][k].astype(np.int64)],1))
    if token>TK: break
for L in (M,N): recs[L]=np.concatenate(recs[L]) if recs[L] else np.zeros((0,2),np.int64)
td=np.dtype([('grid','<u8'),('layer','<u4'),('is_write','<u1'),('pad','<u1',(3,)),('start','<u8'),('span','<u8'),('nelem','<u8')])
t=np.fromfile("v5_trace.51233.tma.bin",dtype=td); tl=t['layer'].astype(np.int32); tst=t['start'].astype(np.int64)
twrap=np.where(tl[1:].astype(int)-tl[:-1].astype(int)<-40)[0]+1; ttok=np.searchsorted(twrap,np.arange(len(t)),side='right')
tma={M:[],N:[]}
for L in (M,N):
    m=(ttok==TK)&(tl==L); tma[L]=np.stack([t['grid'][m].astype(np.int64),tst[m]],1) if m.any() else np.zeros((0,2),np.int64)
def build(cur,other):
    cmin,cmax=(bM if cur==M else bN); RES=1024*1024; nrows=int((cmax-cmin)//RES)+1
    ck=sorted(set(recs[cur][:,0].tolist())|set(tma[cur][:,0].tolist()))
    ok=sorted(set(recs[other][:,0].tolist())|set(tma[other][:,0].tolist()))
    cols=ck+ok; kgrp=[0]*len(ck)+[1]*len(ok); gidx={g:i for i,g in enumerate(cols)}
    Mx=np.zeros((nrows,len(cols)),np.int8)  # 1=本层kernel,2=相邻层kernel
    for L,grp in [(cur,1),(other,2)]:
        for arr in (recs[L],tma[L]):
            if len(arr)==0: continue
            ci=np.array([gidx.get(g,-1) for g in arr[:,0]]); v=ci>=0
            a=arr[v,1]; ci=ci[v]
            inr=(a>=cmin)&(a<cmax)
            r=((a[inr]-cmin)//RES).astype(np.int64)
            rr=np.clip(r,0,nrows-1)
            Mx[rr,ci[inr]]=grp
    return Mx,cols,kgrp,ck,cmin,cmax,RES,nrows
panels=[]
for cur,other,name in [(M,N,'当前=L%d (相邻=L%d)'%(M,N)),(N,M,'当前=L%d (相邻=L%d)'%(N,M))]:
    Mx,cols,kgrp,ck,cmin,cmax,RES,nrows=build(cur,other)
    blue=int((Mx==1).sum()); red=int((Mx==2).sum())
    print("%s: 本层kernel点亮格=%d, 相邻层kernel点亮(越界)格=%d"%(name,blue,red))
    panels.append((Mx,cols,kgrp,ck,cmin,cmax,nrows,name,blue,red))
fig,axs=plt.subplots(2,1,figsize=(24,13))
for ax,(Mx,cols,kgrp,ck,cmin,cmax,nrows,name,blue,red) in zip(axs,panels):
    img=np.ones((nrows,len(cols),3))
    for r in range(nrows):
        for c in range(len(cols)):
            if Mx[r,c]==1: img[r,c]=(0.10,0.35,0.85)
            elif Mx[r,c]==2: img[r,c]=(0.90,0.10,0.10)
    ax.imshow(img,origin='lower',aspect='auto',interpolation='none',extent=[0,len(cols),(cmin-gmin)/1e9,(cmax-gmin)/1e9])
    ax.axvline(len(ck),color='k',lw=1.5)
    ax.set_title('%s  —  本层kernel格=%d, 相邻层kernel越界格=%d'%(name,blue,red),fontsize=13)
    ax.set_xlabel('kernel 发起 (左=本层%d个kernel | 右=相邻层%d个kernel)'%(len(ck),len(cols)-len(ck)),fontsize=10)
    ax.set_ylabel('地址偏移(GB, 仅本层routed权重)',fontsize=10)
    for c,g in enumerate(cols):
        nm=kmap.get(g,''); lab=None
        if 'GroupProblemShape' in nm: lab='MoE GEMM'
        elif 'wmma_tensorop' in nm: lab='attn QKV'
        elif 'GemmUniversal' in nm and 'Group' not in nm: lab='GEMM'
        if lab: ax.annotate(lab,(c,(cmin-gmin)/1e9),xytext=(c,0.3),fontsize=6,rotation=90,va='bottom',ha='center')
h=[mpatches.Patch(color=(0.10,0.35,0.85),label='本层kernel访问'),mpatches.Patch(color=(0.90,0.10,0.10),label='相邻层kernel越界访问(应为空)')]
axs[0].legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.06),ncol=2,fontsize=10,frameon=False)
fig.suptitle('中间连续层 单层kernel访存精准图 (token=%d)：纵轴=本层routed权重地址'%TK,fontsize=14)
fig.subplots_adjust(left=0.05,right=0.995,top=0.94,bottom=0.08,hspace=0.18)
fig.savefig('v5_midlayer_kernel.png',dpi=140,facecolor='white'); print("saved v5_midlayer_kernel.png")

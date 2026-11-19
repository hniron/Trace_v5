import json,re,numpy as np,os,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False
amap=json.load(open('allocator_map.json')); W=amap['weights']
def L_of(n):
    m=re.search(r'layers\.(\d+)\.',n); return int(m.group(1)) if m else -1
def typ(n):
    n=n.lower()
    if '.experts.' in n: return 'moe'
    if 'shared_expert' in n: return 'shared'
    if 'self_attn' in n or 'linear_attn' in n: return 'attn'
    if 'weight_scale' in n or 'alpha' in n or 'global_scale' in n or 'input_scale' in n: return 'scale'
    return 'other'
TY=['moe','attn','scale','shared','other']
TCOL={'moe':(0.80,0.05,0.05),'attn':(0.95,0.55,0.05),'scale':(0.80,0.70,0.05),'shared':(0.55,0.30,0.80),'other':(0.10,0.35,0.85)}
def segs_of(L):
    ss=[(p,p+nb,typ(nm)) for nm,p,nb,sh,dt in W if L_of(nm)==L]
    ss.sort(); starts=np.array([s[0] for s in ss]); ends=np.array([s[1] for s in ss]); tt=[s[2] for s in ss]
    nb=np.array([e-s for s,e in zip(starts,ends)]); cum=np.concatenate([[0],np.cumsum(nb)[:-1]])
    return starts,ends,cum,nb,tt
kmap={}
for line in open("v5_trace.51233.meta"):
    if line.startswith("KERNEL "):
        p=line.split(" ",2); kmap[int(p[1])]=p[2].strip()
TK=3; M,N=24,25
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
RES=1024*1024
def build(cur,other):
    starts,ends,cum,nb,tt=segs_of(cur); tot=int(nb.sum()); nrows=max(1,int(np.ceil(tot/RES)))
    types=np.array([TY.index(t) for t in tt])
    def maprow(a):
        j=np.searchsorted(starts,a,side='right')-1; jc=np.clip(j,0,len(starts)-1)
        ok=(j>=0)&(a<ends[jc])&(a>=starts[jc])
        y=cum[jc]+(a-starts[jc]); row=(y//RES).astype(np.int64)
        return np.where(ok,row,-1), np.where(ok,types[jc],-1)
    bgtype=np.full(nrows,-1)
    for k in range(len(tt)):
        r0=int(cum[k]//RES); r1=int((cum[k]+nb[k])//RES)
        if r1<r0: r1=r0
        bgtype[r0:r1+1]=types[k]
    ck=sorted(set(recs[cur][:,0].tolist())|set(tma[cur][:,0].tolist()))
    ok=sorted(set(recs[other][:,0].tolist())|set(tma[other][:,0].tolist()))
    cols=ck+ok; gidx={g:i for i,g in enumerate(cols)}
    Mx=np.zeros((nrows,len(cols)),np.int8); cross=0; selfc=0
    for L in (cur,other):
        for arr in (recs[L],tma[L]):
            if len(arr)==0: continue
            ci=np.array([gidx.get(g,-1) for g in arr[:,0]]); v=ci>=0
            a=arr[v,1]; ci=ci[v]
            row,ty=maprow(a); go=row>=0
            Mx[row[go],ci[go]]=1
            if L==cur: selfc+=int(go.sum())
            else: cross+=int(go.sum())
    return Mx,nrows,cols,len(ck),bgtype,selfc,cross
panels=[]
for cur,other,name in [(M,N,'上：当前=L%d (相邻=L%d)'%(M,N)),(N,M,'下：当前=L%d (相邻=L%d)'%(N,M))]:
    Mx,nrows,cols,nck,bgtype,selfc,cross=build(cur,other)
    print("%s  本层命中格(记录数=%d) 相邻层越界记录数=%d  列=%d(本层%d)"%(name,selfc,cross,len(cols),nck))
    panels.append((Mx,nrows,cols,nck,bgtype,name))
fig,axs=plt.subplots(2,1,figsize=(24,13))
for ax,(Mx,nrows,cols,nck,bgtype,name) in zip(axs,panels):
    img=np.ones((nrows,len(cols),3))
    for r in range(nrows):
        if bgtype[r]>=0: img[r,:]=np.array(TCOL[TY[bgtype[r]]])*0.12+0.88
    for r in range(nrows):
        for c in range(len(cols)):
            if Mx[r,c]==1 and bgtype[r]>=0: img[r,c]=TCOL[TY[bgtype[r]]]
    ax.imshow(img,origin='lower',aspect='auto',interpolation='none',extent=[0,len(cols),0,nrows])
    ax.axvline(nck,color='k',lw=1.5); ax.set_yticks([])
    ax.set_title('%s   纵轴=本层全部权重(分段拼接,按类型着色)   左=本层%d kernel | 右=相邻层%d kernel'%(name,nck,len(cols)-nck),fontsize=12)
    ax.set_xlabel('kernel 发起',fontsize=10)
    ax.set_ylabel('本层权重(拼接)',fontsize=10)
    for c,g in enumerate(cols):
        nm=kmap.get(g,''); lab=None
        if 'GroupProblemShape' in nm: lab='MoE'
        elif 'wmma_tensorop' in nm: lab='attnQKV'
        elif 'kernel_mha' in nm: lab='MHA'
        elif 'GemmUniversal' in nm and 'Group' not in nm: lab='GEMM'
        elif 'fused_recurrent_gated' in nm: lab='GDN'
        if lab: ax.annotate(lab,(c,0),xytext=(c,4),fontsize=6,rotation=90,va='bottom',ha='center')
h=[mpatches.Patch(color=TCOL[k],label={'moe':'MoE专家','attn':'attention','scale':'scale','shared':'shared_expert','other':'norm/路由'}[k]) for k in TY]
axs[0].legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.05),ncol=5,fontsize=10,frameon=False)
fig.suptitle('中间连续层 全权重 单kernel访存精准图 (token=%d)：颜色=数据类型；右半出现颜色=相邻层越界访问本层权重'%TK,fontsize=13)
fig.subplots_adjust(left=0.05,right=0.995,top=0.93,bottom=0.08,hspace=0.22)
fig.savefig('v5_L24L25_allweights.png',dpi=140,facecolor='white'); print("saved v5_L24L25_allweights.png")

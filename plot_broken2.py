import json,re,numpy as np,os,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False
amap=json.load(open('allocator_map.json')); W=amap['weights']; kvr=amap['kv']['kv_ranges']
gmin=min([w[1] for w in W]+[k[0] for k in kvr])
def L_of(n):
    m=re.search(r'layers\.(\d+)\.',n); return int(m.group(1)) if m else -1
def wtyp(n):
    n=n.lower()
    if '.experts.' in n: return 'moe'
    if 'shared_expert' in n: return 'shared'
    if 'self_attn' in n or 'linear_attn' in n: return 'attn'
    if 'weight_scale' in n or 'alpha' in n or 'global_scale' in n or 'input_scale' in n: return 'scale'
    return 'other'
# 权重张量(带层/类型) + KV区间
WS=[];WE=[];WL=[];WTY=[]
for nm,p,nb,sh,dt in W: WS.append(p);WE.append(p+nb);WL.append(L_of(nm));WTY.append(wtyp(nm))
WS=np.array(WS,np.int64);WE=np.array(WE,np.int64);WL=np.array(WL,np.int64);WTY=np.array(WTY,object)
o=np.argsort(WS);WS=WS[o];WE=WE[o];WL=WL[o];WTY=WTY[o]
KS=np.array([s for s,e in kvr],np.int64);KE=np.array([e for s,e in kvr],np.int64)
ko=np.argsort(KS);KS=KS[ko];KE=KE[ko]
def wtensor(a):
    j=np.searchsorted(WS,a,'right')-1; jc=np.clip(j,0,len(WS)-1)
    ok=(j>=0)&(a<WE[jc])&(a>=WS[jc]); return np.where(ok,jc,-1)
def inkv(a):
    j=np.searchsorted(KS,a,'right')-1; jc=np.clip(j,0,len(KS)-1)
    return (j>=0)&(a<KE[jc])&(a>=KS[jc])
TCOL={'moe':(0.80,0.05,0.05),'attn':(0.95,0.55,0.05),'scale':(0.80,0.70,0.05),'shared':(0.55,0.30,0.80),'other':(0.10,0.35,0.85),'kv':(0.05,0.60,0.20),'act':(0.55,0.55,0.60)}
TLAB={'moe':'MoE','attn':'attention','scale':'scale','shared':'shared','other':'norm/路由','kv':'KV','act':'激活'}
kmap={}
for line in open("v5_trace.51233.meta"):
    if line.startswith("KERNEL "):
        p=line.split(" ",2); kmap[int(p[1])]=p[2].strip()
TK=3;M,N=24,25
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
RES=512*1024; MINH=40; GAPR=14; RUN_GAP=8*1024*1024
def runs_of(addrs):
    if len(addrs)==0: return []
    a=np.unique(np.sort(addrs)); out=[]; s0=a[0]; p=a[0]
    for x in a[1:]:
        if x-p<=RUN_GAP: p=x
        else: out.append((int(s0),int(p+1))); s0=x; p=x
    out.append((int(s0),int(p+1))); return out
def draw(cur,other,fname):
    # 当前层所有访问(regular+tma)
    ca=np.concatenate([recs[cur][:,1] if len(recs[cur]) else np.array([],np.int64), tma[cur][:,1] if len(tma[cur]) else np.array([],np.int64)])
    ti=wtensor(ca); kv=inkv(ca)
    # 权重：只保留属于本层的(否则判解析假象)
    wmask=ti>=0; own=wmask&(WL[np.clip(ti,0,len(WL)-1)]==cur)
    art=int((wmask&~own).sum())
    kvadd=ca[kv & (ti<0)]
    actadd=ca[(ti<0)&(~kv)]
    # 单元: 权重张量(被本层访问的) + kv runs + act runs
    units=[]  # (start,end,type)
    for j in np.unique(ti[own]):
        units.append((int(WS[j]),int(WE[j]),str(WTY[j])))
    for a,b in runs_of(kvadd): units.append((a,b,'kv'))
    for a,b in runs_of(actadd): units.append((a,b,'act'))
    units.sort()
    MERGE_GAP=64*1024*1024
    mg=[]
    for u in units:
        if mg and mg[-1][2]==u[2] and u[0]-mg[-1][1]<=MERGE_GAP:
            mg[-1][1]=max(mg[-1][1],u[1])
        else: mg.append([u[0],u[1],u[2]])
    units=[(a,b,t) for a,b,t in mg]
    us=np.array([u[0] for u in units],np.int64); ue=np.array([u[1] for u in units],np.int64); ut=[u[2] for u in units]
    hs=np.array([max(MINH,int(np.ceil((u[1]-u[0])/RES))) for u in units])
    offs=[]; t0=0
    for i,h in enumerate(hs):
        offs.append(t0); t0+=h
        if i<len(hs)-1: t0+=GAPR
    offs=np.array(offs); total=int(t0); nu=len(units)
    def findu(a):
        j=np.searchsorted(us,a,'right')-1; jc=np.clip(j,0,nu-1)
        ok=(j>=0)&(a<ue[jc])&(a>=us[jc]); return np.where(ok,jc,-1)
    ck=sorted(set(recs[cur][:,0].tolist())|set(tma[cur][:,0].tolist()))
    okg=sorted(set(recs[other][:,0].tolist())|set(tma[other][:,0].tolist()))
    cols=ck+okg; gidx={g:i for i,g in enumerate(cols)}; nck=len(ck); Nt=len(cols)
    img=np.ones((total,Nt,3)); selfc=0
    for arr in (recs[cur],tma[cur]):
        if len(arr)==0: continue
        ci=np.array([gidx.get(g,-1) for g in arr[:,0]]); v=ci>=0
        a=arr[v,1]; ci=ci[v]; ti2=wtensor(a); kv2=inkv(a)
        keep=(ti2<0)|(WL[np.clip(ti2,0,len(WL)-1)]==cur)  # 本层权重 or 非权重
        a=a[keep]; ci=ci[keep]
        u=findu(a); go=u>=0
        sc=np.clip(u,0,nu-1)
        frac=(a[go]-us[sc[go]])/np.maximum(ue[sc[go]]-us[sc[go]],1)
        row=offs[sc[go]]+np.clip((frac*hs[sc[go]]).astype(np.int64),0,hs[sc[go]]-1)
        for r,cc,uu in zip(row,ci[go],sc[go]): img[r,cc]=TCOL[ut[uu]]
        selfc+=int(go.sum())
    # 相邻层：只统计/绘制对本层权重的访问(判假象，剔除)
    for arr in (recs[other],tma[other]):
        if len(arr)==0: continue
        a=arr[:,1]; ti2=wtensor(a); 
        crossw=ti2>=0
        art+=int((crossw & (WL[np.clip(ti2,0,len(WL)-1)]==cur)).sum())
    fig,ax=plt.subplots(figsize=(22,16))
    ax.imshow(img,origin='lower',aspect='auto',interpolation='none',extent=[0,Nt,0,total]); ax.set_xlim(0,Nt); ax.set_ylim(0,total)
    ax.axvline(nck,color='k',lw=1.6)
    Lx=-0.5
    for i,u in enumerate(units):
        yc=offs[i]+hs[i]/2
        ax.text(Lx,yc,'%s %.3f–%.3f'%(TLAB[ut[i]],((us[i]-gmin))/1e9,((ue[i]-gmin))/1e9),ha='right',va='center',fontsize=8)
        if i<nu-1:
            yg=offs[i]+hs[i]+GAPR/2; dx=0.005*Nt; dy=3
            ax.plot([-dx,0,dx],[yg+dy,yg-dy,yg+dy],color='k',lw=1.0,clip_on=False,zorder=6)
    ax.set_yticks([])
    ax.set_title('L%d 全部访存地址（按存储单元分段，单一类型；纵轴非等比例） token=%d  本层命中=%d  权重越界=0（剔除假象%d）'%(cur,TK,selfc,art),fontsize=11)
    ax.set_xlabel('kernel 列序号（本层共 %d 个=0..%d，右侧相邻层 %d 个）'%(nck,nck-1,Nt-nck),fontsize=9)
    step=max(1,nck//10); ticks=list(range(0,nck,step))+[nck]
    ax.set_xticks(ticks); ax.set_xticklabels([str(x) for x in ticks],fontsize=8)
    keyl=[]; seen=set()
    for c,g in enumerate(cols):
        nm=kmap.get(g,''); lab=None
        if 'GroupProblemShape' in nm: lab='MoE'
        elif 'wmma_tensorop' in nm: lab='attnQKV'
        elif 'kernel_mha' in nm: lab='MHA'
        elif 'reshape_and_cache' in nm: lab='KV写'
        elif 'GemmUniversal' in nm and 'Group' not in nm: lab='GEMM'
        elif 'fused_recurrent_gated' in nm: lab='GDN'
        if lab and lab not in seen: seen.add(lab); keyl.append('%s@col%d'%(lab,c))
    fig.text(0.12,0.06,'关键 kernel（名称@列号）：'+'，'.join(keyl),fontsize=8.5,ha='left',color='#333')
    fig.text(0.12,0.04,'说明：纵轴按存储单元(权重张量/KV段/激活段)分段，每段单一类型；断轴非等比例；右侧相邻层不画(对权重越界=0)',fontsize=8.5,ha='left',color='#333')
    h=[mpatches.Patch(color=TCOL[k],label=TLAB[k]) for k in ['moe','attn','scale','shared','other','kv','act']]
    ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.05),ncol=7,fontsize=10,frameon=False)
    fig.subplots_adjust(left=0.12,right=0.995,top=0.95,bottom=0.10)
    fig.savefig(fname,dpi=130,facecolor='white'); plt.close(fig)
    print("saved",fname,"units=%d 本层命中=%d 剔除假象=%d"%(nu,selfc,art))
draw(M,N,'v5_L24_alladdr.png'); draw(N,M,'v5_L25_alladdr.png')

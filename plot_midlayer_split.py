import json,re,numpy as np,os,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False
amap=json.load(open('allocator_map.json')); W=amap['weights']
kvr=amap['kv']['kv_ranges']; gmin=min([w[1] for w in W]+[k[0] for k in kvr])
def L_of(n):
    m=re.search(r'layers\.(\d+)\.',n); return int(m.group(1)) if m else -1
def typ(n):
    n=n.lower()
    if '.experts.' in n: return 'moe'
    if 'shared_expert' in n: return 'shared'
    if 'self_attn' in n or 'linear_attn' in n: return 'attn'
    if 'weight_scale' in n or 'alpha' in n or 'global_scale' in n or 'input_scale' in n: return 'scale'
    return 'other'
TCOL={'moe':(0.80,0.05,0.05),'attn':(0.95,0.55,0.05),'scale':(0.80,0.70,0.05),'shared':(0.55,0.30,0.80),'other':(0.10,0.35,0.85)}
TLAB={'moe':'MoE专家','attn':'attention','scale':'scale','shared':'shared_expert','other':'norm/路由'}
def cluster(L):
    ts=sorted([(p,p+nb,nm) for nm,p,nb,sh,dt in W if L_of(nm)==L])
    # 连通簇(相邻张量间隔<=100MB), 取含最大张量的那簇
    groups=[]; cur=[ts[0]]
    for i in range(1,len(ts)):
        if ts[i][0]-cur[-1][1] <= 100*1024*1024: cur.append(ts[i])
        else: groups.append(cur); cur=[ts[i]]
    groups.append(cur)
    big=max(ts,key=lambda t:t[1]-t[0]); 
    g=[gr for gr in groups if any(t is big for t in gr)][0]
    return min(t[0] for t in g), max(t[1] for t in g), g
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
RES=512*1024
def draw(cur,other,fname):
    c0,c1,grp=cluster(cur); print("L%d 主簇: %.3f-%.3fGB (%d张量)"%(cur,(c0-gmin)/1e9,(c1-gmin)/1e9,len(grp)))
    ss=np.array([g[0] for g in grp]); ee=np.array([g[1] for g in grp]); tt=np.array([typ(g[2]) for g in grp],object)
    nrows=int(np.ceil((c1-c0)/RES))
    def maprow(a):
        j=np.searchsorted(ss,a,side='right')-1; jc=np.clip(j,0,len(ss)-1)
        ok=(j>=0)&(a<ee[jc])&(a>=ss[jc]); r=((a-c0)//RES).astype(np.int64)
        return np.where(ok,r,-1), np.where(ok,tt[jc],'-')
    ck=sorted(set(recs[cur][:,0].tolist())|set(tma[cur][:,0].tolist()))
    ok_=sorted(set(recs[other][:,0].tolist())|set(tma[other][:,0].tolist()))
    cols=ck+ok_; gidx={g:i for i,g in enumerate(cols)}; nck=len(ck)
    img=np.ones((nrows,len(cols),3))
    selfc=cross=0
    for L in (cur,other):
        for arr in (recs[L],tma[L]):
            if len(arr)==0: continue
            ci=np.array([gidx.get(g,-1) for g in arr[:,0]]); v=ci>=0
            a=arr[v,1]; ci=ci[v]; row,ty=maprow(a); go=row>=0
            for r,c,tyy in zip(row[go],ci[go],ty[go]): img[r,c]=TCOL[tyy]
            if L==cur: selfc+=int(go.sum())
            else: cross+=int(go.sum())
    fig,ax=plt.subplots(figsize=(22,16))
    ax.imshow(img,origin='lower',aspect='auto',interpolation='none',extent=[0,len(cols),(c0-gmin)/1e9,(c1-gmin)/1e9])
    ax.axvline(nck,color='k',lw=1.5)
    ax.set_title('当前=L%d (相邻=L%d)  token=%d\n纵轴=本层主权重簇(真实地址) 左=本层%d kernel | 右=相邻层%d kernel   本层命中=%d, 相邻层越界=%d'%(
        cur,other,TK,nck,len(cols)-nck,selfc,cross),fontsize=12)
    ax.set_xlabel('kernel 发起',fontsize=11); ax.set_ylabel('地址偏移 (GB)',fontsize=11)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
    for c,g in enumerate(cols):
        nm=kmap.get(g,''); lab=None
        if 'GroupProblemShape' in nm: lab='MoE'
        elif 'wmma_tensorop' in nm: lab='attnQKV'
        elif 'kernel_mha' in nm: lab='MHA'
        elif 'GemmUniversal' in nm and 'Group' not in nm: lab='GEMM'
        elif 'fused_recurrent_gated' in nm: lab='GDN'
        if lab: ax.annotate(lab,(c,(c0-gmin)/1e9),xytext=(c,4),fontsize=6,rotation=90,va='bottom',ha='center')
    h=[mpatches.Patch(color=TCOL[k],label=TLAB[k]) for k in ['moe','attn','scale','shared','other']]
    ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.05),ncol=5,fontsize=10,frameon=False)
    fig.subplots_adjust(left=0.06,right=0.995,top=0.95,bottom=0.07)
    fig.savefig(fname,dpi=130,facecolor='white'); plt.close(fig); print("saved",fname,"nrows=%d"%nrows)
draw(M,N,'v5_L24_kernel.png')
draw(N,M,'v5_L25_kernel.png')

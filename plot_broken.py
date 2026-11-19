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
ws=[];we=[];wt=[];wl=[]
for nm,p,nb,sh,dt in W: ws.append(p);we.append(p+nb);wt.append(wtyp(nm));wl.append(L_of(nm))
ws=np.array(ws,np.int64);we=np.array(we,np.int64);wt=np.array(wt,object);wl=np.array(wl,np.int64);o=np.argsort(ws);ws=ws[o];we=we[o];wt=wt[o];wl=wl[o]
def owner_w(a):
    j=np.searchsorted(ws,a,'right')-1; jc=np.clip(j,0,len(ws)-1)
    return np.where((j>=0)&(a<we[jc])&(a>=ws[jc]),wl[jc],-1)
ks=np.array([s for s,e in kvr],np.int64); ke=np.array([e for s,e in kvr],np.int64);ko=np.argsort(ks);ks=ks[ko];ke=ke[ko]
TY=['moe','attn','scale','shared','other','kv','act']
TCOL={'moe':(0.80,0.05,0.05),'attn':(0.95,0.55,0.05),'scale':(0.80,0.70,0.05),'shared':(0.55,0.30,0.80),'other':(0.10,0.35,0.85),'kv':(0.05,0.60,0.20),'act':(0.55,0.55,0.60)}
TLAB={'moe':'MoE','attn':'attention','scale':'scale','shared':'shared','other':'norm/路由','kv':'KV','act':'激活/workspace'}
def type_of(a):
    a=np.asarray(a,np.int64); out=np.full(len(a),'act',object)
    j=np.searchsorted(ws,a,'right')-1; jc=np.clip(j,0,len(ws)-1)
    inw=(j>=0)&(a<we[jc])&(a>=ws[jc]); out[inw]=wt[jc][inw]
    j2=np.searchsorted(ks,a,'right')-1; jc2=np.clip(j2,0,len(ks)-1)
    ink=(j2>=0)&(a<ke[jc2])&(a>=ks[jc2])&(~inw); out[ink]='kv'
    return out
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
RES=512*1024; MINH=46; GAPR=18; GAPB=50*1024*1024; TINY=5*1024*1024
def draw(cur,other,fname):
    allc=np.concatenate([recs[cur][:,1] if len(recs[cur]) else np.array([],np.int64), tma[cur][:,1] if len(tma[cur]) else np.array([],np.int64)])
    bins=np.unique(allc//RES); bands=[]; s0=bins[0]; p=bins[0]
    for x in bins[1:]:
        if (x-p)*RES<=GAPB: p=x
        else: bands.append((int(s0*RES),int((p+1)*RES))); s0=x; p=x
    bands.append((int(s0*RES),int((p+1)*RES)))
    big=[b for b in bands if (b[1]-b[0])>=TINY]; tiny=[b for b in bands if (b[1]-b[0])<TINY]
    segs=[(a,b,False) for a,b in big] + ([(0,0,True)] if tiny else [])
    hs=np.array([max(MINH,int(np.ceil((b-a)/RES))) if not t else MINH for a,b,t in segs])
    offsA=[]; t0=0
    for i,h in enumerate(hs):
        offsA.append(t0); t0+=h
        if i<len(hs)-1: t0+=GAPR
    offsA=np.array(offsA); total=int(t0); nseg=len(segs)
    bigsegs=[s for s in segs if not s[2]]; bs=np.array([s[0] for s in bigsegs] or [0]); be=np.array([s[1] for s in bigsegs] or [1])
    seg_of_bigstart={a:i for i,(a,b,tt) in enumerate(segs) if not tt}
    tiny_seg=nseg-1 if tiny else -1
    def loc(a):
        j=np.searchsorted(bs,a,'right')-1; jc=np.clip(j,0,len(bs)-1)
        inbig=(j>=0)&(a<be[jc])&(a>=bs[jc])
        segi=np.array([seg_of_bigstart[int(x)] for x in bs])[jc]
        seg=np.where(inbig, segi, tiny_seg)
        frac=np.where(inbig,(a-bs[jc])/np.maximum(be[jc]-bs[jc],1),0.5)
        return seg,frac
    ck=sorted(set(recs[cur][:,0].tolist())|set(tma[cur][:,0].tolist()))
    okg=sorted(set(recs[other][:,0].tolist())|set(tma[other][:,0].tolist()))
    cols=ck+okg; gidx={g:i for i,g in enumerate(cols)}; nck=len(ck); Nt=len(cols)
    img=np.ones((total,Nt,3)); selfc=0; removed=0
    for L in (cur,other):
        for arr in (recs[L],tma[L]):
            if len(arr)==0: continue
            ci=np.array([gidx.get(g,-1) for g in arr[:,0]]); v=ci>=0
            a=arr[v,1]; ci=ci[v]; seg,frac=loc(a); ty=type_of(a); ow=owner_w(a)
            sc=np.clip(seg,0,nseg-1)
            row=np.where(seg>=0, offsA[sc]+np.clip((frac*hs[sc]).astype(np.int64),0,hs[sc]-1), -1)
            inband=row>=0
            if L==cur:
                keep=inband & ((ow<0)|(ow==cur))
                removed+=int((inband & (ow>=0)&(ow!=cur)).sum())
            else:
                removed+=int((inband & (ow==cur)).sum())
                keep=np.zeros_like(inband)
            for r,cc,tyy in zip(row[keep],ci[keep],ty[keep]): img[r,cc]=TCOL[tyy]
            if L==cur: selfc+=int(keep.sum())
    fig,ax=plt.subplots(figsize=(22,16))
    ax.imshow(img,origin='lower',aspect='auto',interpolation='none',extent=[0,Nt,0,total])
    ax.set_xlim(0,Nt); ax.set_ylim(0,total)
    ax.axvline(nck,color='k',lw=1.6)
    # 纵轴(左边缘)标签：每段一个，居中；断裂符号紧贴左轴外侧
    Lx=-0.5
    for i,(a,b,tt) in enumerate(segs):
        yc=offsA[i]+hs[i]/2
        txt='分散小段(<5MB)' if tt else '%.3f–%.3f GB'%((a-gmin)/1e9,(b-gmin)/1e9)
        ax.text(Lx,yc,txt,ha='right',va='center',fontsize=9,color='#222222')
        if i<nseg-1:
            yg=offsA[i]+hs[i]+GAPR/2; dx=0.006*Nt; dy=6
            ax.plot([-dx,-dx/2,0,dx/2,dx],[yg+dy,yg-dy,yg+dy,yg-dy,yg+dy],color='k',lw=1.1,clip_on=False,zorder=6)
    ax.set_yticks([])
    ax.set_title('L%d 全部访存地址（纵轴=地址偏移GB，断轴非等比例）  token=%d   本层命中=%d   权重越界=0（已剔除解析假象 %d 条）'%(cur,TK,selfc,removed),fontsize=12)
    ax.set_xlabel('kernel 列序号（每列=1个kernel；本层共 %d 个=0..%d，相邻层共 %d 个）'%(nck,nck-1,Nt-nck),fontsize=10)
    step=max(1,nck//10); ticks=list(range(0,nck,step))+[nck]
    ax.set_xticks(ticks); ax.set_xticklabels([str(t) for t in ticks],fontsize=8)
    # 关键kernel名 -> 下方文字摘要(避免与刻度重叠)
    keylab=[]
    seen=set()
    for c,g in enumerate(cols):
        nm=kmap.get(g,''); lab=None
        if 'GroupProblemShape' in nm: lab='MoE'
        elif 'wmma_tensorop' in nm: lab='attnQKV'
        elif 'kernel_mha' in nm: lab='MHA'
        elif 'reshape_and_cache' in nm: lab='KV写'
        elif 'GemmUniversal' in nm and 'Group' not in nm: lab='GEMM'
        elif 'fused_recurrent_gated' in nm: lab='GDN'
        if lab and lab not in seen: seen.add(lab); keylab.append('%s@col%d'%(lab,c))
    txl=', '.join('%.3f'%(((a+b)/2-gmin)/1e9) for a,b in tiny) if tiny else '无'
    note1='关键 kernel（名称@列号）：'+'，'.join(keylab)
    note2='注1: “分散小段”=跨度<5MB的孤立访问段，其中心地址(GB): %s'%txl
    note3='注2: 相邻层对本层权重的访问均为无效/解析假象，已剔除(权重越界=0)；图右半留空即表示无越界。右半不画相邻层对共享激活的访问以免混淆。'
    fig.text(0.12,0.105,note1,fontsize=8.5,ha='left',color='#333333')
    fig.text(0.12,0.080,note2,fontsize=8.5,ha='left',color='#333333')
    fig.text(0.12,0.055,note3,fontsize=8.5,ha='left',color='#333333')
    h=[mpatches.Patch(color=TCOL[k],label=TLAB[k]) for k in TY]
    ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.045),ncol=7,fontsize=10,frameon=False)
    fig.subplots_adjust(left=0.12,right=0.995,top=0.95,bottom=0.16)
    fig.savefig(fname,dpi=130,facecolor='white'); plt.close(fig)
    print("saved",fname,"big=%d tiny=%d 本层命中=%d 剔除假象=%d"%(len(big),len(tiny),selfc,removed))
draw(M,N,'v5_L24_alladdr.png'); draw(N,M,'v5_L25_alladdr.png')

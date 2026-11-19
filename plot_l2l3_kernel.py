#!/usr/bin/env python
"""L2/L3 各 kernel 对权重地址的访问图（本层/他层二值色，紧凑行=该层真实张量地址）。"""
import json,re,numpy as np,os,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt, matplotlib.patches as mpatches
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False
amap=json.load(open('allocator_map.json')); W=amap['weights']
kvr=amap['kv']['kv_ranges']; gmin=min([w[1] for w in W]+[k[0] for k in kvr])
def L_of(n):
    mm=re.search(r'layers\.(\d+)\.',n); return int(mm.group(1)) if mm else -1
def typ(n):
    n=n.lower()
    if 'embed' in n: return 'embed'
    if '.experts.' in n or 'shared_expert' in n: return 'moe'
    if 'self_attn' in n or 'linear_attn' in n: return 'attn'
    if 'weight_scale' in n or 'alpha' in n or 'global_scale' in n or 'input_scale' in n: return 'scale'
    if 'lm_head' in n: return 'lmhead'
    return 'other'
rec=[]
for nm,p,nb,sh,dt in W: rec.append((p,p+nb,L_of(nm),typ(nm)))
rec.sort(); SA=np.array([r[0] for r in rec]); EA=np.array([r[1] for r in rec]); LA=np.array([r[2] for r in rec]); TA=np.array([r[3] for r in rec],object)
def look(a):
    j=np.searchsorted(SA,a,side='right')-1; jc=np.clip(j,0,len(SA)-1)
    ok=(j>=0)&(a<EA[jc])&(a>=SA[jc]); return np.where(ok,LA[jc],-9), np.where(ok,TA[jc],'-')
RES=1024*1024
wmin=min(r[0] for r in rec); wmax=max(r[1] for r in rec)
bins=np.arange(wmin,wmax,RES); lay_bin,_=look(bins+RES//2)
keep=np.isin(lay_bin,[2,3]); rows_addr=bins[keep]; rows_lay=lay_bin[keep]
_,rows_typ=look(rows_addr+RES//2)
rows_gbin=((rows_addr-wmin)//RES).astype(np.int64)
nrows=len(rows_addr); print("行数(L2/L3真实地址bin)=%d (L2=%d,L3=%d)"%(nrows,(rows_lay==2).sum(),(rows_lay==3).sum()))
kmap={}
for line in open("v5_trace.51233.meta"):
    if line.startswith("KERNEL "):
        p=line.split(" ",2); kmap[int(p[1])]=p[2].strip()
def vec_row(gb):
    k=np.searchsorted(rows_gbin,gb); k=np.clip(k,0,nrows-1)
    hit=rows_gbin[k]==gb; return np.where(hit,k,-1)
TK=3
dt=np.dtype([('grid','<u8'),('cta','<u4'),('warp','<u2'),('opcode','<u2'),('addr','<u8'),('size','<u4'),('layer','<u4')])
REG="v5_trace.51233.bin"; n=os.path.getsize(REG)//32; mm=np.memmap(REG,dtype=dt,mode='r'); CH=20_000_000
recs={2:[],3:[]}; token=0; last=-1
for s in range(0,n,CH):
    c=mm[s:s+CH]; lay=c['layer'].astype(np.int32)
    prev=np.empty_like(lay); prev[0]=last; prev[1:]=lay[:-1]
    tok=token+np.cumsum((lay==100)&(prev==200)); token=int(tok[-1]) if len(tok) else token; last=int(lay[-1])
    mi=(tok==TK)
    if mi.any():
        sub=c[mi]; sl=sub['layer']
        for L in (2,3):
            k=(sl==L)
            if k.any(): recs[L].append(np.stack([sub['grid'][k].astype(np.int64),sub['addr'][k].astype(np.int64)],1))
    if token>TK: break
for L in (2,3): recs[L]=np.concatenate(recs[L]) if recs[L] else np.zeros((0,2),np.int64)
td=np.dtype([('grid','<u8'),('layer','<u4'),('is_write','<u1'),('pad','<u1',(3,)),('start','<u8'),('span','<u8'),('nelem','<u8')])
t=np.fromfile("v5_trace.51233.tma.bin",dtype=td); tl=t['layer'].astype(np.int32); tst=t['start'].astype(np.int64)
twrap=np.where(tl[1:].astype(int)-tl[:-1].astype(int)<-40)[0]+1; ttok=np.searchsorted(twrap,np.arange(len(t)),side='right')
tma={2:[],3:[]}
for L in (2,3):
    m=(ttok==TK)&(tl==L); tma[L]=np.stack([t['grid'][m].astype(np.int64),tst[m]],1) if m.any() else np.zeros((0,2),np.int64)
cols=[]; colL=[]
for L in (2,3):
    gs=sorted(set(recs[L][:,0].tolist())|set(tma[L][:,0].tolist()))
    for g in gs: cols.append(g); colL.append(L)
gidx={g:i for i,g in enumerate(cols)}; N=len(cols)
M=np.zeros((nrows,N),np.int8); cross=np.zeros(N,np.int64)
for L in (2,3):
    for arr in (recs[L],tma[L]):
        if len(arr)==0: continue
        ci=np.array([gidx.get(g,-1) for g in arr[:,0]]); ok=ci>=0
        gb=((arr[ok,1]-wmin)//RES).astype(np.int64)
        rown=vec_row(gb)
        ow=np.where(rown>=0, rows_lay[np.clip(rown,0,nrows-1)], -9)
        go=rown>=0
        M[rown[go],ci[ok][go]]=ow[go]
        cross[ci[ok][~go]]+=1
np.savez_compressed("v5_L2L3_kernel.npz",M=M,cols=np.array(cols),colL=np.array(colL),rows_addr=rows_addr,rows_lay=rows_lay,wmin=wmin,RES=RES)
print("交叉访问统计:")
for gname,Lset in [("L2",2),("L3",3)]:
    ci=[i for i,l in enumerate(colL) if l==Lset]
    a2=(M[:,ci]==2).sum(); a3=(M[:,ci]==3).sum(); cr=int(cross[ci].sum())
    print("  %s kernel组: ->L2地址 %d 格, ->L3地址 %d 格, ->其他层 %d 次"%(gname,a2,a3,cr))
TCOL={'moe':(1.0,0.90,0.90),'attn':(1.0,0.95,0.85),'scale':(1.0,1.0,0.85),'other':(0.95,0.95,0.95),'shared':(0.95,0.90,1.0),'embed':(0.95,0.90,1.0),'lmhead':(0.9,0.97,1.0)}
img=np.ones((nrows,N,3))
for r in range(nrows): img[r,:]=TCOL.get(rows_typ[r],(0.95,0.95,0.95))
for r in range(nrows):
    for c in range(N):
        v=M[r,c]
        if v==2: img[r,c]=(0.10,0.30,0.85)
        elif v==3: img[r,c]=(0.05,0.60,0.20)
fig,axs=plt.subplots(2,1,figsize=(24,12),gridspec_kw={'height_ratios':[1,12]},sharex=True)
cimg=np.ones((20,N,3)); mx=cross.max() or 1
for c in range(N):
    if cross[c]>0: cimg[:,c]=[1,1-0.9*cross[c]/mx,1-0.9*cross[c]/mx]
axs[0].imshow(cimg,origin='lower',aspect='auto',interpolation='none',extent=[0,N,0,1]); axs[0].set_yticks([])
axs[0].set_title('L2/L3 各 kernel 的权重地址访问 (token=%d)  顶部红=访问到非本层/他层地址的次数(越界)'%TK,fontsize=13)
axs[1].imshow(img,origin='lower',aspect='auto',interpolation='none',extent=[0,N,(rows_addr[0]-gmin)/1e9,(rows_addr[-1]+RES-gmin)/1e9])
axs[1].set_ylabel('地址偏移(GB，仅 L2/L3 真实张量所在行)',fontsize=11)
axs[1].set_xlabel('kernel 发起 (左=L2的%d个kernel, 右=L3的%d个kernel)'%(colL.count(2),colL.count(3)),fontsize=11)
div=colL.count(2)
for a in axs: a.axvline(div,color='k',lw=1.5)
for c,g in enumerate(cols):
    nm=kmap.get(g,''); lab=None
    if 'GroupProblemShape' in nm: lab='MoE GEMM'
    elif 'wmma_tensorop' in nm: lab='attn QKV GEMM'
    elif 'kernel_mha' in nm: lab='attn MHA'
    elif 'GemmUniversal' in nm and 'Group' not in nm: lab='GEMM'
    elif 'fused_recurrent_gated' in nm: lab='GDN'
    if lab: axs[1].annotate(lab,(c,(rows_addr[0]-gmin)/1e9),xytext=(c,0.3),fontsize=6,rotation=90,va='bottom',ha='center')
h=[mpatches.Patch(color=(0.10,0.30,0.85),label='访问(L2地址行)'),mpatches.Patch(color=(0.05,0.60,0.20),label='访问(L3地址行)'),
   mpatches.Patch(color=(1.0,0.90,0.90),label='MoE带'),mpatches.Patch(color=(1.0,0.95,0.85),label='attn带'),
   mpatches.Patch(color=(1.0,1.0,0.85),label='scale带'),mpatches.Patch(color=(0.95,0.90,1.0),label='shared/other'),
   mpatches.Patch(color=(1,0.1,0.1),label='顶部越界(访问他层)')]
axs[1].legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.10),ncol=7,fontsize=9,frameon=False)
fig.subplots_adjust(left=0.05,right=0.995,top=0.94,bottom=0.11,hspace=0.05)
fig.savefig('v5_L2L3_kernel.png',dpi=140,facecolor='white'); print("saved v5_L2L3_kernel.png",img.shape)

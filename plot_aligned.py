import numpy as np,matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["Noto Sans CJK SC","WenQuanYi Micro Hei"]; plt.rcParams["axes.unicode_minus"]=False
d=np.load("v5_aligned.npz",allow_pickle=True)
H=d['H']; total_rows=int(d['total_rows']); base=d['base']; nsyn=d['nsyn']; LA=d['LA']; TA=d['TA']; gmin=int(d['gmin']); RES=int(d['RES'])
owner_of_row=np.zeros(total_rows,np.int64)
for i in range(len(LA)): owner_of_row[base[i]:base[i]+nsyn[i]]=LA[i]
# 每列跨层占比
print("每层(列) 跨层访问占比:")
bad=[]
for al in range(48):
    t=H[al].sum(); 
    if t<=0: continue
    cr=H[al][(owner_of_row!=al)&(owner_of_row>=0)].sum()
    if cr/t>0.002: bad.append((al,cr/t*100)); print("  L%-2d 跨层=%.3f%%"%(al,cr/t*100))
print("跨层>0.2%%的层:",[b[0] for b in bad] if bad else "无")
cats=['embed','moe','attn','scale','shared','other','lmhead']
col={'moe':(0.80,0.05,0.05),'attn':(0.95,0.55,0.05),'scale':(0.80,0.70,0.05),'shared':(0.55,0.30,0.80),'other':(0.10,0.35,0.85),'embed':(0.60,0.30,0.80),'lmhead':(0.05,0.80,0.90)}
# 行类型颜色
rowtyp=np.empty(total_rows,object)
for i in range(len(LA)): rowtyp[base[i]:base[i]+nsyn[i]]=TA[i]
img=np.ones((total_rows,48,3))
for r in range(total_rows):
    base_c=np.array(col.get(rowtyp[r],(0.5,0.5,0.5)))
    img[r,:,:]=1-(1-base_c)*0.10   # 极淡底
for al in range(48):
    rr=np.where(H[al]>0)[0]
    if len(rr)==0: continue
    vals=np.log1p(H[al][rr]); gmax=np.log1p(H[al]).max() or 1
    for r,v in zip(rr,vals):
        base_c=np.array(col.get(rowtyp[r],(0.5,0.5,0.5)))
        img[r,al]=1-(v/gmax)*(1-base_c)
fig,ax=plt.subplots(figsize=(16,16))
ax.imshow(img,origin='lower',aspect='auto',interpolation='none',extent=[0,48,0,total_rows])
ax.set_title('层×地址 访存热图（张量边界对齐、无插值）  y=权重地址(按张量), x=发起访问的层',fontsize=13)
ax.set_xlabel('发起访问的层 L0..L47',fontsize=11); ax.set_ylabel('地址（按张量顺序，非等距）',fontsize=11)
ax.set_xticks(range(0,48,4)); ax.set_xticklabels(['L%d'%i for i in range(0,48,4)])
# y 轴按 GB 标几个刻度
def row_of_addr(a):
    j=np.searchsorted(d['SA'],a,'right')-1; jc=np.clip(j,0,len(d['SA'])-1)
    if not(a>=d['SA'][jc] and a<d['EA'][jc]): return None
    return base[jc]+min(nsyn[jc]-1,(a-d['SA'][jc])//RES)
tick_gb=[33,40,50,60,70,80]
pos=[];lab=[]
for gb in tick_gb:
    r=row_of_addr(gmin+gb*10**9)
    if r is not None: pos.append(r); lab.append('%dGB'%gb)
ax.set_yticks(pos); ax.set_yticklabels(lab)
import matplotlib.patches as mpatches
h=[mpatches.Patch(color=col[k],label={'moe':'MoE','attn':'attention','scale':'scale','shared':'shared','other':'norm/路由','embed':'embedding','lmhead':'lm_head'}[k]) for k in cats]
ax.legend(handles=h,loc='upper center',bbox_to_anchor=(0.5,-0.06),ncol=7,fontsize=10,frameon=False)
fig.subplots_adjust(left=0.07,right=0.995,top=0.95,bottom=0.10)
fig.savefig('v5_layers_aligned.png',dpi=130,facecolor='white'); plt.close(fig); print("saved v5_layers_aligned.png",img.shape)

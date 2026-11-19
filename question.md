# v5 问答记录（权重区布局 / MoE 专家访问）

> 基于 `allocator_map.json`（权重 47.42GB + KV 25GB，同一次运行）与 `v5_trace.51233.*`（4.187B 条 LD/ST + 2986 万条 TMA）及 `v5_Hcat.npy` 统计分析。

## Q1. 一组 = SCALE+W2+W13+ATTN，那么每组数据总量都一样吗？

不完全一样，**差在 attention**。

每层的 **MoE 部分完全相同**：

| 组成 | 每层大小 |
|---|---|
| W13（`routed_experts.w13_weight`，gate_up，shape `[512,1024,1024]`） | 536.9 MB |
| W2（`routed_experts.w2_weight`，down） | 268.4 MB |
| SCALE（NVFP4 分专家量化 scale，432 个小张量） | 101.2 MB |
| SHARED（`shared_expert.*`，约 1.77MB/层） | 1.77 MB |

差异只在 attention：

- **36 层线性注意力(GDN)**：attn = 54.9 MB → 每层合计 **965.2 MB**
- **12 层全注意力**（L3, 7, 11, …, 47，每 4 层一个）：attn = 41.9 MB → 每层合计 **952.3 MB**

即：MoE 部分 48 层一致；每层总量因 attn 类型不同而差 ~12.9MB。

## Q2. 一层含 512 个专家、token 只激活 10 个 → 访存图每组红色块应是 10/512 的细线？

**单个 token 上是对的；整段 trace 累计下来不是。**

- W13 张量 `[512, 1024, 1024]`，**512 个专家连续打包**，每专家 **1.05 MB**。
- **单个 token**：只激活 top-10 → 只读 10 个专家 ≈ `10 × 1.05MB = 10.5MB`，占 W13 的 **~2%**；且这 10 个是**按专家下标的 10 条离散细条**（并非一条连续细带）。加上 W2(≈5.2MB)+scale，每层每 token ≈ 17MB / ~950MB ≈ **2%**。
- **但这段 trace 是 batch=1、3 个请求 × 20 decode token ≈ 60 个 token 累加**，每个 token 选不同的 10 个专家。用 TMA 起点精确统计，累计后每层 W13 **平均访问 201/512 ≈ 39% 的专家**（中位 194，范围 140–350）。
- 且我们每像素 ≈ 1.3MB ≈ 一个专家大小，散条会**粘连成"整块"**，所以图中红块看起来较满，**不是 2% 细线**。

> 要看到真正的"10/512 细条"，需按**单个 token** 出图，或把地址分辨率做到 <1MB。

各层访问专家数（W13 / W2，累计，batch=1，60 token）：

```
层  W13专家/512   W2专家/512
 0   350 (68%)     298 (58%)
 1   270 (53%)     227 (44%)
 2   318 (62%)     267 (52%)
 3   158 (31%)     122 (24%)
 4   177 (35%)     149 (29%)
 5   194 (38%)     158 (31%)
...
47   240 (47%)     199 (39%)
W13: min=140 中位=194 max=350 平均=201.4 /512
W2 : min=110 中位=157 max=298 平均=162.7 /512
```

## Q3. SHARED 在什么地址区间？

**不是一个连续区间。** 每层各有约 **1.77MB** 的 `shared_expert` 小张量，**紧挨在该层那块旁边**，从 **34.3GB 一直散到 82.5GB，每层一个 ~2MB 小簇**（例如 35.29、38.28、48.00 … 79.61、81.10、82.46GB）。它被"揉"进了每层的组里，不是单独一块。

## Q4. 之前说 MoE "离散分布"，现在说 512 专家连续——矛盾吗？

**不矛盾。**

- **在一个 w13 / w2 张量内部**：512 个专家**是连续的**（packed）。
- "离散 / 多块"指的是**张量层面**：一层的 MoE 由 **w13 + w2 + scale（432 个小张量）+ shared** 这些**彼此分开的**张量组成，且层与层在地址上是**倒序 + 交错**（还夹着 attn）。所以一层在地址上表现为"好几个红色块"，但每个块内部的 512 专家是连续的。

## Q5. 为什么 L0–L3 这 4 层被劈开？是刻意设计的吗？

**不是模型设计，是分配/加载顺序的副作用**（CUDA caching allocator + 权重落地顺序）：

- 大张量按 **L47 → … → L0 倒序**分配；`embedding`（622MB，78.99GB）恰好被放到了 L0–L3 剩余部件的位置上；
- 于是 L0–L3 的 **W13 留在 embed 前面（76.5–78.99GB）**，而它们的 **W2 / ATTN / SCALE 被挤到 embed 后面（81.1–82.4GB）**，中间 79.62–81.10GB 留下 **1.48GB 空洞**（灰带）。
- 这是**内存分配碎片 / 顺序**造成的，可复现但并非有意布局。

## 附：层布局（倒序，每层 = SCALE+W2+W13+ATTN+SHARED）

大张量（≥50MB）按地址的实际序列（节选，倒序）：

```
LMHEAD@32.75  L47@33.4  L46@34.4  L45@35.4  ...  L4@75.6  L3@76.5
W13L2@77.38  W13L1@77.91  W13L0@78.45  EMBED@78.99
[1.48GB 空洞 79.62–81.10]
ATTNL0@81.10  W2L0@81.17  ATTN/W2/SCALE of L1,L2,L3 @81.5–82.4   (被劈开的 L0–3 剩余部件)
```

权重区总览（按地址长度）：MoE 50.2% / KV 28.8% / 空洞 16.5% / attention 2.9% / embed 0.7% / lm_head 0.7% / 其他 0.1%。

---

# 附：load/store 与 TMA 的端到端流程（原理说明）

## Q6. SM 是什么？先建立一张"工厂"地图

**SM = Streaming Multiprocessor（流式多处理器）**，是 GPU 里真正"干活的计算单元"。一块 GPU 有很多个 SM（RTX 6000D 约 190 个）。kernel 的线程最终被分到某个 SM 上执行。

每个 SM 内部关键部件：

- **寄存器堆（register file）**：每个线程**私有**的小抽屉，最快。
- **shared memory（共享内存）**：同一线程块（CTA）内所有线程**共用**的片上货架，快，几十~200KB。
- **L1/L2 cache**：片上缓存。
- **LSU（Load/Store Unit）**：执行线程级 load/store 的单元。
- **TMA 引擎**：专用"批量搬运引擎"（DMA，Hopper/Blackwell 才有）。
- 还有 CUDA core、Tensor core、warp 调度器等。

**工厂类比：**

| 硬件 | 类比 |
|---|---|
| GPU | 整个工厂 |
| SM | 一个车间 |
| 线程 / warp | 车间里的工人 |
| 寄存器 | 每个工人自己手边的抽屉（私有、最快、很小） |
| shared memory | 车间里大家共用的货架（片上、快、中等） |
| global memory | 远处的大仓库（显存 HBM，很大、很远、较慢） |

## Q7. global 是什么？

**global memory = GPU 的主显存（HBM），即"大仓库"。** 模型权重、KV cache、激活值都放这里；trace 里记录的 `addr` 就是 **global 地址**。它容量大（几十 GB）、带宽高，但**离计算单元远、延迟高**（比寄存器/共享内存慢一到两个数量级）。数据要参与计算，通常得先从 global 搬到片上的 register 或 shared memory。

## Q8. load/store 的端到端过程

### 术语版
1. 某线程执行到 `LDG`（load global）指令，指令带**一个 global 地址**和**一个目标寄存器**。
2. 指令进 SM 的 **LSU** 发射队列，warp 调度器调度整个 warp 执行。
3. LSU 把 warp 内各线程地址做 **coalescing（合并）**，合成若干 32B/64B/128B 的 memory transaction。
4. 请求经 L1 → L2 → 显存控制器 → **HBM 取数**。
5. 数据原路返回，写进该线程的**目标寄存器**。
6. store（`STG`）反过来：寄存器 → global。

=== 要点：以线程为单位、每个线程自己算地址、结果进 register（线程级访存）。

### 大白话版
每个工人（线程）自己推小车，去大仓库（global）取一小箱货，放进**自己的抽屉**（寄存器）。货多就要很多工人、跑很多趟，且每个工人都得记住"我这箱在第几排第几格"（地址寄存器）。

## Q9. TMA 的端到端过程

### 术语版
1. host 端用 `cuTensorMapEncodeTiled` 生成一张 **tensor-map 描述符**（放常量/全局内存）：写明"这块数据什么形状、stride、元素类型、一次搬多大的 tile、如何 swizzle"。
2. kernel 里**只由一个线程**执行一条 `cp.async.bulk.tensor.2d/3d/4d ...` 指令（抓到的 SASS 为 **`UTMALDG`**；写回为 **`UTMASTG`**）。参数只有：描述符地址 + **tile 起始坐标** + **shared 目标地址** + **mbarrier（完成通知）**。
3. **TMA 引擎**读描述符，自己算出该 tile 覆盖的所有 global 地址，自己生成海量内存事务。
4. 数据从 HBM 经 L2 **直接写进 shared memory**（DMA 直写，**不经过寄存器**）。
5. 搬完通过 mbarrier 通知；消费者（如 Tensor Core 做 GEMM）再从 shared memory 取用。
6. store 方向同理：`UTMASTG`，shared → global。

=== 要点：一条指令搬一大块、单线程发起、引擎算地址、异步、结果进 shared memory。

### 大白话版
不再让每个工人自己搬，而是给"叉车调度员"一张**取货单**（描述符：哪个区、几排几列、搬到哪）。调度员（TMA 引擎）**一次把整托盘货**直接送到**车间共享货架**（shared memory），工人不用一个个搬、也不用记地址。搬的时候工人可以干别的，搬完**按个铃**（mbarrier）通知。

## Q10. 两条路径对比（抽象）

| 角度 | load/store | TMA |
|---|---|---|
| 谁在搬 | 每个工人（线程） | 一个调度员（引擎） |
| 搬多少 | 一小箱（几个字节/线程） | 一整托盘（KB~MB 的 tile） |
| 谁算地址 | 每人都要算 | 调度员按单据算 |
| 放哪 | 自己抽屉（**register**） | 共享货架（**shared memory**） |
| 同步 | 搬完才能用 | 异步，搬时干别的，铃响再用 |
| 适合 | 零散、不规则访问 | 大块、规则数据（GEMM 权重/激活） |

## Q11. global → register 路径（load/store 走的）

- **终点**：寄存器堆（线程私有）。
- **路径**：`LDG` → LSU → L1/L2 → HBM → 数据经同样路径回来 → 写进寄存器。
- **优点**：算的时候 ALU 直接拿寄存器当操作数，**立刻能算**。
- **代价**：①寄存器珍贵有限；②每线程都要存地址、做地址运算；③大块规则数据用线程级搬很"碎"，指令多、发射压力大、效率低。
- **典型场景**：逐元素算子、标量/向量访存、随机访问。

## Q12. global → shared 路径（TMA 走的）

- **终点**：shared memory（片内、CTA 内线程共享，Tensor Core / MMA 可直接读）。
- **路径**：`UTMALDG` → TMA 引擎 → L2 → HBM → **直写 shared memory**（DMA，绕开"逐线程 + 寄存器"通路）。
- **优点**：①一条指令搬整块；②地址由描述符算，**省寄存器**；③**异步**，搬时线程能做别的；④数据直接落 shared，正好喂给 GEMM 的 Tensor Core（CUTLASS 主循环的标准玩法）。
- **代价**：①要预先建描述符；②有对齐/布局要求；③对零散随机小访问不划算。
- **典型场景**：**GEMM 主循环把权重/激活 tile 搬进 shared 再算**——这正是 trace 里 MoE/attention 权重走 TMA 的原因。

## Q13. 回到我们的 trace

- `LDG/STG` 是"每个线程搬几个字节进寄存器"，工具逐线程就能抓到。
- `UTMALDG/UTMASTG` 是"一条指令搬整块到共享内存"，**不是线程级访存**，所以旧的逐线程工具完全漏掉——这就是 v4 看不到权重流量的原因。
- v5 加了第二通道：截获 TMA 指令 → 读 tensor-map 描述符 → 用 `nvbit_parse_tma_src_addrs/dst_addrs` 还原"这一整块 tile 的 global 地址范围" → 记 `start + span + nelem`。

**一句话总结：load/store 是"工人自己跑腿搬小件进抽屉"，TMA 是"调度员按单据开叉车把整托盘货送到共享货架"。**

---

# 附：每层 MoE 组成 / 专家结构 / shared expert / 蓝灰条

## Q14. "每层 MoE 部分完全相同" 是指大小和数据都一样吗？

**尺寸/形状相同，数值不同。**

每层 W13=536.9MB、W2=268.4MB、SCALE=101.2MB、SHARED=1.77MB，**结构和大小一模一样**（所以每层总量一样）；但**权重数值是各层各自训练出来的，不一样**。相同的是"规格"，不是"内容"。

## Q15. W13 就是"专家"吗？W2、SCALE 又是干什么的？

**W13 不是"专家"，一个专家是 W13+W2+SCALE 三者按专家切片的组合。**

实测形状（层0）：

| 张量 | shape | dtype | 含义 |
|---|---|---|---|
| `routed_experts.w13_weight` | `[512,1024,1024]` | uint8 | 512 个专家的 **gate+up** 权重打包 |
| `routed_experts.w2_weight` | `[512,2048,256]` | uint8 | 512 个专家的 **down** 权重打包 |
| `routed_experts.w13_weight_scale` | `[512,1024,128]` | fp8 | W13 的 NVFP4 **反量化 scale** |
| `routed_experts.w2_weight_scale` | `[512,2048,32]` | fp8 | W2 的 NVFP4 **反量化 scale** |

这三个张量的**第 0 维都是 512（专家数）**。第 e 个专家 = `{ W13[e], W2[e], w13_scale[e], w2_scale[e], 几个每专家标量 }` 组合起来才是一个完整的专家 FFN。

（该模型 hidden=2048、MoE 中间维=512：单专家逻辑上 gate_up=`[1024,2048]`、down=`[2048,512]`；NVFP4 每字节存 2 个 4bit 值，故打包成 uint8 为 `[1024,1024]` 和 `[2048,256]`。SCALE 用来把 4bit 权重还原成数值。）

## Q16. shared_expert 是每个 token 每层都一定访问吗？

**是，48/48 层都有访问活动（实测）。**

`shared_expert` 是一个**所有 token 都会经过的稠密小 FFN**（gate_up `[1024,2048]` + down `[2048,512]`，约 1.77MB/层），与路由专家**并行**：

> 每个 token 经过一层 = 路由选 **top-10 个专家** + **1 个 shared 专家**（shared 不参与路由、永远算）。

所以是"**top10(路由) + 1(共享)**"，不是只在 top10 里。

## Q17. 存储彩带里 34.3–82.5GB 的蓝条、灰条分别是什么？是否每层都有？

**蓝条（OTHER）不是 W2，也不是 SCALE**（W2/SCALE 是红色、算在 MoE 里）。蓝条实测只有 4 种：

- `input_layernorm.weight`（每层 1 个）
- `post_attention_layernorm.weight`（每层 1 个）
- `mlp.gate.weight`（**MoE 路由门控**，每层 1 个）
- `model.norm.weight`（最后 1 个，全局只 1 个）

这些**每层都有**（每层 3 个小张量），所以蓝细条**每层都会出现**，位置和 shared 一样散在每层块旁边（实测 35.3–82.45GB，每层一个 ~几 KB 小簇）。

**灰条 = 未分配空洞**：①张量之间**对齐填充**的细缝（每层之间会有细灰线）；②那 **1.48GB 大灰带（79.6–81.1GB）**。这些地址没存模型权重，运行时被当作 **workspace/激活**用。

**小结：蓝 = 每层 2 个 LayerNorm + 路由门控（+末尾 norm），每层都有；灰 = 对齐缝隙 + 未分配空洞（运行时 workspace）；W2/SCALE 都是红色 MoE 部分。**

---

# 附：存储图 vs 访问图 / workspace 是什么

## Q18. 访问热度彩带里 KV、workspace 有大片空白——是很多地址没被访问？没访问怎么知道存的是 KV/workspace？存了 KV 却没访问矛盾吗？

**核心：存储和访问是两件独立的事。**

### 存储分布彩带是根据什么画的（与 trace 无关）
来自**同一次运行单独 dump 的 `allocator_map.json`**：

- 权重：遍历 `model.named_parameters()`，每个参数记 `名字 + data_ptr + 字节数` → 得每个张量的 `[起始, 结束)` 区间；按名字归类（embed/moe/attn/lmhead/weight）。
- KV：遍历 `model_runner.kv_caches`（KV 缓存张量）的 `[ptr, ptr+size)`，合并成区间 → 标为 `kv`。
- **任何没被上面区间覆盖的地址 → 归为"空洞/workspace"（灰色）**。

即：某地址是什么类型 = 用地址去这些区间**查表**，**和有没有被访问完全无关**。

### 空白 = 这段地址在这 57 个 token 里访问量为 0
访问热度彩带来自 trace（统计每个地址 bin 的访问次数）。白色 = 该 bin 访问量 0。

**"某段地址存了 KV，但这段 trace 没访问它" —— 完全不矛盾**（存储=放了什么，访问=用没用）。

### 实测：两块的地址被访问比例很低
| 区域 | 地址bin | 被访问bin | 访问到的地址占比 |
|---|---|---|---|
| **KV/状态** | 18885 | 53 | **0.28%** |
| **空洞/workspace** | 10827 | 78 | **0.72%** |
| MoE 专家 | 32917 | 23193 | 70.5% |
| attention | 1893 | 1893 | 100% |
| lm_head / 其他权重 | — | — | 100% |
| embedding | 470 | 48 | 10.2% |

- **KV 池≈25GB（按 `max_model_len=2048` 预留整池），实际被访问仅≈0.07GB（0.28%）**——只跑了 ~20 token 短上下文，绝大多数 KV 页从未读写。
- **空洞/workspace≈14.3GB，只访问≈0.7%**。
- （灰色严格说是"未登记地址"：既不在参数里也不在 KV 张量里；运行时作激活/临时缓冲，但我们未逐个枚举，故统一标灰。）

## Q19. workspace 到底是干什么用的？（通俗解释）

**workspace = GPU 里"干活时铺开的工作台"**，不是长期数据，而是算完就丢的临时区。

### 厨房类比
| 类比 | GPU | 特点 |
|---|---|---|
| 仓库食材 | 权重 | 长期存、反复读 |
| 剩菜/半成品 | KV cache | 长期存、逐 token 追加读 |
| **案板/备菜区** | **workspace/激活** | **临时用、算完就清、可复用** |

### 里面具体装什么
1. **激活 / 中间结果**：每层算出的 hidden_states、MLP 中间值、attention 输出等临时张量。
2. **算子内部 scratch**：GEMM / FlashAttention / CUTLASS 的分块暂存、累加/规约缓冲等。
3. **框架内部缓冲**：采样 logits、block table、元数据等。
4. **分配器"预留但未用"**：PyTorch caching allocator 一次性向 CUDA 要大块留着复用，其中"已占但当前没用"的部分。
5. **对齐填充（padding）**：张量间为地址对齐留的小缝。

### 为什么图里"灰且大片空白"
- 地图只登记了权重和 KV 两类，其余未覆盖地址统一标灰，**未逐个枚举**其中缓冲。
- 它是**逐层产生又释放的临时区**，单 token、短上下文下激活很小，且反复在很小一块内复用 → 整片只蹭到约 0.7%，其余自然空白。

**一句话：它不是"忘了访问"，而是"本来就是干活的临时台面，干完就空着了"。**

---

# 附：L24 单层地址图（张量排布 / 小标量 / 62 个 kernel）

## Q20. v5_L24_alladdr.png 里 MoE 55.835–56.741 的红线很稀疏，和"激活 10/512 专家"有关吗？

有关，就是 10/512。该 MoE 段（906MB）由 4 个大张量组成：

| 地址(GB) | 大小 | 张量 |
|---|---|---|
| 55.835–55.868 | 33.55MB | `routed_experts.w2_weight_scale` |
| 55.868–55.935 | 67.11MB | `routed_experts.w13_weight_scale` |
| 55.935–56.204 | 268.4MB | `routed_experts.w2_weight`(down) |
| 56.204–56.741 | 536.9MB | `routed_experts.w13_weight`(gate+up) |

w13 是 `[512 experts,…]`、每专家 1.05MB；w2 每专家 0.52MB。**一个 token 只激活 top-10 专家**，专家按 index 分散在张量里 → 只读到 10 小段（≈10MB）/ 每层层 906MB → **红线稀疏 = 10/512**，红线位置就是那 10 个专家的偏移。

## Q21. 纵轴"MoE、attention、shared、激活、MoE、norm、attention…交错"是真实的吗？各类型对应哪些张量？

**是真实的**（按真实地址排序）。L24 是**线性注意力(GDN)**层，其张量在 HBM 上真实排布（节选）：

```
MoE 55.835–56.741  (w2_scale, w13_scale, w2_weight, w13_weight)
attention 56.741–56.791  (linear_attn.in_proj_qkvz.weight)
shared/norm 56.791
attention/norm 57.756–57.762 (out_proj, mlp.gate)
attn/shared 57.775–57.777
一堆 0.00MB 的 scale/alpha 60.685
激活 81/85GB（不是 L24 的存储，是它访问的共享激活地址）
```

各类型张量（L24）：
- **MoE**：`routed_experts.w13_weight`(537MB)、`w2_weight`(268MB)、`w13_weight_scale`(67MB)、`w2_weight_scale`(34MB) + 每专家标量 ≈906MB/512 专家
- **attention(GDN)**：`linear_attn.in_proj_qkvz.weight`、`out_proj.weight(+scale)`、`conv1d.weight`、`in_proj_ba.weight`、`A_log`、`dt_bias`、`norm.weight`、scale/alpha ≈55MB
- **shared**：`shared_expert.gate_up_proj/down_proj`(+scale) + 全局 scale/alpha ≈1.8MB
- **norm/路由**：`input_layernorm`、`post_attention_layernorm`、`mlp.gate.weight` ≈2.1MB
- **scale**：`weight_scale/alpha/global_scale/input_scale`（按名字归入其父类，不单列）

> 注意：图里标"激活/KV"的段不是 L24 的存储，而是 L24 的 kernel **访问到**的共享激活/workspace、KV 地址，按其真实位置画在低位/高位。

## Q22. `MoE 56.791–56.791`、`MoE 60.685–60.685` 怎么解释？和 55.835–56.741 同类型吗？

同一数据类型（都属于 routed_experts/量化参数），但**不是那 4 个大张量**，而是**每专家标量小张量**：

| 地址(GB) | 大小 | 张量 |
|---|---|---|
| 56.7915 | 2KB | `routed_experts.w2_input_scale` |
| 56.7915 | 2KB | `routed_experts.w2_weight_scale_2` |
| 60.6853 | 2KB | `routed_experts.w13_weight_scale_2` |
| 35.2887 | 2KB | `routed_experts.w13_input_scale` |

它们被放在"小张量堆放区"（56.79、60.685 等），同处混着别的类型小张量；分段规则"同类型且相邻才合并"，被别类型隔开就各自成段；又因断轴图给**最小显示高度**，所以 2KB 的标量看起来和 906MB 的大块差不多高（并不代表它真的大）。

## Q23. 图例里有土黄的 scale，但地址空间里没看到 scale 色块？

"scale"指 NVFP4 量化参数（`weight_scale/weight_scale_2/input_scale/global_scale/alpha`）。但本图类型判定是**先把 scale 归到其父模块**：`.experts.` → MoE(红)、`shared_expert` → shared(紫)、`linear_attn` → attention(橙)。因此 L24 里没有张量落进独立的 `scale` 类 → 土黄空着（scale 被"染"成了父类颜色）。若要单独显示，可把 `*_scale/alpha/...` 一律标为 `scale`（则 MoE 大块变 805MB，另 101MB scale 变土黄）。

## Q24. 为什么热图里"其他层的小标量"没画出来？

两层原因：
1. **行都存在，但是否上色取决于该层是否真的访问了它**。实测：`w13_weight_scale_2`（2KB）**48 层全部访问**；`w13_input_scale`（2KB）**48 层都没访问**（永不显示）。
2. **渲染分辨率吃掉了 1 行小张量**：`v5_layers_aligned_clean.png` 共 4265 行，渲染高度约 2080 像素（≈2 行/像素）；2KB 张量只占 1 行≈半像素，下采样时被吃掉/成噪点。对比断轴图给每单元最小 40 行，所以那里看得清楚。

## Q25. L24 的 62 个 kernel 分别是什么？L24 的算子及其对应？

L24 = **线性注意力(GDN) + MoE**。62 个 kernel（token3）按执行顺序分组：

- **[0]–[12] 输入 RMSNorm + 量化**：direct_copy/add_/elementwise/bf16_copy，`pow`+`reduce_mean`+`rsqrt` = RMSNorm；copy/cat/fill 做 reshape。（不读权重）
- **[13]–[27] GDN 线性注意力**：
  - [13] cutlass wmma GEMM = **in_proj**(in_proj_qkvz)
  - [14] gemvx（小）
  - [23] `causal_conv1d_update` = GDN 短卷积
  - [24] `fused_recurrent_gated_delta_rule` = GDN 门控 delta-rule 递推
  - [25] `layer_norm_fwd` = GDN 门控 RMSNorm
  - [26] `cvt_fp16_to_fp4` = 量化注意力输出
  - [27] cutlass GEMM = **out_proj**
- **[28]–[41] post_attention RMSNorm + 量化**：与 A 段同构；[41] `cvt_fp16_to_fp4` 量化 MoE 输入。
- **[42]–[49] shared expert（稠密）**：[42] cutlass GEMM=gate_up；[43] `act_and_mul`(SiLU×门)；[44] 量化；[45] cutlass GEMM=down；[46] cublas dot；[47] reduce_1Block；[48] sigmoid；[49] elementwise（sigmoid 门控合并）。
- **[50]–[61] 路由 + 路由专家 MoE**：
  - [50] gemvx(66048) = **router 门控**（读 mlp.gate）
  - [51] `topkGating` = 选 top-10
  - [52] `cvt_fp16_to_fp4` 量化
  - [53]–[55] block/global/merge `ExpertPrefixSum`（grouped GEMM 元数据）
  - [56] `expandInputRows` 按专家展开 token
  - [57] `computeStridesTma` 计算 TMA strides
  - [58] cutlass grouped GEMM = **专家 w13(gate+up)**
  - [59] `doActivation` 专家 SiLU
  - [60] cutlass grouped GEMM = **专家 w2(down)**
  - [61] elementwise 合并

映射到图：
- 红 MoE=[58][60]（读 routed w13/w2，10/512 稀疏红条）；56.791/60.685 红点=量化标量
- 橙 attention=[13][23][24][27]
- 紫 shared=[42][43][45][49]
- 蓝 norm/路由=[50]（router）+ 各 RMSNorm 权重
- 灰 激活/workspace=其余 copy/elementwise/quant/prefix-sum 等（不碰权重）

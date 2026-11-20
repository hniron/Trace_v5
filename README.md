# HBM 访存 Trace 采集与分析（v5）—— 普通 LD/ST + TMA

本目录是 v5 阶段：在 v4（带 layer_id/相位码）基础上，**补上之前缺失的 TMA 流量**，并给出**严格的"访存图"**。

## 一、为什么做 v5
v4 只抓了普通 load/store（LDG/STG 等），**漏掉了 CUTLASS GEMM 通过 TMA(Tensor Memory Accelerator) 搬运权重/激活的流量**（kernel 名含 `TmaWarpSpecialized`、`SM90_TMA_LOAD`）。所以 v4 热图里"看不到权重被访问"是漏抓 TMA 造成的。v5 把 TMA 也抓进来。

## 二、图的两套语义（务必分清）

| | 访存图（本目录 `v5_*` 的图） | 存储图（v4 的 `*storage*` 那类） |
|---|---|---|
| 颜色含义 | 该层**实际访问到的地址**的真实数据类型 | 该地址**存放了**什么类型 |
| 上色条件 | **只有该层真访问了的地址块才上色**，未访问=白 | 按地址存的东西上色，与是否访问无关 |
| 目的 | 看"每层真正读了/写了哪些地址、是什么" | 看"内存怎么排布、各类型占多大" |

本 v5 的图是**访存图**。

## 三、访存图的画法（v5 最终版）
- **x 轴**：三相位 `[emb] | 层0…47 | [out]`（相位码 100=embed, 0-47=层, 200=output）。
- **y 轴**：地址偏移（0 = KV 池起点，单位 GB）。
- **颜色 = 该格(该层 × 该地址块)实际访问到的地址的"精确数据类型"**（见第四节）。
- **亮度 = 活动量**：普通记录数 + TMA 元素数（对数）。

## 四、边界处理（关键，"彻底清理"embedding 误标）
早期版本按"地址分桶中心"判类型，导致跨在 embedding 张量边界上的桶被误标成 embed（把隔壁权重也涂紫）。v5 改为：
- **逐记录精确判定**：每条记录用其**真实地址**在"张量地址区间"里二分查找，精确归到所属张量类型；**TMA 用每个 tile 的起点地址**精确归类（tile 跨度 < 0.26MB，远小于一个桶，不跨界）。
- 结果：**层列(L0–47)内 embedding 计数 = 0**（embedding 只在 `emb` 相位出现，且是真的嵌入选址访问）；紫色不再出现在任何层。
- 已验证：TMA 起点落在 embed 精确范围=0、TMA 跨度与 embed 相交=0 → **embedding 没被 TMA 访问**。

## 五、TMA 采集与格式（方案A）
- 设备端把 TMA 的 tensor-map 句柄推给 CPU；CPU 用 `nvbit_parse_tma_src_addrs/dst_addrs` 解析出这次传输的**全局地址**。
- 记录粒度（方案A，体积可控）：每个 TMA 传输记一条 `tma_rec_t`(40B)：`grid, layer_id, is_write, start_addr, span, n_elems`（span=元素地址的 max-min，n_elems=元素数）。
- opcode：`UTMALDG.2D/3D/4D`（读）、`UTMASTG.2D/3D`（写）。

## 六、文件与统计

### 数据
| 文件 | 内容 |
|---|---|
| `v5_trace.51233.bin` | 普通 LD/ST，**4,187,095,040 条**（32B/条，含 layer_id） |
| `v5_trace.51233.tma.bin` | **TMA 传输 2986 万条**（40B/条，方案A） |
| `v5_trace.51233.meta` | opcode/kernel 映射 |
| `allocator_map.json` | 本次运行 权重 47.42GB + KV 25GB 精确地址 |
| `v5_Hcat.npy` | 按 7 类精确累计的直方图 (7, 52, 65536) |
| `v5_H.npz` / `v5_meta.npz` | 中间产物 |

### 图
| 文件 | 说明 |
|---|---|
| `v5_tma_weights.png` | **仅 TMA** 权重区访存图（MoE≈67% / attention≈10%） |
| `v5_combined_weights.png` | LD/ST+TMA 权重区访存图 |
| `v5_combined_KV.png` | KV/状态区 |
| `v5_combined_act.png` | 激活区 |

### 各类型活动量（普通记录 + TMA 元素）
| 类型 | 活动量 |
|---|---|
| MoE 专家 | ~570 亿（TMA 主导） |
| attention 权重 | ~129 亿 |
| 激活/workspace | ~80 亿 |
| lm_head | 5.7 亿 |
| 其他权重 | 0.9 亿 |
| KV/状态 | 0.11 亿 |
| embedding | 5504（仅 emb 相位，真实嵌入选址） |

> 关键结论：**权重（尤其 MoE/attention）的 HBM 流量绝大部分来自 TMA**——这是 v4 缺、v5 补上的核心。

## 七、工具（`hbm_trace/`，编译产物 `hbm_trace_v5.so`）
- 在 v2（带 layer_id + 相位码 + decode-only 门控）基础上，加了 **TMA 第二通道**：
  - 普通 LD/ST 走通道1 → `.bin`（32B 记录）；
  - TMA 走通道2（push tensor-map 句柄）→ CPU 解析地址 → `.tma.bin`（40B 记录）。
- 插桩：`instr->isTMAMem()||isTMTensor()` 时注入 `instrument_tma`；否则按原 MREF 注入 `instrument_mem`。
- 相位码沿用：`100=embed, 0-47=层, 200=output`（由 Python 侧 forward hook 打 marker 实现）。

## 八、局限与注意事项
1. **单位不可直接相加**：亮度里"普通记录数"与"TMA 元素数"维度不同（指令/请求 vs 数据元素），联合图只作定性活动量；若要严格，需按字节换算（TMA 每元素字节数需确认）。
2. **本次非洁净收尾**：采集时因工具超时把主进程杀过，普通 trace 32B 对齐完整、TMA 已截到 40B 对齐，但**尾部有少量未 flush**；若要 100% 干净需后台重跑。
3. **TMA 方案A 的取舍**：每传输只记"起点+跨度+元素数"，不逐元素记地址；复杂度低、体积小，但无法在 trace 里逐元素回放（对 HBM 建模按"搬运范围+字节量"足够）。
4. 地址随运行变化，allocator 地图与 trace 必须同一次运行（本目录均是同一次）。

## 九、绘图方法论（最终版：消除"跨层/多层共用色带"歧义）

### 问题根因（为什么旧图会"看起来跨层"）
1. **粗分桶跨张量边界**：地址按固定 bin（1/20MB）聚合，一个 bin 可能横跨两层张量 → 该 bin 只判给一方，另一层访问时也被画到同一带 → 同一条带被多层点亮（假跨层）。
2. **小张量密堆**：norm/路由/scale/shared 等小张量相邻甚至同 bin → 按 bin 中心判归属会错层。
3. **渲染插值/下采样**跨边界混合；按"块中心类型"整体上色。
4. **TMA 解析假象**（见下）。

关键事实：**权重张量在地址上不重叠，任一地址只属于唯一张量/层**。只要"每行只对应单一张量"，就不可能一行被多层点亮。

### 方法（已落地于脚本）
1. **逐记录精确归属**（`analyze_v5b.py` / `analyze_aligned2.py`）：每条访存用其**精确地址**在"张量区间表"中二分查到**唯一所属张量**→ 精确 owner 层 + 类型；不用 bin 中心。
2. **纵轴按张量边界对齐**（`analyze_aligned2.py` + `plot_aligned.py`）：y 轴由张量区间构成，大张量按固定 RES（如 16MB）再细分，小张量各占 ≥1 行；**跨张量处断开，绝不合并** → 每行只属唯一张量，结构性消除"多层共用一行"。
3. **不插值 + 分类上色**：`imshow(interpolation='none')`；颜色=该地址的数据类型（MoE 红 / attention 橙 / scale 黄 / shared 紫 / norm·路由 蓝 / KV 绿 / 激活·workspace 灰）；热图底色**纯白**。
4. **剔除解析假象**：若某记录所属层 L 的 kernel 访问到**别层的权重地址**（物理上不可能，真实跨层≈0）→ 判为 TMA/地址解析假象并丢弃。本次仅命中 **L47**（其约 280 个 `GemmUniversal` grid 解析出"均匀撒到所有层、每层等量 640 条"的假地址）：剔除 regular **9.37M 条** + TMA **70.9 亿元素**；**L0–L46 一条不删**。
5. **断轴图**（`plot_broken.py`）：纵轴按"被访问到的地址段"分段拼接、非等比例；**锯齿断裂符号居中画在纵轴轴线上**；每段仅一个地址标签；kernel 名称放在横轴下方；相邻层列只标"对本层权重"的访问（假象已剔除→右半为空=无越界）。
6. **KV/激活热图**（`plot_kvact.py`）：细对齐分桶、不插值、白底；KV 绿、激活灰。激活/workspace 是**各层复用的同一块 scratch**，故表现为固定地址上的横线（正常）。

### 验证
- 张量对齐后：**每层跨层访问=0（除 L47 假象；剔除后 L47 也=0）**；逐记录精确跨层矩阵对角≈100%。
- 结论：**相邻层之间不存在跨层权重访问**；旧的"拖尾/多层色带"确为绘图/分桶假象。

### 产物（图）
- `v5_layers_aligned_clean.png`：层×地址访存热图（张量对齐、纯白、无假跨层）。
- `v5_L24_kernel.png` / `v5_L25_kernel.png`：L24/L25 权重区（单层×kernel，本层/他层）。
- `v5_L24_alladdr.png` / `v5_L25_alladdr.png`：单层全部访存（断轴，含权重+KV+激活）。
- `v5_KV_aligned.png` / `v5_act_aligned.png`：KV / 激活热度图（同方法）。
- 数据：`v5_aligned_clean.npz`（清洗后张量对齐直方图）。

### 脚本
`analyze_aligned2.py`（精确归属+对齐+清洗）、`plot_aligned.py`/`plot_aligned_clean.py`（层热图）、`plot_broken.py`（断轴单层图）、`plot_kvact.py`（KV/激活热图）。

> 注：如需**根修** TMA 对 grouped-GEMM 的地址解析（而非事后剔除），需改 `hbm_trace` 并重抓一次（约 126GB；当前剩 168GB 可容纳）。对分析而言事后剔除已足够。

## 十、从 trace 到 FPGA 仿真激励的转化方案

目标：为 FPGA 硬件仿真平台（Host FPGA → HBM FPGA，用 MB 级 SRAM 模拟 HBM）提供**统计等价的 decode 访存激励**。原始 125GB trace / 80GB 地址空间**不能直接作为输入**，需抽象为"访存特性指纹 + 生成器"。

### 管线
```
真实 trace → ①特征提取 → ②地址/容量抽象 → ③激励生成 → ④校准验证 ↺
```
交付给 FPGA 的是**紧凑描述符 + 生成器**（KB~MB 级），而非原始 trace。

### ① 提取"访存特性指纹"（通俗详细版）

**什么是"指纹"**：原始 trace 像"所有工人进出仓库的完整监控录像"（125GB）；**指纹 = 从录像里总结出的行为画像**——每层、每类操作，**读/写什么类型、一次搬多大、是"整块搬"还是"东一块西一块"、读还是写、多少并发、落在哪些抽屉(bank/row)**。它很小（KB~MB），却足以让 FPGA 在小仓库里复刻同样的"忙碌/冲突/命中"节奏。

**怎么提取（分步）**：
- **Step0 分桶**：按"层 + 阶段 + 算子类型"分组，如 (L24, MoE GEMM)、(L24, attention/GDN)、(output, lm_head)、(L24, norm)。
- **Step1 逐条贴标签**：每条记录 → ①谁在读(层/kernel)②读什么(用地址查 allocator_map：MoE/attn/KV/激活)③多大怎么搬(LD/ST size、TMA span/nelem)④读/写。
- **Step2 每桶算画像指标**（即指纹）：
  1. **请求大小分布**：LD/ST size 直方图；TMA 每次搬运字节数直方图。
  2. **连续性(run 长度)**：把相邻地址合并成连续块，统计"一次连续搬多少"（区分整块大搬运 vs 碎小随机读）。
  3. **空间散度**：在对象内摊多大范围、几个位置（例：MoE 是 512 个 1.77MB 槽里只碰 10 个）。
  4. **读写比**：权重=读；KV/激活有写。
  5. **时间局部性/复用**：同一地址重复访问的间隔（复用距离）。
  6. **并发度**：多少 CTA/warp 同时发请求（grid/cta/warp），决定能否打满内存。
  7. **地址→bank/row/col 分布**（若知目标 HBM 映射）：每 bank 请求数、行命中 vs 行冲突比例。
  8. **每层/每阶段字节预算与突发性**。

**指纹示例（本项目真实数据）**：
```jsonc
// L24 MoE（grouped GEMM）
{ "type":"moe", "read_bytes":17.7e6, "n_bursts":10, "burst_bytes":1.77e6,
  "contiguity":"每段连续", "spatial":"512槽中散选10个,落在906MB内", "rw":"read" }
// L24 attention/GDN
{ "type":"attn", "read_bytes":55.4e6, "contiguity":"整块连续", "rw":"read" }
// output lm_head
{ "type":"lmhead", "read_bytes":0.62e9, "contiguity":"整块连续", "rw":"read" }
// 激活/workspace
{ "type":"act", "size":"小且瞬时", "reuse":"层间复用同一批地址" }
```

**规模参考（每 token）**：确定性权重≈3.3GB（attention/GDN 2.5 + shared 0.09 + norm/router 0.10 + lm_head 0.62）；MoE 专家≈0.85GB（10/512×48）；合计≈4.2GB/token。

> 全部桶加起来，一个模型 decode 的指纹也就几百个桶 × 十几项统计，非常小；这是后续"地址抽象→激励生成→校准"的输入。

### ② 地址/容量抽象（统计等价映射）
- 定义模拟 DRAM 几何 `channels×banks×rows×cols×burst`，总容量=FPGA SRAM。
- 真实地址 → 模拟地址：**保留低位(burst/col)与 bank 交错规律**，高位用受限范围/哈希压缩：`addr_sim = f(col, bank=hash(中位), row=hash(高位))`。
- 目标：生成流的**每 bank 分布、行命中/冲突率、size 分布**与真实 trace 统计一致（专家"大范围分散读"通过"专家槽↔bank 交错映射"体现，无需真存 80GB）。

### ③ 激励生成（三方案，推荐 C+A 混合）
- **A 统计生成器**：按 `(layer,算子)` 的请求数/size/run/读写/落点分布采样生成；地址用抽象对象表缩放到 MB。输入极小、可实时生成。
- **B 精简回放/地址折叠**：抽代表片段或哈希折叠地址，保留顺序与粒度；时序最真但体积大、折叠需校准。
- **C 结构化模型感知**：用层/算子模板复刻序列，专家路由用热度分布、KV 按上下文增长建模；语义最真、易扩展长上下文。
- 给 FPGA 的每条激励：`{cycle, op(R/W), addr_sim, size, burst, type, src(TMA/LSU)}`；可"描述符+RNG 实时生成"或"预生成紧凑 delta 二进制流"。

### ④ 校准验证
- 指标：size 分布、TMA/LSU 比、读写比、run 长度、复用距离、每 bank 请求数、**行命中率**、bank 冲突率、并发度、每 token/层字节。
- 闭环：生成→跑 FPGA 模型→与 trace 目标对比→调映射/并发/size 分布，迭代到误差达标。

### 当前数据不足，建议补采
长上下文 KV（512/2K/4K）、专家并集与路由热度分布、lm_head/采样占比、必要时 prefill。

### 第一步可交付
`traffic_profile.py`（输出每 `(layer,kernel-type)` descriptor JSON）+ `stim_gen.py`（按 descriptor+DRAM 几何生成激励，并自带与 trace 目标对比报告）。

> 待确认（影响方案收敛）：①FPGA 需要的是"地址+读写+size 流量激励"还是"真实数据搬移"；②目标存储模型是完整 HBM 时序(bank/row)还是仅带宽，容量与 bank 数。

# M188：Motion-aware Event Suppression 冻结迁移审计

> 状态：阶段 A 已完成（官方下载包、`strict=True` 权重加载、输入表示逐位核对和确定性前向）；阶段 B 的 99 个训练视频无标签冻结缓存与 outer `0/2` 跨视频筛选均已完成并否决。**无分数增益、不得接入当前生产链。**

## 1. 为什么这是一个可以承担风险、但仍值得先审计的激进候选

当前最高候选是 `M111 + M169 + P32(seed=3) + M124(0.89)`，完整公开验证
`Score=0.9642995840`。M150、M177、M178、M182、M184 以及多种无标签路由/后处理已经说明：继续从
M26/P41 的已有事件密度、局部时序、通用检测特征或阈值中挤出增益，几乎没有空间。

M188 使用 [Motion-aware Event Suppression for Event Cameras](https://github.com/uzh-rpg/event_suppression)
（RSS 2026）的公开 EVIMO/DSEC 预训练模型。它与上述失败路线的区别是：外部模型直接接受事件体素并学习**动态对象掩码**，
有真实像素级运动掩码监督和 recurrent U-Net 输出，而不是把通用检测或语义 backbone 的粗特征硬接到 P41。

这不是“模型名字新”就默认有用。EVIMO 的 independently moving object 与 EV-UAV 的小型 UAV 有类别、目标尺度和
相机运动差异，因此本轮只给它一次严格的冻结迁移机会。只有证明它为未见训练视频提供 P41 之外的目标/背景排序信息，
才讨论融合或外部微调。

## 2. 已核实的兼容性事实

官方仓库当前 `main` commit 为 `1628d81a8118d25818e18374cc0ce7364133cd0b`，许可证为 GPLv3；官方
`validate_evimo.json` 固定如下输入：

```text
resolution        = [260, 346]       # H, W
representation    = voxel
voxel_bins        = 2
event_dt_ms       = 50
architecture      = 4 encoder ConvGRU U-Net, base_channels=64
output            = full-resolution dynamic-mask logits + future flow/mask
```

本地 `train_000.npz` 的 `ev` 字段为 `(x, y, t, p, label, name)`；已只读核查到：

```text
x in [0, 345], y in [0, 259]   -> 原始传感器正好为 346 x 260
p in {0, 1}                     -> 与官方 polarity 编码一致
t from 0 到约 7988.592          -> 160 个 50-unit 窗口，单位为 ms
```

因此不需要 resize、letterbox、patch token 插值或把小目标压缩到低分辨率网格。这个几何/窗口对齐是 M188 比
已否决外部 backbone 更值得审计的原因。

## 3. 不可更改的输入契约

官方模型并不接受两个 polarity count image。它使用 `ev-loader` 的 `VoxelGrid(normalize=True)`：

1. 每个完整 50 ms 窗口按该窗口第一个和最后一个 event 的时间，将 event 线性插值到 2 个 temporal bins；
2. 输入值为 `2 * p - 1`，即 `p=0` 为 `-1`、`p=1` 为 `+1`；
3. 仅对非零 voxel 做零均值、单位方差归一化；
4. 输出必须为 `[1, 2, 260, 346]` 的 `float32`；视频开始时 reset ConvGRU state，随后按时间顺序连续前向；
5. `dt` 使用官方 EVIMO 的秒单位（50 ms 即 `0.05`），不能把 EV-UAV 的毫秒 `50` 直接送给 time-attention。

任一条不一致时，官方权重即使能 `load_state_dict`，输出也不具备可解释性。先对三段低/中/高密度窗口保存输入
`shape/min/max/nonzero/mean/std`，并与官方 `VoxelGrid` 的源码逐项核对，再允许产生任何候选分数。

## 4. 资产与最小运行步骤

官方 checkpoint 包：

```text
https://download.ifi.uzh.ch/rpg/event_suppression/event_suppression_checkpoints.zip
```

`curl -4 -I` 已确认该地址返回 `200`、`Accept-Ranges: bytes` 和 `Content-Length: 370156537`；不是 403 或
资源不存在。由于持续下载会降到约 `0.10--0.25 MiB/s`，完整包预计约一小时，下载应由用户在 WSL 前台执行并保留
可见进度。隔离目录已经存在于项目外的 `D:\AI\ESOD\_m188_event_suppression`，其中已有一个可续传的
`4,022,272` 字节首段。下面命令可直接粘贴到 WSL：

```bash
conda activate EV39

export M188_DIR=/mnt/d/AI/ESOD/_m188_event_suppression
mkdir -p "$M188_DIR"

if [ ! -d "$M188_DIR/source/.git" ]; then
  git clone --depth 1 https://github.com/uzh-rpg/event_suppression.git "$M188_DIR/source"
fi
git -C "$M188_DIR/source" rev-parse HEAD

curl -4 -L --fail --retry 10 --retry-delay 10 --continue-at - --progress-bar \
  -o "$M188_DIR/event_suppression_checkpoints.zip" \
  https://download.ifi.uzh.ch/rpg/event_suppression/event_suppression_checkpoints.zip

ls -lh "$M188_DIR/event_suppression_checkpoints.zip"
unzip -t "$M188_DIR/event_suppression_checkpoints.zip"
unzip -l "$M188_DIR/event_suppression_checkpoints.zip"
sha256sum "$M188_DIR/event_suppression_checkpoints.zip"
```

不在 `EV39` 环境中直接 `pip install -r` 或升级 torch。官方源码面向较新的 PyTorch，原样调用
`torch.meshgrid(indexing="ij")` 与 `torch.load(weights_only=False)` 会分别在 EV39 的 Torch `1.9.1` 上报错。
因此只在项目外的 `D:\AI\ESOD\_m188_event_suppression\source` 增加两个 API 兼容分支：旧 Torch 的
`meshgrid` 回退到其默认的同一 `ij` 约定，`torch.load` 在不支持 `weights_only` 参数时回退为普通加载。
它们不改变网络参数、checkpoint key、输入表示或预测公式；严禁把该外部补丁复制进 EVSOD 主工程。

兼容后的无权重结构 smoke 已在正式 `EV39` 环境（Python `3.9.25`、PyTorch `1.9.1+cu111`、RTX 3050 Laptop GPU）
完成：官方 `HydraEVNet` 用 `[1,2,260,346]` 零输入和 `dt=0.05` 可以正常前向，4 个 multi-scale current/future
mask 均为 `[1,1,260,346]`、全部有限；热身后的单窗前向为 `0.062670 s`，峰值 CUDA 分配为 `214.768 MiB`。这只
证明 EV39 可运行结构，不能替代 checkpoint 的 `strict=True` 加载或迁移效果审计。

## 5. 预注册的冻结审计

### 5.1 阶段 A：只能检查运行正确性

下载后先找到包内 EVIMO checkpoint，按官方 `HydraEVNet` 配置创建模型并严格加载
`checkpoint["model_state_dict"]`。必须记录：checkpoint 路径、SHA-256、state key 数、missing/unexpected key 数、
3 个代表窗口的输出 shape、有限性、重复前向最大绝对误差、单视频耗时和峰值显存。

任何 strict-load 错误、非有限输出、表示契约不一致、单视频超过 30 秒或显存超过 3.2 GiB，直接停止 M188；
不修改 key、不 `strict=False`、不换窗口长度补救。

### 5.1a 阶段 A 实测结果（已完成）

官方下载包与解压出的 checkpoint 已校验：压缩包为 `370,156,537` bytes，SHA-256 为
`a092c53577ca4aa36dd4b6a175a27de683a1a19778f94f69b715ea53b96bcb0b`；实际使用的
`model_epoch_49.pth` 为 `407,924,163` bytes，SHA-256 为
`8b1851aa75710dc652300d9db09d43e37506f4f86667ecbb0ccf60b303f86283`。

在 EV39 下，官方 `model_state_dict` 与 `HydraEVNet` 都是 `76` 个 key，`strict=True` 加载成功，没有
missing/unexpected key。三个固定代表视频 `train_080/train_067/train_096` 的每个 50-unit 时间窗都使用
官方等价的 signed-and-normalized two-bin voxel；对 `train_096` 的 bin `0/79/159` 又逐位复核为与 pinned
`VoxelGrid` 一致。每个视频从 reset state 连续前向两次，保存窗口的 logits 都逐位相同。

| 视频 | event 数 | 时间窗数 | 单遍耗时 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: |
| `train_080` | 7,230 | 153 | 7.87 s | 224.4 MiB |
| `train_067` | 32,582 | 160 | 8.11 s | 224.4 MiB |
| `train_096` | 625,178 | 160 | 8.36 s | 224.4 MiB |

需要特别谨慎的事实是，EV-UAV 上的 `sigmoid(dynamic-mask)` 几乎饱和在背景值：常见范围约
`0.26894--0.27003`，高密度 `train_096` 的首窗最大才为 `0.35389`。这既可能是严重域偏移，也可能仍保留弱排序
信息；因此不能仅看幅度直接宣布失败，也不能据此改变预注册的候选门槛。阶段 B 会保留原定
`max(P41, M188) >= 0.10`，并把由此产生的候选膨胀作为审计结果的一部分。

### 5.2 阶段 B：训练视频冻结分数，不读公开验证标签

仅使用 99 个 `train/` 视频，按完整 50-unit 时间窗运行外部模型，将
`sigmoid(dynamic_mask_logits)` 以 `max_pool2d(kernel=4, stride=4, ceil_mode=True)` 映射为与 P41 一致的
`65 x 87` cell score。候选是所有满足

```text
max(P41 score, M188 dynamic-mask score) >= 0.10
```

的**事件活跃** cell；该条件、时间窗、pooling 和 score 方向在读取任意标签前冻结到磁盘。不得使用视频名、
target id、验证标签或人工浏览结果选择窗口。

冻结后才使用既有 `m75_video_folds.json`：循环 `3 fold fit / 1 fold calibration / 1 fold outer`。只允许三个
预先固定的线性 BCE 头：`P41-only`、`M188-only`、`P41+M188`；背景门限为 calibration fold 的最大背景 logit
加 `1e-6`。不扫阈值、融合权重、层、seed、候选下限或时间窗。

为了避免全量五折成本建立在偶然性上，先运行相互独立的 outer `0` 和 `2`。两折都通过后，才运行余下三折。

### 5.3 继续门槛

M188 只有同时满足以下条件才允许进入融合原型；否则永远停止，不训练适配器、不做公开验证细扫：

```text
残余漏检时间窗候选覆盖 >= 0.85，且每个先验 outer fold >= 0.70
P41+M188 相对 P41-only 的 pooled AUC 增益 >= +0.030
视频中位 AUC 增益 >= +0.020
5 个 outer 中至少 4 个正向（前两折必须都正向，且各自 >= +0.030）
零背景 calibration 门限下至少恢复 20 个既有漏检时间窗
```

即使通过上述排序审计，第一版生产融合也只能让 M188 作为保守的 M124 组件特征或 P41 残余候选证据，先做完整
24 视频回归；只有总 `Score` 至少高于 `0.9642995840` 且不牺牲 Pd/Fa，才有资格取代当前 B 版本。

### 5.4 阶段 B 实测结果与停止决定（已完成）

无标签缓存脚本为 `log/analysis/cache_m188_event_suppression_cells.py`，在 99 个训练视频完成一次连续
reset-state 前向，耗时 `826.26s`（均值 `8.35s/视频`），写出 99 个不可覆盖 shard。缓存只保存 raw
`ev.x/ev.y/ev.t/ev.p` 衍生结果和冻结 M111/P41 分数；不保存 `seg_label`、`idx_label` 或 target id。

预注册的 `max(P41,M188)>=0.10` 不能更改。它导致 M188 的背景基线 `0.26894` 使所有 event-active H/4 cell
入选：`6,462,116 / 6,462,116 = 100%`。全缓存动态掩码分数均值为 `0.26894210`、标准差仅
`0.00034236`、最大值 `0.62532824`。这说明少数响应确实存在，但没有足够跨视频稳定性。

随后 `log/analysis/eval_m188_event_suppression_oof.py` 仅运行预注册的独立 outer `0`、`2`，每个都使用
`3 fold fit / 1 fold calibration / 1 fold outer`，三个固定线性 BCE 头为 P41-only、M188-only、P41+M188；
公开验证视频完全没有被读取。报告为 `log/analysis/m188_event_suppression_outer02_oof.json`：

| outer | P41 AUC | M188 AUC | P41+M188 AUC | 融合相对 P41 | 视频中位 AUC 增量 | 严格阈值恢复 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | `0.96291659` | `0.60016226` | `0.96281803` | `-0.00009856` | `-0.00006901` | `0 / 159` |
| 2 | `0.96295244` | `0.57233009` | `0.96295012` | `-0.00000232` | `-0.00002469` | `0 / 187` |

两折候选覆盖都为 `100%`，但两个融合增量都不为正，更远低于每折 `+0.03` / `+0.02` 的继续门槛；M188-only
也远弱于 P41。故不运行余下三折，不训练 adapter，不扫描阈值/融合权重，也不在公开验证集进行生产链尝试。
这证明官方模型可以正确运行，却没有为 EV-UAV 提供可部署的独立排序信息。

## 6. 与“最终公布视频、标签不公开”规则的关系

最终推理只使用每个公开视频自身的事件流和固定外部权重。训练标签只在离线视频留出评估中用于判断是否值得采用，
从不进入最终逐视频选择、阈值或模型更新。因此若 M188 通过，它满足最终测试视频公开但标签隐藏的规则；若必须用
公开验证标签来挑某个视频的阈值或分支，就立即判为不可采用。

## 7. 当前决定

M188 **否决**：其外部动态掩码权重、输入几何和模型状态均已正确审计，但冻结特征在两个独立视频留出折没有提供
P41 之外的稳定目标/背景排序信息。当前可复现最高仍为
`M111 + M169 + P32(seed=3) + M124(0.89)`，完整公开验证 `Score=0.9642995840`；M188 不修改该配置、README、
checkpoint 或提交文件。

本次结论同时收紧了后续外部模型筛选标准：只有“可 strict-load、输入几何兼容、冻结跨视频排序显著优于 P41、并且
能在独立零背景阈值下恢复漏检”的新表征，才值得进入总分生产链。不能仅凭能够运行、少数窗口有较高激活或单视频
视觉响应重新启动训练。RVT/TESPEC/SAST/E2VID、流/轨迹、稀疏 3D、query、EV-Flying 弱框预训练和现在的 M188
均不得以换名或增加 epoch 的方式重开。

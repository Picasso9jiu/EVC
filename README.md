# EVSOD

## EV-UAV Challenge 2 当前最优方案复现

本分支保存 EV-UAV Challenge 2 当前已验证的事件级微小目标检测方案：按输入事件数路由的
全事件流双向时序记忆网络，并在高密度分支加入时序自注意力、目标质心监督的有界平流对齐、半时间箱相位集成和长期背景组件 verifier。仓库包含复现当前分数所需的
代码、固定配置、验证脚本、提交生成脚本和 checkpoint；无需重新训练即可直接验证。

本仓库同时提供中文实验日志 [`note.md`](note.md)。日志按实验编号记录各方向的动机、运行命令、
验证结果、失败原因和停止结论。开始新的优化前请先阅读该文件，避免重复已经否决的方向；日志仅用于
研究参考，不影响免训练评估和提交流程。

项目基于 ICCV 2025 EV-UAV 官方基线实现整理。EV-SpSegNet、EV-UAV 数据集和原始预训练
资源的版权归原论文作者所有。

以下分数来自 `val/` 的 24 个视频，是本地验证结果，不代表未知官方测试集分数。不同 CUDA、
PyTorch、spconv 或 HAIS_OP 编译版本可能造成轻微数值差异。

## 当前最高验证候选：M111 + M124

在正式 M26/P41 基线之后，当前公开 24 个验证视频上的最高完整分数为：

| 指标 | 数值 |
| --- | ---: |
| IoU | **0.9425080419** |
| Acc | 0.9762769938 |
| Pd | 0.9785804284 |
| Fa | **4.6129243890e-06** |
| Score_Fa | **0.9549185368** |
| Score | **0.9640370402** |

该候选在 M111（三个独立相位专家等权平均）上增加 M124 长期背景 verifier。verifier 只使用当前完整
视频的事件坐标、时间和模型分数，在 P0/P0c/P18 后删除背景概率不低于 `0.90` 的最终组件；不使用
视频名、target id 或测试标签。训练好的 verifier 文件为
`checkpoints/m124_m115_long_background_verifier_v1.pkl.gz`，其特征定义和训练统计见 `note.md`。
该分数是本地公开验证集结果，最终测试集仍需使用同一固定开关，不应做逐视频标签调参。注意：固定 `0.90`
在训练视频五折 OOF 中会误删少量含目标组件并使 Pd 下降，因此它是当前最高**提交候选**，而不是已证明
跨域安全的正式替代；保守正式基线仍是 M26/P41 的 `0.9638562171`。

相对正式 M26/P41 基线，Score 提升 `+0.0001808231`，Pd 不变，IoU 提升 `+0.0002057553`，Fa 下降。
M26/P41 基线结果仍保留如下，便于定位 verifier 或环境差异：

| 指标 | 数值 |
| --- | ---: |
| IoU | 0.9423022866 |
| Acc | 0.9763227701 |
| Pd | 0.9785804284 |
| Fa | 4.6632902944e-06 |
| Score_Fa | 0.9544377181 |
| Score | **0.9638562171** |

评分由仓库内的 Challenge 2 评估器计算：

```text
Score_Fa = exp(-10000 * Fa)
Score = 0.4 * Pd + 0.3 * Score_Fa + 0.2 * IoU + 0.1 * Acc
```

M26 从 M20 epoch 003 零扰动挂接，训练保存 12 个 checkpoint。最佳 Challenge 2 checkpoint 是
**epoch 003**；在高密度路由叠加固定 P41 相位集成、P6、P0/P0c 与 P18-global 后得到正式基线。训练 loss 最低的是 epoch 011，验证分数反而明显更低，不能以训练 loss 选模型。

## M26 方案组成

| 环节 | 固定设置 | 作用 |
| --- | --- | --- |
| 低密度路由 | `event_count <= 30000` 使用 M10 epoch 002 | 保留低密度视频上更稳定的专家 |
| 高密度路由 | `event_count > 30000` 使用 M26 epoch 003 | 当前主模型 |
| M26 基础 | M20 的双向 ConvGRU + 时序自注意力记忆网络 | 继承已验证的全事件流时序表征 |
| M26 新增模块 | 有界平流对齐和目标质心位移监督 | 让递归记忆随标注目标的相邻时间箱运动对齐 |
| 训练采样 | `event_count > 200000` 的视频每轮使用 8 个确定性视图 | 提高高密度输入的时序覆盖 |
| P41 相位集成 | 仅 `event_count > 30000`，时间偏移 `25`，原流/偏移流权重 `0.75/0.25` | 降低 50-unit 时间分箱边界对高密度目标轨迹的影响 |
| P6 阈值 | 低密度 `0.718`，其他 `0.7226` | 生成最终事件标签的阈值 |
| P0/P0c | 半径 2、相邻 1 个时间箱、最少 3 事件和 5 时间箱、保留分数 0.95 | 过滤弱时空连通簇，同时保留高置信小簇 |
| P18-global | 不限制事件数、候选下限 0.53、半径 5、连接距离 8、最少 4 时间箱 | 每个稳定弱轨迹组件仅恢复一个最优事件 |

M26 的 flow 输出层采用零初始化。因此从 M20 加载时 flow 严格为零、warp 是精确恒等，初始预测不变；训练期再通过同一 `target_id` 的相邻时间箱质心监督逆位移。flow 以 `2.0 * tanh(raw_flow)` 限制在 2 个 bottleneck 单元内，避免异常位移破坏已收敛的记忆状态。

## 固定推理顺序

1. 读取完整原始事件流，按宽度 `50` 的时间箱构建上下文为 5 的时序输入帧。
2. 事件数不超过 30000 的视频路由到 M10，其余视频使用 M26。
3. 高密度路由额外以时间偏移 `25` 构造第二个流，原流与偏移流的逐事件概率按 `0.75/0.25` 融合；低密度 M10 路由不执行该步骤。
4. 按 P6 的密度自适应阈值生成初始二值事件。
5. 应用 P0 时空连通簇过滤，再应用 P0c 高置信恢复。
6. 应用 P18-global 弱轨迹恢复，每个符合条件的组件仅恢复一个最优事件。
7. M124 verifier 在最终阈值化前删除背景概率不低于 `0.90` 的最终组件。
8. 生成提交时保留原始 `x y t p`，仅写入最终二值 `label`。

所有路由条件都只依赖可观察的输入事件数。推理过程中不读取验证标签、目标 ID 或视频名称规则。

## 已包含权重

| 文件 | 用途 | SHA-256 |
| --- | --- | --- |
| `checkpoints/m4_dacc_m5_best_loss_seed42.pt` | 完整重训链条的版本化起点 | M13-FT30 的初始化模型 |
| `checkpoints/m10_dense_views2_epoch_002_seed42.pt` | 固定低密度路由模型 | `5C89C89A165469C0A4E8286D4644D60D2F82CF5775EDBB724F626E24E67D8935` |
| `checkpoints/m20_attn_dense_views8_epoch_003_seed48.pt` | 固定高密度 M20 模型 | `4B8B2B19EA9D913EE4E52CB21AE52BF945B2B0F3CEFD5CB5AB6F64D51BF49849` |
| `checkpoints/m26_targetflow_m20e3_epoch_003_seed53.pt` | 固定高密度 M26 模型 | `13F7D4D8AB6BDCAAA98F3F906A7D32E687C17454B88B42B94752EEC04257F7C4` |
| `checkpoints/m111_phase_specialist_seed72_73_76_average.pt` | M111 三 seed 相位专家平均 | `15FD690E3BB177649F2A995AAF00B947FD6AC0DFB147EA6A9CDF5847B96ADED2` |
| `checkpoints/m124_m115_long_background_verifier_v1.pkl.gz` | M124 纯背景组件 verifier | `B39A2DD93A2B6B499F558338FCFEE3B7BD20F8CECCFB66499549617E832D1C8B` |

直接评估当前最高候选需要 M10、M26、M111 和 M124 verifier；P41 是无参数的推理期集成。M20 是 M26 的训练初始化，M4 只在下文的完整重训链条中使用。

## 仓库结构

```text
EVSOD-main/
|-- checkpoints/                 # 当前最高方案的 M10、M26、M111 与 M124 权重
|-- configs/evisseg_evuav.yaml   # 固定配置
|-- dataset/                     # 数据集目录，不上传 Git
|-- model/temporal_memory_net.py # ConvGRU、时序自注意力和有界平流对齐
|-- utils/                       # 全事件流推理、评估器和 M124 verifier
|-- train_temporal_memory.py     # M13/M15/M20/M26 训练入口
|-- test2.py                     # 本地 Challenge 2 验证
|-- submit_challenge2.py         # 提交 TXT 生成
`-- README.md                    # 当前最优方案复现文档
```

`log/`、数据集、生成的提交文件以及本地 HAIS_OP 编译产物均不上传 Git。首次使用时需要在目标
环境中编译 HAIS_OP。

## 环境配置

已验证环境：WSL/Ubuntu、Python 3.9、PyTorch 1.9.1 + CUDA 11.1、torchvision 0.10.1、
spconv-cu111、NumPy 1.23.5，以及 CUDA 11.x Toolkit。

```bash
git clone --branch evsod-main https://github.com/Picasso9jiu/EVC.git EVSOD-main
cd EVSOD-main

conda create -n EV39 python=3.9 pip -y
conda activate EV39

python -m pip install --upgrade pip
python -m pip install \
  torch==1.9.1+cu111 torchvision==0.10.1+cu111 \
  -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install \
  numpy==1.23.5 pyyaml==6.0.2 tqdm==4.66.5 pandas==2.0.3 \
  opencv-python==4.8.1.78 mlflow==2.17.2 spconv-cu111 \
  typing-extensions==4.12.2 pillow==10.4.0 scikit-learn==1.6.1
```

安装 PyTorch 后首次编译 HAIS_OP。系统需要兼容的 CUDA Toolkit、C++ 编译器和
`libsparsehash-dev`：

```bash
sudo apt update
sudo apt install -y build-essential libsparsehash-dev ninja-build

export PROJECT_DIR="$(pwd)"
cd "$PROJECT_DIR/lib/hais_ops"
python setup.py build_ext develop

export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
TORCH_CUDART="$(find "$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib" -maxdepth 1 -name 'libcudart-*.so.11.0' -print -quit)"
if [ -n "$TORCH_CUDART" ]; then export LD_PRELOAD="$TORCH_CUDART${LD_PRELOAD:+:$LD_PRELOAD}"; fi
cd "$PROJECT_DIR"
python -c "import torch; import spconv.pytorch; import HAIS_OP; print(torch.cuda.is_available(), 'HAIS_OP: ok')"
```

每次打开新终端后，重新激活环境并设置路径：

```bash
conda activate EV39
export PROJECT_DIR=/absolute/path/to/EVSOD-main
export DATA_ROOT="$PROJECT_DIR/dataset/训练集、验证集"
export PYTHONPATH="$PROJECT_DIR/lib/hais_ops/build/lib.linux-x86_64-cpython-39:$PROJECT_DIR:$PYTHONPATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$CONDA_PREFIX/lib:/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
TORCH_CUDART="$(find "$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib" -maxdepth 1 -name 'libcudart-*.so.11.0' -print -quit)"
if [ -n "$TORCH_CUDART" ]; then export LD_PRELOAD="$TORCH_CUDART${LD_PRELOAD:+:$LD_PRELOAD}"; fi
cd "$PROJECT_DIR"
```

## 数据准备

从官方渠道下载 EV-UAV Challenge 2 数据包，放置为：

```text
dataset/训练集、验证集/
|-- train/       # 99 个 .npz 视频
|-- val/         # 24 个 .npz 视频
`-- val_Challenge2.py
```

数据集不随 Git 发布。官方数据可从[百度网盘](https://pan.baidu.com/s/15pAlu3KP1uXych-c3SC5qA?pwd=sbr2)
（提取码 `sbr2`）或 [Google Drive](https://drive.google.com/drive/folders/1VIkBFx5Po0KPIFBYOL_appLVie5wgdyi?usp=drive_link) 下载。

## 复现流程

本仓库提供两条路径：路径 1 直接验证已发布的当前最高分，适合提交和结果核对；路径 2 从仓库
版本化的 M4+DACC+M5 起点按完整训练链条重新生成 M26，适合研究复现。两条路径都使用前文完成的
环境配置和数据目录。

### 1. **免训练评估**：直接复现当前最高候选

下列命令使用 M10/M26/M111 和 M124 verifier，在 24 个验证视频上直接复现当前最高候选，不需要训练。
在已验证 GPU 环境中通常需要数分钟。

```bash
M10_CKPT="$PROJECT_DIR/checkpoints/m10_dense_views2_epoch_002_seed42.pt"
M26_CKPT="$PROJECT_DIR/checkpoints/m26_targetflow_m20e3_epoch_003_seed53.pt"
M111_CKPT="$PROJECT_DIR/checkpoints/m111_phase_specialist_seed72_73_76_average.pt"
M124_VERIFIER="$PROJECT_DIR/checkpoints/m124_m115_long_background_verifier_v1.pkl.gz"

python test2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.eval=true TEST.roc=true TEST.prediction_threshold=0.7226 \
  TEMPORAL_FRAME.temporal_frame_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$M26_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_model_path="$M10_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000 \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_model_path="$M111_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_event_count_cutoff=30000 \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_weight=0.25 \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_offset=25 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0 \
  TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8 \
  INFERENCE_TTA.p41_temporal_phase_enabled=true \
  INFERENCE_TTA.p41_temporal_phase_offset=25 \
  INFERENCE_TTA.p41_temporal_phase_original_weight=0.75 \
  INFERENCE_TTA.p41_temporal_phase_min_event_count=30000 \
  POSTPROCESS.p0_enabled=true POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 POSTPROCESS.p0_min_duration_bins=5 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.95 POSTPROCESS.p0b_enabled=false \
  POSTPROCESS.p18_score_track_recovery_enabled=true \
  POSTPROCESS.p18_event_count_cutoff=1 POSTPROCESS.p18_max_event_count=0 \
  POSTPROCESS.p18_candidate_floor=0.53 POSTPROCESS.p18_spatial_radius=5 \
  POSTPROCESS.p18_temporal_bin_size=50 POSTPROCESS.p18_max_link_distance=8.0 \
  POSTPROCESS.p18_max_gap_bins=1 POSTPROCESS.p18_min_track_bins=4 \
  POSTPROCESS.p18_restore_mode=best \
  POSTPROCESS.p6_density_threshold_enabled=true \
  POSTPROCESS.p6_event_count_cutoff=30000 \
  POSTPROCESS.p6_low_density_threshold=0.718 \
  POSTPROCESS.p6_high_density_threshold=0.7226 \
  POSTPROCESS.m124_background_verifier_enabled=true \
  POSTPROCESS.m124_background_verifier_model_path="$M124_VERIFIER" \
  POSTPROCESS.m124_background_verifier_threshold=0.90
```

P41 的四项覆盖必须使用 `INFERENCE_TTA.p41_*` 前缀。配置加载会把分组字段展平；若误写为
`TEMPORAL_MEMORY.p41_*`，后续的 `INFERENCE_TTA` 默认值会将其覆盖，实际不会启用 P41，也就无法复现本节分数。

预期输出接近：

```text
IoU:      0.9425080419
Acc:      0.9762769938
Pd:       0.9785804284
Fa:       4.6129243890e-06
Score_Fa: 0.9549185368
Score:    0.9640370402
```

### 1.1 正式 M26/P41 基线回归

若需要排查环境差异或对比 verifier 增益，将上一条命令中的 M111 五项覆盖和 M124 三项覆盖去掉，
即可得到正式 M26/P41 基线，预期 `Score=0.9638562171`。M124 配置默认关闭，旧命令无需修改即可保持
该基线。

### 2. 从版本化起点完整重训 M26

当前 M26 的完整训练链为：M4+DACC+M5 -> M13-FT30 epoch 003 -> M15 epoch 008 ->
M20 attention epoch 003 -> M26 target-flow epoch 003。每一步均只读取 `train/`；最终分数使用第 1 步的完整 `test2.py`
验证。训练时间较长，且不同硬件环境可能产生轻微数值差异。

#### 2.1 训练 M13-FT30

M13-FT30 从仓库提供的 M4+DACC+M5 checkpoint 初始化，固定训练 30 个 epoch。即使训练 loss
在其他轮更低，后续链条固定使用 epoch 003。

```bash
M4_CKPT="$PROJECT_DIR/checkpoints/m4_dacc_m5_best_loss_seed42.pt"
M13_ROOT="$PROJECT_DIR/log/m13_dense_views4_ft30_seed42"

python train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=42 TRAIN.epochs=30 TRAIN.batch_size=1 TRAIN.lr=0.00002 \
  TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.000001 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$M13_ROOT" \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M4_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=4 \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_weight=0.05 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_margin_logit=1.0 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_min_points=3 \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_warmup_epochs=3

M13_E3="$(find "$M13_ROOT/runs" -type f -name 'epoch_003_seed42.pt' -print -quit)"
test -n "$M13_E3"
```

#### 2.2 训练 M15

M15 从 M13-FT30 epoch 003 低学习率续训 8 个 epoch。固定使用 M15 epoch 008 作为下一步初始化，
不要以训练 loss 最低的 epoch 004 替代。

```bash
M15_ROOT="$PROJECT_DIR/log/m15_e3_low_lr_seed43"

python train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=43 TRAIN.epochs=8 TRAIN.batch_size=1 TRAIN.lr=0.000003 \
  TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.0000003 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$M15_ROOT" \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M13_E3" \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=1.0 \
  TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=4 \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=false

M15_E8="$(find "$M15_ROOT/runs" -type f -name 'epoch_008_seed43.pt' -print -quit)"
test -n "$M15_E8"
```

#### 2.3 训练 M20 attention

M20 在 M15 epoch 008 上附加零初始化时序自注意力残差。训练保存每轮 checkpoint；当前固定使用
epoch 003，而非训练 loss 最低的 epoch 011。

```bash
M20_ROOT="$PROJECT_DIR/log/m20_attn_dense_views8_e12_seed48"

python train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=48 TRAIN.epochs=12 TRAIN.batch_size=1 \
  TRAIN.lr=0.000001 TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.0000001 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$M20_ROOT" \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M15_E8" \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=8 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_metric_aux_enabled=false \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_FRAME.temporal_frame_trajectory_extrapolation_enabled=false

M20_E3="$(find "$M20_ROOT/runs" -type f -name 'epoch_003_seed48.pt' -print -quit)"
test -n "$M20_E3"
```

#### 2.4 训练 M26 target-flow 平流对齐

M26 在 M20 epoch 003 上附加零初始化的 flow head。目标质心监督只在训练期使用；推理只依赖模型权重。保持 M20 的 dense 8x 采样，训练 12 个 epoch 并保存每轮 checkpoint；固定使用 epoch 003，而非最低训练 loss 的 epoch 011。

```bash
M26_ROOT="$PROJECT_DIR/log/m26_targetflow_m20e3_dense8_e12_seed53"

python train_temporal_memory.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TRAIN.seed=53 TRAIN.epochs=12 TRAIN.batch_size=1 \
  TRAIN.lr=0.000001 TRAIN.scheduler=cosine TRAIN.scheduler_min_lr=0.0000001 \
  TRAIN.checkpoint_interval=1 TRAIN.model_save_root="$M26_ROOT" \
  TEMPORAL_FRAME.temporal_frame_density_calibration_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_init_model_path="$M20_E3" \
  TEMPORAL_MEMORY.temporal_memory_dense_sampling_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_dense_event_count_cutoff=200000 \
  TEMPORAL_MEMORY.temporal_memory_dense_view_multiplier=8 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_base_lr_multiplier=0.25 \
  TEMPORAL_MEMORY.temporal_memory_memory_lr_multiplier=0.25 \
  TEMPORAL_MEMORY.temporal_memory_attention_lr_multiplier=0.25 \
  TEMPORAL_MEMORY.temporal_memory_advection_alignment_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_advection_alignment_loss_weight=0.01 \
  TEMPORAL_MEMORY.temporal_memory_advection_alignment_lr_multiplier=4.0 \
  TEMPORAL_MEMORY.temporal_memory_advection_max_flow=2.0 \
  TEMPORAL_MEMORY.temporal_memory_advection_target_flow_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_advection_target_flow_weight=0.5 \
  TEMPORAL_MEMORY.temporal_memory_advection_target_flow_huber_delta=1.0

M26_E3="$(find "$M26_ROOT/runs" -type f -name 'epoch_003_seed53.pt' -print -quit)"
test -n "$M26_E3"
```

将第 1 步中的 `M26_CKPT` 改为 `$M26_E3`，使用完全相同的完整验证命令（包括 P41 与 P18-global）评估重新训练的模型。只有完整验证达到或超过预期时，才可用该新权重替换发布的 M26 checkpoint。

### 3. 生成 Challenge 2 提交文件

提交必须使用与 **免训练评估** 完全相同的 M10/M26/M111/M124 权重及固定参数，只将验证选项替换为输出目录：

```bash
OUTPUT_DIR="$PROJECT_DIR/log/challenge2/m111_m124_current_best"
M10_CKPT="$PROJECT_DIR/checkpoints/m10_dense_views2_epoch_002_seed42.pt"
M26_CKPT="$PROJECT_DIR/checkpoints/m26_targetflow_m20e3_epoch_003_seed53.pt"
M111_CKPT="$PROJECT_DIR/checkpoints/m111_phase_specialist_seed72_73_76_average.pt"
M124_VERIFIER="$PROJECT_DIR/checkpoints/m124_m115_long_background_verifier_v1.pkl.gz"

python submit_challenge2.py --config configs/evisseg_evuav.yaml --set \
  DATA.root="$DATA_ROOT" \
  TEST.challenge_output_dir="$OUTPUT_DIR" TEST.prediction_threshold=0.7226 \
  TEMPORAL_FRAME.temporal_frame_enabled=false \
  TEMPORAL_MEMORY.temporal_memory_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_model_path="$M26_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_model_path="$M10_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_secondary_max_event_count=30000 \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_model_path="$M111_CKPT" \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_event_count_cutoff=30000 \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_weight=0.25 \
  TEMPORAL_MEMORY.temporal_memory_phase_specialist_offset=25 \
  TEMPORAL_MEMORY.temporal_memory_temporal_attention_enabled=true \
  TEMPORAL_MEMORY.temporal_memory_sparse_weight=0.0 \
  TEMPORAL_MEMORY.temporal_memory_inference_batch_size=8 \
  INFERENCE_TTA.p41_temporal_phase_enabled=true \
  INFERENCE_TTA.p41_temporal_phase_offset=25 \
  INFERENCE_TTA.p41_temporal_phase_original_weight=0.75 \
  INFERENCE_TTA.p41_temporal_phase_min_event_count=30000 \
  POSTPROCESS.p0_enabled=true POSTPROCESS.p0_spatial_radius=2 \
  POSTPROCESS.p0_temporal_bin_size=50 POSTPROCESS.p0_temporal_radius_bins=1 \
  POSTPROCESS.p0_min_cluster_events=3 POSTPROCESS.p0_min_duration_bins=5 \
  POSTPROCESS.p0c_high_confidence_recovery_enabled=true \
  POSTPROCESS.p0c_retain_min_score=0.95 POSTPROCESS.p0b_enabled=false \
  POSTPROCESS.p18_score_track_recovery_enabled=true \
  POSTPROCESS.p18_event_count_cutoff=1 POSTPROCESS.p18_max_event_count=0 \
  POSTPROCESS.p18_candidate_floor=0.53 POSTPROCESS.p18_spatial_radius=5 \
  POSTPROCESS.p18_temporal_bin_size=50 POSTPROCESS.p18_max_link_distance=8.0 \
  POSTPROCESS.p18_max_gap_bins=1 POSTPROCESS.p18_min_track_bins=4 \
  POSTPROCESS.p18_restore_mode=best \
  POSTPROCESS.p6_density_threshold_enabled=true \
  POSTPROCESS.p6_event_count_cutoff=30000 \
  POSTPROCESS.p6_low_density_threshold=0.718 \
  POSTPROCESS.p6_high_density_threshold=0.7226 \
  POSTPROCESS.m124_background_verifier_enabled=true \
  POSTPROCESS.m124_background_verifier_model_path="$M124_VERIFIER" \
  POSTPROCESS.m124_background_verifier_threshold=0.90

cd "$OUTPUT_DIR"
zip -j ../m111_m124_current_best.zip val_*.txt
```

## 引用

```bibtex
@misc{chen2025eventbasedtinyobjectdetection,
  title={Event-based Tiny Object Detection: A Benchmark Dataset and Baseline},
  author={Nuo Chen and Chao Xiao and Yimian Dai and Shiman He and Miao Li and Wei An},
  year={2025},
  eprint={2506.23575},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2506.23575}
}
```

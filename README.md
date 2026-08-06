# 二自由度机械臂 CBF-RL 源码运行说明

本文面向第一次拿到本仓库的使用者，说明如何在 Windows 原生环境下安装依赖、验证代码、
训练四种配置，并生成论文 Fig.3/Fig.4 风格结果图。当前对象是二维二连杆机械臂的低阶
关节速度模型，不是完整力矩动力学模型；算法细节见 [LOW_ORDER_MODEL.md](./LOW_ORDER_MODEL.md)。

## 1. 实验内容与源码入口

低阶机械臂实验包含四种训练配置：

| 命令行名称 | CBF 安全奖励 | 训练时 CBF-filter |
|---|---:|---:|
| `nominal` | 否 | 否 |
| `reward_only` | 是 | 否 |
| `filter_only` | 否 | 是 |
| `dual` | 是 | 是 |

主要文件如下：

| 文件 | 用途 |
|---|---|
| [vec_env.py](./vec_env.py) | 向量化机械臂环境、CBF 约束、二维 QP、安全奖励和任务奖励 |
| [train_arm2d.py](../train_arm2d.py) | 单种配置的 PPO 训练入口 |
| [run_arm2d_training.ps1](../run_arm2d_training.ps1) | Windows 下依次训练四种配置 |
| [demo.py](./demo.py) | 不使用 RL checkpoint 的 CBF-filter 演示 |
| [plot_arm2d_paper_fig3.py](../plot_arm2d_paper_fig3.py) | 分别绘制奖励曲线和碰撞曲线 |
| [plot_arm2d_fig4_no_dr.py](../plot_arm2d_fig4_no_dr.py) | 无 DR 固定场景轨迹对比及 seed 搜索 |
| [plot_arm2d_paper_figures.py](../plot_arm2d_paper_figures.py) | 一次生成训练曲线和四方法轨迹图 |
| [test_arm2d.py](../tests/test_arm2d.py) | 运动学、雅可比和 CBF-QP 单元测试 |

所有命令都应在仓库根目录执行。以下用 `<repo>` 表示仓库根目录：

```powershell
cd <repo>
```

例如开发机器上的目录是 `D:\cbf_rl\cbf-rl-navigation-demo-main`，但源码本身不要求使用
这个固定路径。

## 2. Windows 环境安装

### 2.1 关于仓库中的 environment.yml

根目录的 `environment.yml` 是最初拿到源码时就存在的上游 Linux 环境快照，其中包含
Linux 构建号和 Linux 安装前缀。它不是本机械臂版本实际使用的 Windows 环境定义，
Windows 用户不要直接运行 `conda env create -f environment.yml`。

当前代码在 Windows 原生 Conda 环境中验证过的主要版本为：

| 软件包 | 版本 |
|---|---|
| Python | 3.10.20 |
| PyTorch | 2.7.0+cu128 |
| NumPy | 1.26.4 |
| SciPy | 1.11.4 |
| Matplotlib | 3.10.7 |
| rsl-rl-lib | 2.3.1 |
| TensorBoard | 2.21.0 |

### 2.2 创建 Windows Conda 环境

```powershell
conda create -n cbf_learning python=3.10 -y
conda activate cbf_learning
python -m pip install --upgrade pip
```

安装 NVIDIA CUDA 12.8 版 PyTorch：

```powershell
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
```

安装其余依赖：

```powershell
python -m pip install numpy==1.26.4 scipy==1.11.4 matplotlib==3.10.7 rsl-rl-lib==2.3.1 tensorboard==2.21.0 pytest
```

如果只做 CPU 验证，可将 PyTorch 安装命令中的索引改为：

```powershell
python -m pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cpu
```

GPU 和 CPU 版本二选一。若本机驱动不支持 CUDA 12.8，请安装与本机驱动兼容的 PyTorch；
其余源码无需改变。低阶机械臂的二维 QP 由源码枚举求解，不依赖 `qpth`、OSQP 等外部
QP 求解器。

### 2.3 检查解释器和 GPU

```powershell
python -c "import torch, rsl_rl, numpy, matplotlib, tensorboard; print('torch=', torch.__version__); print('cuda=', torch.cuda.is_available()); print('device=', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

正式 GPU 训练前，`cuda` 应为 `True`。如果为 `False`，代码仍可通过 `--device cpu`
运行，但训练会明显变慢。

## 3. 第一次运行：最小验证

### 3.1 运行单元测试

```powershell
python -m pytest tests/test_arm2d.py -q
```

测试会验证正运动学、末端雅可比、二维多约束速度投影，以及障碍物和关节限位约束构造。

### 3.2 运行不依赖 checkpoint 的 CBF 演示

```powershell
python -m arm2d.demo --output logs/plots/arm2d_cbf_demo.png
```

程序会对比未过滤名义控制器和 CBF-filter，并生成：

```text
logs/plots/arm2d_cbf_demo.png
logs/plots/arm2d_cbf_demo.pdf
```

这一步不训练 PPO，可先独立确认运动学、碰撞约束和过滤器能够运行。

### 3.3 可选：短时冒烟训练

GPU：

```powershell
python train_arm2d.py --method nominal --seed 42 --num-envs 64 --iterations 5 --steps-per-env 16 --save-interval 5 --device cuda --run-label smoke
```

CPU：

```powershell
python train_arm2d.py --method nominal --seed 42 --num-envs 32 --iterations 2 --steps-per-env 16 --save-interval 2 --device cpu --run-label smoke
```

冒烟训练只检查训练链路，不代表策略已收敛，也不会被只读取 `model_1000.pt` 的正式
Fig.3 脚本选中。

## 4. 正式训练四种配置

默认正式参数为 1024 个并行环境、1000 次 PPO 迭代、每环境每次 rollout 32 步、seed 42。

### 4.1 单独训练

```powershell
python train_arm2d.py --method nominal --seed 42 --num-envs 1024 --iterations 1000 --device cuda
python train_arm2d.py --method reward_only --seed 42 --num-envs 1024 --iterations 1000 --device cuda
python train_arm2d.py --method filter_only --seed 42 --num-envs 1024 --iterations 1000 --device cuda
python train_arm2d.py --method dual --seed 42 --num-envs 1024 --iterations 1000 --device cuda
```

训练入口支持以下参数：

| 参数 | 含义 | 默认值 |
|---|---|---:|
| `--method` | 四种训练方法之一，必填 | 无 |
| `--seed` | 随机种子 | 42 |
| `--num-envs` | 并行环境数 | 1024 |
| `--iterations` | PPO 迭代数 | 1000 |
| `--steps-per-env` | 每次迭代中每环境采样步数 | 32 |
| `--save-interval` | 中间 checkpoint 间隔 | 100 |
| `--device` | `cuda` 或 `cpu` | 自动选择 |
| `--run-label` | 日志目录的可选后缀 | 空 |

### 4.2 一次训练四种配置

在 PowerShell 中运行：

```powershell
powershell -ExecutionPolicy Bypass -File run_arm2d_training.ps1 `
  -Seed 42 `
  -NumEnvs 1024 `
  -Iterations 1000 `
  -PythonExecutable "$env:CONDA_PREFIX\python.exe"
```

脚本按 `nominal → reward_only → filter_only → dual` 的顺序串行训练。如果某一种失败，
脚本会停止，不会继续下一种。

## 5. 日志和 checkpoint

每次训练会创建独立目录：

```text
logs/arm2d/<method>/<时间戳>_seed<seed>[_<run-label>]/
├── experiment_config.json
├── events.out.tfevents...
├── model_100.pt
├── model_200.pt
├── ...
└── model_1000.pt
```

- `experiment_config.json`：本次运行参数以及安全奖励/训练过滤器开关；
- `events.out.tfevents...`：训练曲线和机械臂安全指标；
- `model_*.pt`：策略、价值网络和优化器状态。

实时查看训练过程：

```powershell
tensorboard --logdir logs/arm2d
```

然后打开终端提示的地址，通常为 `http://localhost:6006`。

注意：`.gitignore` 排除了 `logs/` 和 `*.pt`。仅通过 Git 克隆源码不会得到已经训练好的
checkpoint 和 TensorBoard 事件。如果希望别人无需重新训练就能画图，需要另行打包整个
`logs/arm2d/` 目录；接收者应将它放回仓库根目录下的相同位置。

## 6. 生成 Fig.3 风格训练曲线

确认四个方法都存在完整的 `model_1000.pt` 和对应 TensorBoard 事件后运行：

```powershell
python plot_arm2d_paper_fig3.py
```

默认输出：

```text
logs/plots/arm2d_paper_style/arm2d_paper_reward_curve.png
logs/plots/arm2d_paper_style/arm2d_paper_reward_curve.pdf
logs/plots/arm2d_paper_style/arm2d_paper_collision_curve.png
logs/plots/arm2d_paper_style/arm2d_paper_collision_curve.pdf
```

自定义参数示例：

```powershell
python plot_arm2d_paper_fig3.py `
  --log-root logs/arm2d `
  --output-dir logs/plots/arm2d_paper_style `
  --smooth 0.92 `
  --num-envs 1024
```

`--num-envs` 必须与训练时的并行环境数一致，因为碰撞图会把逐环境均值换算成该批次的
事件数量。如果训练时改成 512 个环境，绘图时也应传 `--num-envs 512`。

## 7. 生成 Fig.4 风格无 DR 轨迹图

该图使用四个 checkpoint，并形成六条部署轨迹：Nominal、Dual（开/关运行时过滤器）、
Reward Only、Filter Only（开/关运行时过滤器）。当前机械臂版本没有 DR，因此生成的是
无 DR 结果。

例如尝试 seed 300～399，并在其中自动选择最符合论文消融趋势的场景：

```powershell
python plot_arm2d_fig4_no_dr.py `
  --seed-start 300 `
  --seed-count 100 `
  --device cpu `
  --output logs/plots/arm2d_paper_style/arm2d_fig4_no_dr_seeds_300_399
```

输出包括同名的 `.png`、`.pdf` 和 `.txt`。`.txt` 会记录最终选中的 seed、搜索得分、
六种部署方式的结果和最小安全裕度。

要搜索其他 seed 区间，可运行：

```powershell
python plot_arm2d_fig4_no_dr.py --seed-start 0   --seed-count 100 --output logs/plots/arm2d_paper_style/fig4_seeds_0_99
python plot_arm2d_fig4_no_dr.py --seed-start 100 --seed-count 100 --output logs/plots/arm2d_paper_style/fig4_seeds_100_199
python plot_arm2d_fig4_no_dr.py --seed-start 200 --seed-count 100 --output logs/plots/arm2d_paper_style/fig4_seeds_200_299
```

同一命令中的场景生成是确定性的。脚本优先寻找 Nominal 失败、安全方法成功、关闭运行时
过滤后出现差异的代表性场景。这只是可视化场景筛选，不等于统计性能评测。

## 8. 一次生成训练曲线和四方法轨迹图

```powershell
python plot_arm2d_paper_figures.py `
  --logs-root logs/arm2d `
  --output-dir logs/plots/arm2d_paper `
  --scenario-seed 20260807 `
  --num-scenarios 256 `
  --device cuda
```

该脚本输出训练曲线和 2×2 四方法轨迹图。它与 `plot_arm2d_fig4_no_dr.py` 的区别是：
前者只比较四种方法按其训练配置部署；后者还比较 Dual 和 Filter Only 关闭运行时过滤器，
更接近论文 Fig.4 的消融表达。

## 9. 推荐执行顺序

```text
1. 创建并激活 Windows cbf_learning 环境
2. 安装 PyTorch、rsl-rl-lib、TensorBoard 等依赖
3. python -m pytest tests/test_arm2d.py -q
4. python -m arm2d.demo
5. 运行 2～5 次迭代的 nominal 冒烟训练
6. 正式训练 nominal / reward_only / filter_only / dual
7. 检查四个 model_1000.pt 和 TensorBoard 事件文件
8. 运行 plot_arm2d_paper_fig3.py
9. 用 plot_arm2d_fig4_no_dr.py 搜索多个 seed 区间
10. 保存图片、experiment_config.json 和 Fig.4 的 txt 结果
```

## 10. 常见问题

### 10.1 `ModuleNotFoundError: No module named 'arm2d'`

先 `cd` 到包含 `train_arm2d.py` 的仓库根目录，再执行命令。

### 10.2 找不到 `rsl_rl`、`tensorboard` 或 `pytest`

确认已激活正确环境，并运行：

```powershell
python -m pip install rsl-rl-lib==2.3.1 tensorboard==2.21.0 pytest
where python
```

### 10.3 `torch.cuda.is_available()` 为 `False`

检查 NVIDIA 驱动、PyTorch 是否为 CUDA wheel，以及当前 Python 是否来自刚创建的环境。
可先添加 `--device cpu` 验证代码，正式训练建议先解决 CUDA 环境。

### 10.4 GPU 显存不足

将 `--num-envs` 从 1024 降到 512 或 256。并行环境数会改变每次 PPO 更新的数据量，结果
不再与默认配置完全等价；Fig.3 的 `--num-envs` 也必须同步修改。

### 10.5 Fig.3 报 `No completed Arm2D TensorBoard run`

Fig.3 脚本只读取同时具有 TensorBoard 事件和 `model_1000.pt` 的目录。请确认四种方法都
完成了 1000 次迭代。

### 10.6 Fig.4 报找不到 checkpoint

确认以下四个路径各至少有一个 `model_*.pt`：

```text
logs/arm2d/nominal/<run>/
logs/arm2d/reward_only/<run>/
logs/arm2d/filter_only/<run>/
logs/arm2d/dual/<run>/
```

Fig.4 会优先寻找 `model_1000.pt`，不存在时才回退到其他 `model_*.pt`。

### 10.7 相同 seed 仍有细微差异

脚本固定了 Python、NumPy 和 PyTorch 随机种子，但不同 GPU、驱动、CUDA 和 PyTorch 版本
仍可能产生细微差异。发布结果时应同时保存运行配置、依赖版本、TensorBoard 事件、最终
checkpoint、绘图命令和 Fig.4 的 `.txt` 结果。

## 11. 当前范围

本手册只覆盖 `arm2d` 低阶机械臂实验。根目录中原有导航环境的 DR 训练与 Table III 评测
不能和机械臂 checkpoint 混用。

当前机械臂版本使用关节速度输入、一阶 CBF 和二维多约束 QP，已实现四种无 DR 训练配置
与无 DR 轨迹图；尚未实现机械臂 DR、完整动力学/HOCBF，以及标准化多 episode 统计表。
因此 Fig.4 的 seed 搜索图用于展示代表性轨迹，不应单独作为总体性能的统计结论。

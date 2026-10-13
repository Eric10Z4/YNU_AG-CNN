# YNU_AG-CNN：手写 CNN 与并行 MCTS 的 AlphaZero 五子棋系统

围绕“**手写神经网络 → 策略价值预测 → MCTS 搜索 → 自博弈训练**”构建的强化学习实验项目。项目以 PyTorch 张量为计算基础，手动实现网络前向与反向传播，并接入可选的 C++ 搜索后端，连接底层数值计算、决策搜索与训练工程。

**技术栈：** Python · C++17 · PyTorch · NumPy · pybind11 · CMake · TensorBoard

## 项目实现要点

| 方向 | 实现内容 | 代码入口 |
| --- | --- | --- |
| 手写 CNN | 卷积、BatchNorm、线性层及对应梯度计算；组合策略头与价值头 | [core](core/)、[policy_value_net.py](pipeline/policy_value_net.py) |
| 自博弈学习 | 由搜索结果生成策略目标，结合对局结果更新网络，维护经验回放与评估过程 | [Python 训练](gat/train.py)、[训练入口](gat_cpp/cpp_train.py) |
| C++ 搜索后端 | 棋盘状态、共享搜索树、virtual loss、常驻工作线程与神经网络批量评估接口 | [mcts_stub.cpp](gat_cpp/src/mcts_stub.cpp)、[bindings.cpp](gat_cpp/src/bindings.cpp) |
| 训练可靠性 | 完整网络存档、恢复前校验、随机状态记录、配置快照与训练日志 | [checkpoint.py](pipeline/checkpoint.py)、[回归测试](tests/test_checkpoint.py) |

这些模块提供了从算子实现到搜索决策的可阅读路径，也便于在同一任务下对比 Python 与 C++ 后端的行为。具体运行方式、检查点格式和验证范围见下文。

原始思路与部分实现基于 [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)。项目实现包含团队协作与上游工作，来源和授权边界见 [AUTHORIZATION.md](AUTHORIZATION.md)。

## 目录

| 目录 | 用途 |
| --- | --- |
| `core/` | 手写卷积、归一化、线性层、激活及基础算子 |
| `pipeline/` | 手写策略价值网络、优化器、检查点验证 |
| `gat/` | Python 棋盘、搜索、训练和人机对弈入口 |
| `gat_cpp/` | C++ 棋盘与搜索、Python 绑定及训练入口 |
| `tests/` | 检查点完整性与真实梯度更新的回归检查 |

## 安装

建议使用 Python 3.12 独立环境。直接依赖固定为 NumPy 2.2.6、PyTorch 2.6.0、TensorBoard 2.19.0；传递依赖没有完整锁定。

```sh
python -m venv .venv
# Windows PowerShell：.venv\Scripts\Activate.ps1
# Linux / macOS：source .venv/bin/activate
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

也可使用 `conda env create -f environment.yml`。上面使用 CPU 基线；需要 CUDA 时，根据本机环境安装对应 PyTorch 发行包并记录版本。GUI 额外安装 `requirements-gui.txt`；C++ 构建额外安装 `requirements-build.txt` 和 C++17 编译器。

## 快速验证

在仓库根目录执行：

```sh
python -B -m unittest discover -s tests -v
python core/test.py
```

下面用小棋盘和低模拟次数运行一局 Python 后端自博弈，验证入口可运行，不代表训练收敛：

```sh
python gat_cpp/cpp_train.py --selfplay-backend python --eval-backend python --fresh-start --board-size 5 --n-in-row 4 --game-batch-num 1 --n-playout 8 --disable-selfplay-async --torch-cpu-threads 1
```

完整参数见 `python gat_cpp/cpp_train.py --help`。默认 C++ 后端必须先构建；选择 Python 后端可以不安装 C++ 扩展。

## C++ 后端

```sh
python -m pip install -r requirements-build.txt
cmake -S gat_cpp -B gat_cpp/build -DCMAKE_BUILD_TYPE=Release
cmake --build gat_cpp/build --config Release
python gat_cpp/tools/run_parallel_smoke.py --playout 32 --threads 2
```

多 Python 环境下应向 CMake 显式指定当前环境的 `Python3_EXECUTABLE`，详见 [C++ 说明](gat_cpp/README.md)。自动检查在 Linux 上构建扩展并执行小规模搜索；长时间多线程训练和硬件性能需要另行测量。

## 模型保存与恢复

检查点 v2 保存网络参数、BatchNorm 运行统计量、网络规格、棋盘规则、优化器与训练状态。恢复前检查完整参数和优化器张量，并核对回放数据的尺寸，避免不同棋盘的权重被部分混合加载。

新模型按 `models/<棋盘尺寸>_n<连珠数>/` 保存。原有模型文件保持原位；旧格式缺少 BatchNorm 统计量，无法可靠恢复完整预测状态，加载时会明确拒绝。保留旧实验前先备份，再安排重新训练或经过验证的数据校准迁移。

已有 `.model/.model2` 独立 legacy 推理路径保留；手写网络的新参数字典使用 v2 格式。只加载可信来源的检查点，模型读取包含 Python 序列化对象。

配置快照在命令行覆盖与线程调优后更新。训练输出、模型、运行日志与本地配置已加入忽略规则，不提交个人路径、私有数据或服务凭据。

## 验证范围

已通过检查点 / 梯度回归、底层网络前后向检查、小棋盘 Python 自博弈，以及 Linux 上的 C++ 构建和搜索检查。它们用于验证工程链路，不代表棋力或训练收敛结论。

性能对比需同时记录棋盘尺寸、模拟次数、线程数、硬件和随机种子；当前仓库尚未提供可复核旧胜率描述的完整模型与评估数据。

## 授权

项目原创部分尚未提供统一开源许可证，来源与授权边界见 [AUTHORIZATION.md](AUTHORIZATION.md)。第三方代码和依赖仍适用原有许可证。

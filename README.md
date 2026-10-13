# YNU_AG-CNN：AlphaZero 五子棋实验

结合手写 CNN 前向 / 反向传播与蒙特卡洛树搜索的五子棋学习项目，包含 Python 训练流程和可选 C++ 搜索后端。原始思路与部分实现基于 [AlphaZero_Gomoku](https://github.com/junxiaosong/AlphaZero_Gomoku)，来源与授权边界见 [AUTHORIZATION.md](AUTHORIZATION.md)。

项目用于学习和实验，训练效果取决于棋盘、模拟次数、随机种子、硬件和训练时长。当前仓库没有可据以复核旧胜率描述的完整模型与评估数据，因此不作性能或逐位复现保证。

## 目录

| 目录 | 用途 |
| --- | --- |
| `core/` | 棋盘、搜索与底层算子 |
| `pipeline/` | 手写策略价值网络、优化器、检查点验证 |
| `gat/` | Python 训练和人机对弈入口 |
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

## 授权

本次仅补充授权与来源说明，暂不为项目原创部分新增开源许可证。第三方代码和依赖仍适用原有许可证，不以本说明替换。

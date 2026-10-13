# gat_cpp

C++ scaffold for gomoku engine acceleration.

## Current scope

- C++ `Board` implementation (move/update/winner check).
- C++ `AlphaZeroPlayer` shared-tree parallel PUCT search (virtual loss + persistent worker pool).
- Python bindings via `pybind11` module: `mcts_cpp`.

## Build (Windows, from repo root)

1. Install build deps in active Python env:

```powershell
python -m pip install -r requirements-build.txt
```

2. Configure and build:

```powershell
cmake -S gat_cpp -B gat_cpp/build -DPython3_EXECUTABLE=(python -c "import sys; print(sys.executable)")
cmake --build gat_cpp/build --config Release
```

3. Locate built module under the CMake output directory and import from Python:

```python
import mcts_cpp
b = mcts_cpp.Board(8, 8, 5)
b.init_board(0)
ai = mcts_cpp.AlphaZeroPlayer()
print(ai.get_action(b))
```

## End-to-end parallel flow (current)

1. Build module:

```powershell
cmake -S gat_cpp -B gat_cpp/build -DPython3_EXECUTABLE=(python -c "import sys; print(sys.executable)")
cmake --build gat_cpp/build --config Release
```

2. Run smoke test (parallel rollout path):

```powershell
python gat_cpp/tools/run_parallel_smoke.py --playout 1600 --threads 8
```

3. Validate auto thread mode:

```powershell
python gat_cpp/tools/run_parallel_smoke.py --playout 1600 --threads 0
```

4. Benchmark repeated get_action calls (persistent worker pool benefit):

```powershell
python gat_cpp/tools/run_parallel_session_bench.py --playout 6400 --threads 8 --steps 20
```

Notes:
- `AlphaZeroPlayer` now runs shared-tree parallel PUCT with virtual loss and persistent workers.
- Worker threads are persistent across calls to reduce per-move thread creation overhead.
- Measure throughput and playing strength separately for the selected hardware and search settings.

## Training and checkpoints

The training entry is `python gat_cpp/cpp_train.py --help`. Neural evaluation callbacks and tree reuse are implemented in the current source; benchmark them on your own workload. See the root README for a small Python-only smoke run, checkpoint v2 requirements, and authorization notes.

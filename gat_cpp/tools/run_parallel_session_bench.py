import argparse
import importlib
import os
import shutil
import sys
import time


def _setup_windows_dll_dirs(extra_dirs):
    if os.name != "nt" or not hasattr(os, "add_dll_directory"):
        return

    runtime_candidates = [
        os.path.dirname(shutil.which("c++.exe") or ""),
        os.path.dirname(shutil.which("g++.exe") or ""),
        "C:/msys64/ucrt64/bin",
        "C:/msys64/mingw64/bin",
    ]

    dll_dirs = []
    for p in list(extra_dirs) + runtime_candidates:
        if p and os.path.isdir(p) and p not in dll_dirs:
            dll_dirs.append(p)

    for p in dll_dirs:
        os.add_dll_directory(p)


def _setup_paths() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    build_release = os.path.join(repo_root, "gat_cpp", "build", "Release")
    build_root = os.path.join(repo_root, "gat_cpp", "build")
    _setup_windows_dll_dirs([build_release, build_root])
    for p in [build_release, build_root]:
        if p not in sys.path:
            sys.path.insert(0, p)



def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark repeated get_action calls")
    parser.add_argument("--size", type=int, default=8)
    parser.add_argument("--n-in-row", type=int, default=5)
    parser.add_argument("--playout", type=int, default=6400)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    _setup_paths()
    mcts_cpp = importlib.import_module("mcts_cpp")

    b = mcts_cpp.Board(args.size, args.size, args.n_in_row)
    b.init_board(0)

    ai = mcts_cpp.AlphaZeroPlayer(n_playout=args.playout, num_threads=args.threads)
    ai.set_player_ind(1)

    elapsed = []
    for _ in range(args.steps):
        if len(b.availables) == 0:
            break
        t0 = time.perf_counter()
        mv = ai.get_action(b)
        t1 = time.perf_counter()
        b.do_move(mv)
        ai.update_with_move(mv)
        elapsed.append((t1 - t0) * 1000.0)

    if not elapsed:
        print("no steps executed")
        return 1

    avg_ms = sum(elapsed) / len(elapsed)
    print("=== Parallel Session Bench ===")
    print(f"steps={len(elapsed)}, playout={args.playout}, threads={args.threads}")
    print(f"avg_ms_per_get_action={avg_ms:.2f}")
    print(f"effective_playout_per_sec={args.playout / (avg_ms / 1000.0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

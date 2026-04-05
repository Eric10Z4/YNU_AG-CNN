import argparse
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Run parallel smoke test for mcts_cpp")
    parser.add_argument("--size", type=int, default=8, help="board size")
    parser.add_argument("--n-in-row", type=int, default=5, help="n in row")
    parser.add_argument("--playout", type=int, default=1200, help="playout count")
    parser.add_argument("--threads", type=int, default=0, help="thread count, 0=auto")
    parser.add_argument("--repeat", type=int, default=1, help="repeat runs for average timing")
    args = parser.parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    build_release = os.path.join(repo_root, "gat_cpp", "build", "Release")
    build_root = os.path.join(repo_root, "gat_cpp", "build")
    _setup_windows_dll_dirs([build_release, build_root])
    for p in [build_release, build_root]:
        if p not in sys.path:
            sys.path.insert(0, p)

    import mcts_cpp  # noqa: WPS433

    b = mcts_cpp.Board(args.size, args.size, args.n_in_row)
    b.init_board(0)

    ai = mcts_cpp.AlphaZeroPlayer(n_playout=args.playout, num_threads=args.threads)
    ai.set_player_ind(1)

    elapsed_ms_list = []
    move = -1
    for _ in range(max(1, args.repeat)):
        start = time.perf_counter()
        move = ai.get_action(b)
        elapsed_ms_list.append((time.perf_counter() - start) * 1000.0)

    elapsed_ms = elapsed_ms_list[-1]
    avg_elapsed_ms = sum(elapsed_ms_list) / len(elapsed_ms_list)

    print("=== Parallel Smoke ===")
    print(f"board={args.size}x{args.size}, n_in_row={args.n_in_row}")
    print(f"playout={ai.n_playout}, threads={ai.num_threads} (0 means auto)")
    print(f"repeat={len(elapsed_ms_list)}")
    print(f"chosen_move={move}")
    print(f"elapsed_ms={elapsed_ms:.2f}")
    print(f"avg_elapsed_ms={avg_elapsed_ms:.2f}")
    print(f"playout_per_sec={ai.n_playout / (avg_elapsed_ms / 1000.0):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

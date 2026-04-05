"""将 legacy .model/.model2 权重转换为 UI 可识别的 .pth 文件。

支持输入:
- Python2 pickle 的 list[np.ndarray]（长度 16）

输出:
- torch checkpoint: {"legacy_params": [...], "format": ..., "board_size": 8}
"""

import argparse
import os
import pickle
from typing import List

import numpy as np
import torch


def load_legacy_params(src_path: str) -> List[np.ndarray]:
    with open(src_path, "rb") as f:
        obj = pickle.load(f, encoding="latin1")
    if not isinstance(obj, (list, tuple)):
        raise ValueError(f"{src_path} 不是 list/tuple，无法识别为 legacy 模型")
    if len(obj) != 16:
        raise ValueError(f"{src_path} 参数个数异常，期望 16，实际 {len(obj)}")
    arrays = [np.asarray(x, dtype=np.float32) for x in obj]
    return arrays


def convert_one(src_path: str, dst_dir: str, board_size: int = 8) -> str:
    params = load_legacy_params(src_path)
    os.makedirs(dst_dir, exist_ok=True)

    stem = os.path.splitext(os.path.basename(src_path))[0]
    src_ext = os.path.splitext(src_path)[1].lstrip(".").lower()
    if src_ext:
        dst_name = f"{stem}_{src_ext}.pth"
    else:
        dst_name = f"{stem}.pth"
    dst_path = os.path.join(dst_dir, dst_name)

    checkpoint = {
        "format": "legacy_alpha_zero_pickle_v1",
        "board_size": board_size,
        "source_path": os.path.abspath(src_path),
        "legacy_params": [torch.tensor(p, dtype=torch.float32) for p in params],
    }
    torch.save(checkpoint, dst_path)
    return dst_path


def parse_args():
    parser = argparse.ArgumentParser(description="convert legacy .model/.model2 to .pth")
    parser.add_argument("inputs", nargs="+", help="输入模型路径，可传多个")
    parser.add_argument(
        "--out-dir",
        default=os.path.join("models", "gui_models"),
        help="输出目录，默认 models/gui_models",
    )
    parser.add_argument("--board-size", type=int, default=8, help="棋盘尺寸，默认 8")
    return parser.parse_args()


def main():
    args = parse_args()
    success = 0

    for src in args.inputs:
        src_abs = os.path.abspath(src)
        if not os.path.exists(src_abs):
            print(f"[SKIP] 文件不存在: {src_abs}")
            continue
        try:
            out = convert_one(src_abs, args.out_dir, board_size=args.board_size)
            print(f"[OK] {src_abs} -> {os.path.abspath(out)}")
            success += 1
        except Exception as e:
            print(f"[FAIL] {src_abs}: {e}")

    if success == 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""edgebuild.py — libnilm_edge 跨平台构建与加载（无 make 依赖，Windows 友好）。

策略：
  1. 仓库附带预编译库（Windows=edge/nilm_edge.dll x86_64；Linux=edge/libnilm_edge.so），
     存在且不比源码旧 → 直接使用，**无需本机编译器**；
  2. 否则自动寻找编译器重建（gcc / clang / cc / cl(MSVC) / zig cc）；
  3. 都没有 → 明确报错并给出安装指引（conda m2w64-gcc / pip ziglang / VS Build Tools / WSL）。

手动重建：python edge/edgebuild.py [--force]
测试与验证 harness（tests/test_edge_parity.py::build_lib、scripts/edge_stream_test.py）
统一走本模块——在 Windows 上不再依赖 make。
"""
import argparse
import platform
import shutil
import subprocess
import sys
from pathlib import Path

EDGE_DIR = Path(__file__).resolve().parent
SRC = EDGE_DIR / "nilm_edge.c"
HDR = EDGE_DIR / "nilm_edge.h"
IS_WIN = platform.system() == "Windows"
LIB = EDGE_DIR / ("nilm_edge.dll" if IS_WIN else "libnilm_edge.so")

_INSTALL_HINT = """解决任选其一：
  1) git pull 获取仓库预编译库（Windows: edge/nilm_edge.dll / Linux: edge/libnilm_edge.so）；
  2) pip install ziglang   （zig cc 交叉/本机编译，~50MB，无需系统工具链）；
  3) conda install -c msys2 m2w64-gcc   （Anaconda 环境，提供 x86_64-w64-mingw32-gcc）；
  4) 安装 VS Build Tools（cl）或 mingw-w64 或 WSL 后重试。"""


def _stale():
    if not LIB.exists():
        return True
    newest_src = max(SRC.stat().st_mtime, HDR.stat().st_mtime)
    return LIB.stat().st_mtime < newest_src


def _find_compiler():
    for cc in ("gcc", "clang", "cc", "cl", "zig"):
        p = shutil.which(cc)
        if p:
            return cc, p
    return None, None


def build(force=False):
    """确保库存在且新鲜；返回库路径。无编译器且库缺失/过期时抛 SystemExit（带指引）。"""
    if not force and not _stale():
        return LIB
    cc, path = _find_compiler()
    if cc is None:
        raise SystemExit(f"找不到可用的 {LIB.name}（不存在或过期），且本机无 C 编译器。\n{_INSTALL_HINT}")
    if cc == "cl":
        cmd = [path, "/O2", "/LD", "/DNILM_EDGE_BUILDING", SRC.name, f"/Fe:{LIB.name}"]
    elif cc == "zig":
        cmd = [path, "cc", "-O2", "-std=c99", "-fPIC", "-shared",
               "-DNILM_EDGE_BUILDING", SRC.name, "-o", LIB.name]
        if IS_WIN:
            cmd[2:2] = ["-target", "x86_64-windows-gnu"]
    else:  # gcc / clang / cc
        cmd = [path, "-O2", "-std=c99", "-fPIC", "-shared", "-Wall",
               "-DNILM_EDGE_BUILDING", SRC.name, "-o", LIB.name]
    if cc != "cl" and not IS_WIN:
        cmd.append("-lm")
    print(f"[edgebuild] {'重建' if LIB.exists() else '构建'} {LIB.name}（{cc}）…")
    r = subprocess.run(cmd, cwd=EDGE_DIR, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"编译失败（{' '.join(cmd)}）：\n{r.stdout}\n{r.stderr}")
    if not LIB.exists():
        raise SystemExit(f"编译 seemingly 成功但未产出 {LIB}")
    return LIB


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="源码未变也强制重建")
    a = ap.parse_args()
    print(f"库路径：{build(a.force)}")

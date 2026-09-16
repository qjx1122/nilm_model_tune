"""edgebuild.py — libnilm_edge 跨平台构建与加载（无 make 依赖，Windows 友好）。

策略：
  1. 仓库附带预编译库（Windows=edge/nilm_edge.dll x86_64；Linux=edge/libnilm_edge.so）。
     新鲜度用 **内容哈希构建戳**（edge/.build_stamp = sha256(nilm_edge.c + nilm_edge.h)）
     判定——不用 mtime：git checkout 不保留修改时间且写入顺序不定，mtime 判定在用户机器上
     会把预编译库误判过期（2026-09-16 用户实跑教训）。戳匹配 → 直接加载，无需本机编译器；
  2. 源码 sha 与戳不符（源码被改过）或库缺失 → 自动寻找编译器重建（gcc/clang/cc/cl/zig cc）；
  3. 都没有 → 明确报错并给出安装指引（git pull / conda m2w64-gcc / pip ziglang / VS Build Tools）。

手动重建：python edge/edgebuild.py [--force]
测试与验证 harness（tests/test_edge_parity.py::build_lib、scripts/edge_stream_test.py）
统一走本模块——在 Windows 上不依赖 make，也不依赖任何编译器。
"""
import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

EDGE_DIR = Path(__file__).resolve().parent
SRC = EDGE_DIR / "nilm_edge.c"
HDR = EDGE_DIR / "nilm_edge.h"
STAMP = EDGE_DIR / ".build_stamp"
IS_WIN = platform.system() == "Windows"
LIB = EDGE_DIR / ("nilm_edge.dll" if IS_WIN else "libnilm_edge.so")

_INSTALL_HINT = """解决任选其一：
  1) git pull 获取仓库预编译库（Windows: edge/nilm_edge.dll / Linux: edge/libnilm_edge.so）；
  2) pip install ziglang   （zig cc 交叉/本机编译，~50MB，无需系统工具链）；
  3) conda install -c msys2 m2w64-gcc   （Anaconda 环境，提供 x86_64-w64-mingw32-gcc）；
  4) 安装 VS Build Tools（cl）或 mingw-w64 或 WSL 后重试。"""


def _source_sha():
    h = hashlib.sha256()
    for f in (SRC, HDR):
        h.update(f.read_bytes())
    return h.hexdigest()


def _find_compiler():
    for cc in ("gcc", "clang", "cc", "cl", "zig"):
        p = shutil.which(cc)
        if p:
            return cc, p
    return None, None


def build(force=False):
    """确保库可用；返回库路径。

    判定顺序：--force → 重建；库存在且（构建戳匹配 或 无戳）→ 直接用；
    库存在但戳不匹配（源码改过）/ 库缺失 → 找编译器重建；无编译器 → SystemExit（带指引）。
    """
    if LIB.exists():
        if not force:
            if STAMP.exists():
                if STAMP.read_text(encoding="utf-8").strip() == _source_sha():
                    return LIB          # 预编译库与源码内容一致：直接使用
            else:
                print("[edgebuild] 预编译库存在但无构建戳：直接使用（源码未随仓库变更则安全；"
                      "若改过 C 源码请用 --force 重建）")
                return LIB
    cc, path = _find_compiler()
    if cc is None:
        why = "库不存在" if not LIB.exists() else "源码与构建戳不匹配（源码改过？）"
        raise SystemExit(f"找不到可用的 {LIB.name}（{why}），且本机无 C 编译器。\n{_INSTALL_HINT}")
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
    STAMP.write_text(_source_sha() + "\n", encoding="utf-8")
    return LIB


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="源码未变也强制重建")
    a = ap.parse_args()
    print(f"库路径：{build(a.force)}")

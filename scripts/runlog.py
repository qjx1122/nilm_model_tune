"""控制台输出留痕：stdout/stderr 同步写入日志文件（REPORT_TEST.md 实录 45）。

动机：用户侧控制台缓冲被冲掉，24 份 train stdout 无法回传（实录 44 之后）——
evaluate JSON 是权威，但 stdout（best_ep 分布、warning、早停曲线）同样有档案价值。
此后所有用户侧脚本默认留痕，控制台行为不变，仅开头多一行「日志留痕: <路径>」。

接入（各脚本两行）：
    from runlog import setup_run_log, default_log_path
    setup_run_log(Path(args.out) / "train.log", enabled=not args.no_log)

行为：
- 双写：控制台照常显示 + 日志文件（UTF-8，覆盖写）；
- stderr 一并捕获（torch UserWarning、traceback 同入文件）；
- 文件头：时间戳 / 命令行 / cwd；文件尾（atexit）：结束时间 / 时长——
  脚本崩溃时 traceback 也在文件里（footer 前最后内容）；
- 幂等：同进程第二次调用不再生效；enabled=False（--no-log）完全关闭。
"""
import atexit
import io
import sys
import time
from datetime import datetime
from pathlib import Path

_installed = False


class _Tee:
    """按序写入多个流（控制台 + 文件）。"""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for s in self._streams:
            s.write(data)

    def flush(self):
        for s in self._streams:
            try:
                s.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def fileno(self):
        # 个别库（tqdm 等）需要真实 fd：委托给原始终端流
        #（fd 层直写不进文件，但本项目脚本全部走 print，无此路径）。
        for s in self._streams:
            try:
                return s.fileno()
            except Exception:
                continue
        raise io.UnsupportedOperation("fileno")

    def __getattr__(self, name):
        # encoding 等属性兜底委托给第一个流
        return getattr(self._streams[0], name)


def default_log_path(stem):
    """无 out 目录脚本的日志命名：logs/<stem>_<yyyymmdd_HHMMSS>.log（logs/ 已 gitignore）。"""
    return Path("logs") / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.log"


def setup_run_log(log_path, *, enabled=True):
    """安装 stdout/stderr 留痕；返回日志 Path；禁用时返回 None 且不产生任何文件。"""
    global _installed
    if _installed or not enabled:
        return None
    _installed = True
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(log_path, "w", encoding="utf-8", buffering=1)  # 行缓冲：中途崩溃已写内容仍在
    start = time.time()
    orig_out, orig_err = sys.stdout, sys.stderr
    sys.stdout = _Tee(orig_out, f)
    sys.stderr = _Tee(orig_err, f)
    sys.stdout.write(
        f"===== 日志留痕 {datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
        f"cmd: {' '.join(sys.argv)}\n"
        f"cwd: {Path.cwd()}\n"
    )

    def _finish():
        dur = time.time() - start
        try:
            sys.stdout.write(
                f"===== 结束 {datetime.now():%Y-%m-%d %H:%M:%S}（时长 {dur:.1f}s）=====\n")
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            f.flush()
            f.close()
            sys.stdout, sys.stderr = orig_out, orig_err

    atexit.register(_finish)
    print(f"日志留痕: {log_path}")
    return log_path

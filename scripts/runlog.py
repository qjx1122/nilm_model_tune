"""控制台输出留痕：stdout/stderr 同步写入日志文件（REPORT_TEST.md 实录 45-46）。

两级留痕：
1. 运行日志（每次运行独立文件，覆盖写）：
   - train/evaluate/tune → 各自输出目录（train.log / evaluate.log / tune.log）
   - prepare_ukdale → <out>.log（与 npz/data_spec.json 同目录）
   - 其余脚本 → logs/<名称>_<时间戳>.log
2. 总日志（所有运行按序追加到同一文件，默认 logs/console_all.log）——
   foreach 批次跑完后一个文件装下全部控制台输出，直接整份回传。
   路径/开关用环境变量 NILM_CONSOLE_LOG 控制：
     $env:NILM_CONSOLE_LOG = "D:\\logs\\session.log"   # PowerShell 指定路径
     $env:NILM_CONSOLE_LOG = "off"                     # 关闭总日志（运行日志不受影响）
   （取消：Remove-Item Env:\\NILM_CONSOLE_LOG）

行为：
- 多路写：控制台照常显示 + 运行日志 + 总日志（UTF-8；行缓冲，中途崩溃已写内容仍在）；
- stderr 一并捕获（torch UserWarning、traceback 同入文件）；
- 文件头：时间戳 / 命令行 / cwd；文件尾（atexit）：结束时间 / 时长——
  脚本崩溃时 traceback 也在文件里（footer 前最后内容）；
- 总日志在相邻两次运行之间留空行分隔；每次运行的头部自带 cmd/cwd 上下文；
  并行进程会交错（foreach 串行批次无此问题）；
- 幂等：同进程第二次调用不再生效；enabled=False（--no-log）两级留痕全关。
"""
import atexit
import io
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_installed = False

_TOTAL_OFF_VALUES = {"", "0", "off", "false", "no"}


class _Tee:
    """按序写入多个流（控制台 + 运行日志 + 总日志）。"""

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
    """无 out 目录脚本的运行日志命名：logs/<stem>_<yyyymmdd_HHMMSS>.log（logs/ 已 gitignore）。"""
    return Path("logs") / f"{stem}_{datetime.now():%Y%m%d_%H%M%S}.log"


def total_log_path():
    """总日志路径：环境变量 NILM_CONSOLE_LOG 优先（off/0/空=关闭），默认 logs/console_all.log。"""
    env = os.environ.get("NILM_CONSOLE_LOG")
    if env is not None:
        env = env.strip()
        if env.lower() in _TOTAL_OFF_VALUES:
            return None
        return Path(env)
    return Path("logs") / "console_all.log"


def setup_run_log(log_path, *, enabled=True, total_log=None):
    """安装 stdout/stderr 两级留痕；返回运行日志 Path；禁用时返回 None 且不产生任何文件。

    total_log：总日志路径——None=按 total_log_path() 解析（环境变量 / 默认）；
    False=本次强制不用总日志；传路径=显式指定（测试用）。
    """
    global _installed
    if _installed or not enabled:
        return None
    _installed = True
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f_run = open(log_path, "w", encoding="utf-8", buffering=1)  # 行缓冲：中途崩溃已写内容仍在

    if total_log is False:
        tl = None
    elif total_log is None:
        tl = total_log_path()
    else:
        tl = Path(total_log)
    f_total = None
    if tl is not None:
        tl.parent.mkdir(parents=True, exist_ok=True)
        f_total = open(tl, "a", encoding="utf-8", buffering=1)  # 总日志=追加模式
        if tl.stat().st_size > 0:
            f_total.write("\n")  # 相邻运行之间空行分隔

    start = time.time()
    orig_out, orig_err = sys.stdout, sys.stderr
    extra = [f_total] if f_total is not None else []
    sys.stdout = _Tee(orig_out, f_run, *extra)
    sys.stderr = _Tee(orig_err, f_run, *extra)
    sys.stdout.write(
        f"===== 日志留痕 {datetime.now():%Y-%m-%d %H:%M:%S} =====\n"
        f"cmd: {' '.join(sys.argv)}\n"
        f"cwd: {Path.cwd()}\n"
    )
    if f_total is not None:
        print(f"总日志: {tl}")

    def _finish():
        dur = time.time() - start
        try:
            sys.stdout.write(
                f"===== 结束 {datetime.now():%Y-%m-%d %H:%M:%S}（时长 {dur:.1f}s）=====\n")
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            f_run.flush()
            f_run.close()
            if f_total is not None:
                f_total.flush()
                f_total.close()
            sys.stdout, sys.stderr = orig_out, orig_err

    atexit.register(_finish)
    print(f"日志留痕: {log_path}")
    return log_path

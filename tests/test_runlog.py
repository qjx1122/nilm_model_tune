"""runlog 控制台留痕测试（REPORT_TEST.md 实录 45-46）。

纯 stdlib（不依赖 numpy/torch）；核心用例在子进程中运行以隔离
「setup 后本进程 stdout 被替换」的副作用。pytest 或直接 python 均可跑：
    python tests/test_runlog.py
"""
import os
import subprocess
import sys
import tempfile
from io import StringIO
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from runlog import _Tee, default_log_path, setup_run_log, total_log_path  # noqa: E402


def test_tee_writes_both_streams():
    a, b = StringIO(), StringIO()
    t = _Tee(a, b)
    t.write("hello\n")
    t.flush()
    assert a.getvalue() == "hello\n"
    assert b.getvalue() == "hello\n"


def _run_py(code, cwd, env=None):
    return subprocess.run([sys.executable, "-c", code], cwd=cwd, env=env,
                          capture_output=True, text=True, check=True)


def test_setup_captures_print_stderr_and_footer():
    """端到端：print/stderr/头部（cmd、cwd）/尾部（结束+时长）落运行日志与默认总日志，控制台同步可见。"""
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "sub" / "run.log"  # 顺带验证父目录自动创建
        code = (
            "import sys\n"
            f"sys.path.insert(0, r'{SCRIPTS}')\n"
            "from runlog import setup_run_log\n"
            f"p = setup_run_log(r'{log}')\n"
            "assert p is not None\n"
            "print('MARKER_PRINT')\n"
            "sys.stderr.write('MARKER_STDERR\\n')\n"
        )
        env = {**os.environ, "NILM_CONSOLE_LOG": str(Path(td) / "total.log")}
        r = _run_py(code, cwd=td, env=env)
        assert "MARKER_PRINT" in r.stdout, "控制台多路写失效"
        text = log.read_text(encoding="utf-8")
        assert "MARKER_PRINT" in text
        assert "MARKER_STDERR" in text, "stderr 未捕获"
        assert "cmd: " in text and "cwd: " in text
        assert "===== 结束" in text and "时长" in text
        assert "日志留痕: " in text
        total = (Path(td) / "total.log").read_text(encoding="utf-8")
        assert "MARKER_PRINT" in total and "MARKER_STDERR" in total, "总日志未捕获"


def test_disabled_creates_nothing():
    with tempfile.TemporaryDirectory() as td:
        code = (
            "import sys\n"
            f"sys.path.insert(0, r'{SCRIPTS}')\n"
            "from runlog import setup_run_log\n"
            "assert setup_run_log(r'x.log', enabled=False) is None\n"
            "print('OK')\n"
        )
        r = _run_py(code, cwd=td)
        assert "日志留痕" not in r.stdout
        assert not (Path(td) / "x.log").exists()


def test_double_install_is_idempotent():
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "once.log"
        code = (
            "import sys\n"
            f"sys.path.insert(0, r'{SCRIPTS}')\n"
            "from runlog import setup_run_log\n"
            f"p1 = setup_run_log(r'{log}')\n"
            f"p2 = setup_run_log(r'{Path(td) / 'twice.log'}')\n"
            "assert p1 is not None and p2 is None\n"
        )
        _run_py(code, cwd=td)
        assert log.exists() and not (Path(td) / "twice.log").exists()


def test_default_log_path_shape():
    p = default_log_path("diagnose_kettle")
    assert p.parent == Path("logs")
    assert p.name.startswith("diagnose_kettle_2")


def test_total_log_path_resolution():
    """env NILM_CONSOLE_LOG：未设=默认 logs/console_all.log；off=关闭；路径=覆盖。"""
    code = (
        "import sys, os\n"
        "from pathlib import Path\n"
        f"sys.path.insert(0, r'{SCRIPTS}')\n"
        "os.environ.pop('NILM_CONSOLE_LOG', None)\n"
        "from runlog import total_log_path\n"
        "assert total_log_path() == Path('logs') / 'console_all.log'\n"
        "os.environ['NILM_CONSOLE_LOG'] = 'off'\n"
        "assert total_log_path() is None\n"
        "os.environ['NILM_CONSOLE_LOG'] = r'D:\\logs\\session.log'\n"
        "assert total_log_path() == Path(r'D:\\logs\\session.log')\n"
        "print('OK')\n"
    )
    _run_py(code, cwd=tempfile.gettempdir())


def test_total_log_appends_across_runs():
    """总日志跨运行按序追加 + 空行分隔；运行日志覆盖写只含本次。"""
    with tempfile.TemporaryDirectory() as td:
        total = Path(td) / "session.log"
        code_tmpl = (
            "import sys\n"
            f"sys.path.insert(0, r'{SCRIPTS}')\n"
            "from runlog import setup_run_log\n"
            f"p = setup_run_log(r'{Path(td) / 'run.log'}', total_log=r'{total}')\n"
            "assert p is not None\n"
            "print('MARKER_%s')\n"
        )
        _run_py(code_tmpl % "ONE", cwd=td)
        r2 = _run_py(code_tmpl % "TWO", cwd=td)
        text = total.read_text(encoding="utf-8")
        assert "MARKER_ONE" in text and "MARKER_TWO" in text
        assert text.index("MARKER_ONE") < text.index("MARKER_TWO"), "总日志须按序追加"
        assert "\n\n" in text, "相邻运行之间应有空行分隔"
        run_text = (Path(td) / "run.log").read_text(encoding="utf-8")
        assert "MARKER_ONE" not in run_text and "MARKER_TWO" in run_text, "运行日志覆盖写"
        assert "总日志: " in r2.stdout, "控制台应提示总日志路径"


def test_total_log_env_off_keeps_run_log():
    """NILM_CONSOLE_LOG=off：总日志关闭，运行日志不受影响。"""
    with tempfile.TemporaryDirectory() as td:
        code = (
            "import sys\n"
            f"sys.path.insert(0, r'{SCRIPTS}')\n"
            "from runlog import setup_run_log\n"
            f"p = setup_run_log(r'{Path(td) / 'run.log'}')\n"
            "assert p is not None\n"
            "print('MARKER')\n"
        )
        env = {**os.environ, "NILM_CONSOLE_LOG": "off"}
        r = _run_py(code, cwd=td, env=env)
        assert (Path(td) / "run.log").exists(), "off 只关总日志"
        assert not (Path(td) / "logs").exists(), "总日志不应创建"
        assert "总日志" not in r.stdout
        assert "MARKER" in (Path(td) / "run.log").read_text(encoding="utf-8")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}: {e}")
    sys.exit(1 if failed else 0)

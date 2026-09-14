import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True)
p.add_argument("--no-log", action="store_true",
               help="禁用控制台输出留痕（默认写 <run-dir>/evaluate.log）")
args = p.parse_args()

# 控制台输出留痕（实录 45）：输出 JSON 与控制台同步落盘
from runlog import setup_run_log
setup_run_log(Path(args.run_dir) / "evaluate.log", enabled=not args.no_log)

run = Path(args.run_dir)
result = json.loads((run / "result.json").read_text(encoding="utf-8"))

# best_epoch 的完整 val 指标（fit() 逐 epoch 落盘在 history.json，实录 14：
# result.json 只含 test 指标，val KPI 需从此处补读——不重训、不碰 Test）。
best_epoch = result.get("best_epoch")
hist_path = run / "history.json"
if best_epoch and hist_path.exists():
    hist = json.loads(hist_path.read_text(encoding="utf-8"))
    row = next((r for r in hist if r.get("epoch") == best_epoch), None)
    if row:
        result["best_epoch_val"] = {k: v for k, v in row.items() if k.startswith("val_")}

print(json.dumps(result, indent=2, ensure_ascii=False))

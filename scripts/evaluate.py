import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True)
args = p.parse_args()
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

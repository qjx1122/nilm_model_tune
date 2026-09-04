import argparse, json
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument("--run-dir", required=True)
args = p.parse_args()
path = Path(args.run_dir) / "result.json"
if not path.exists():
    raise FileNotFoundError(path)
print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TARGET = ROOT / "illustrative_simulation"


def run(script, *args):
    subprocess.run([sys.executable, str(ROOT / script), *map(str, args)], check=True, cwd=ROOT)


run("make_illustrative_predictions.py")
run("make_results.py", "--predictions", TARGET / "results" / "oof_predictions.csv",
    "--results-dir", TARGET / "results", "--figures-dir", TARGET / "figures", "--illustrative")
run("build_illustrative_report.py")

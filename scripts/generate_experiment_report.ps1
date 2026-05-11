param(
  [switch]$RunBaselines,
  [switch]$RunMainModel,
  [switch]$RunAblation
)

$python = "C:\Users\cathy\anaconda3\envs\muti\python.exe"

if ($RunBaselines) {
  & $python "baseline_experiments.py"
}

if ($RunMainModel) {
  & $python "train_recommender.py"
}

if ($RunAblation) {
  & $python "ablation_experiments.py"
}

& $python "compare_experiments.py"

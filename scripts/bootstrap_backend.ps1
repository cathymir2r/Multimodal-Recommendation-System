param(
  [int]$SyncLimit = 1000,
  [string]$WarmUserId = ""
)

$python = "C:\Users\cathy\anaconda3\envs\muti\python.exe"

& $python "backend\manage.py" migrate
& $python "backend\manage.py" sync_catalog --limit $SyncLimit --clear

if ($WarmUserId -and $WarmUserId.Trim()) {
  & $python "backend\manage.py" warm_recommendations --user-id $WarmUserId --top-k 10
} else {
  & $python "backend\manage.py" warm_recommendations
}

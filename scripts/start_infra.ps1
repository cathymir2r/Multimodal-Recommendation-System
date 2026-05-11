param(
  [switch]$Detach
)

$composeFile = Join-Path $PSScriptRoot "..\deploy\docker-compose.yml"
if ($Detach) {
  docker compose -f $composeFile up -d
} else {
  docker compose -f $composeFile up
}

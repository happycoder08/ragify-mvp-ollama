param(
    [switch]$Quick = $false,
    [switch]$Debug = $false
)

$args = @()
if ($Quick) { $args += "--quick" }
if ($Debug) { $args += "--debug" }

Write-Host "Running Evaluation Harness..."
python -m scripts.eval_run @args

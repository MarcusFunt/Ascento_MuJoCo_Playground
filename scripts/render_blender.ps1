[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string] $Capture,

  [Parameter(Mandatory = $true)]
  [string] $Description,

  [Parameter(Mandatory = $true)]
  [string] $Name,

  [string] $Blender = "blender",
  [string] $WslRepository = "",
  [string] $Checkpoint = "",
  [string[]] $CameraShots = @("low_front", "side_follow", "orbit"),
  [int] $Resolution = 1280,
  [switch] $CinematicDof,
  [switch] $RenderPngFrames
)

$ErrorActionPreference = "Stop"

if ($Name -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$') {
  throw "Name must be 1–64 characters and contain only letters, numbers, dot, underscore, or hyphen."
}
if ($Resolution -lt 16) {
  throw "Resolution must be at least 16 pixels."
}

if ([string]::IsNullOrWhiteSpace($WslRepository)) {
  if (-not [string]::IsNullOrWhiteSpace($env:ASCENTO_WSL_REPOSITORY)) {
    $WslRepository = $env:ASCENTO_WSL_REPOSITORY
  } else {
    $WslRepository = "\\wsl.localhost\Ubuntu\root\Ascento_MuJoCo_Playground"
  }
}

$repoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$importer = Join-Path $repoRoot "tools\blender\import_motion.py"
if (-not (Test-Path -LiteralPath $importer -PathType Leaf)) {
  throw "Blender importer not found in this checkout: $importer"
}
$capturePath = (Resolve-Path -LiteralPath $Capture).ProviderPath
$descriptionPath = (Resolve-Path -LiteralPath $Description).ProviderPath
$checkpointPath = $null
if (-not [string]::IsNullOrWhiteSpace($Checkpoint)) {
  $checkpointPath = (Resolve-Path -LiteralPath $Checkpoint).ProviderPath
}
$blenderCommand = Get-Command -Name $Blender -CommandType Application -ErrorAction Stop

$wslCaptures = Join-Path $WslRepository "captures"
if (-not (Test-Path -LiteralPath $WslRepository -PathType Container)) {
  throw "The selected Linux checkout is not accessible: $WslRepository"
}
if (-not (Test-Path -LiteralPath $wslCaptures -PathType Container)) {
  throw "The selected Linux checkout has no captures directory: $wslCaptures"
}
$outputDirectory = Join-Path (Join-Path $wslCaptures "blender") $Name
if (Test-Path -LiteralPath $outputDirectory -PathType Container) {
  $existingOutput = Get-ChildItem -LiteralPath $outputDirectory -Force | Select-Object -First 1
  if ($null -ne $existingOutput) {
    throw "Output directory is not empty; use a new render name to preserve existing artifacts: $outputDirectory"
  }
} else {
  $null = New-Item -ItemType Directory -Path $outputDirectory -Force
}
$scenePath = Join-Path $outputDirectory "scene.blend"
$videoPath = Join-Path $outputDirectory "render.mp4"
$framesPath = Join-Path $outputDirectory "frames"

$blenderArgs = @(
  "--background",
  "--python", $importer,
  "--",
  "--capture", $capturePath,
  "--description", $descriptionPath,
  "--output", $scenePath,
  "--video-output", $videoPath,
  "--resolution", $Resolution.ToString()
)
if ($CameraShots.Count -gt 0) {
  $blenderArgs += "--camera-shots"
  $blenderArgs += $CameraShots
}
if ($CinematicDof) {
  $blenderArgs += "--cinematic-dof"
}
if ($RenderPngFrames) {
  $blenderArgs += @("--render-dir", $framesPath)
}
if ($null -ne $checkpointPath) {
  $blenderArgs += @("--checkpoint", $checkpointPath)
}

Write-Host "Importer checkout: $repoRoot"
Write-Host "Dashboard-visible output: $outputDirectory"
& $blenderCommand.Source @blenderArgs
if ($LASTEXITCODE -ne 0) {
  throw "Blender exited with code $LASTEXITCODE. See the render manifest for the failure state if it was written."
}

Write-Host "Render complete: $(Join-Path $outputDirectory 'scene.manifest.json')"

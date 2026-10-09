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
  [ValidateSet("auto", "gpu", "cpu")]
  [string] $RenderDevice = "auto",
  [int] $Resolution = 1280,
  [double] $Fps = 0,
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
if ($Fps -lt 0) {
  throw "Fps must be zero to use the capture rate or a positive override."
}

if ([string]::IsNullOrWhiteSpace($WslRepository)) {
  if (-not [string]::IsNullOrWhiteSpace($env:ASCENTO_WSL_REPOSITORY)) {
    $WslRepository = $env:ASCENTO_WSL_REPOSITORY
  } else {
    $WslRepository = "\\wsl.localhost\Ubuntu\root\Ascento_MuJoCo_Playground"
  }
}

$canonicalWslPath = "\\wsl.localhost\Ubuntu\root\Ascento_MuJoCo_Playground"
if ([System.IO.Path]::GetFullPath($WslRepository).TrimEnd([char]92) -ne $canonicalWslPath) {
  throw "Blender renders must use the canonical WSL checkout: $canonicalWslPath"
}
$canonicalState = & wsl.exe --exec bash -lc 'cd /root/Ascento_MuJoCo_Playground && branch=$(git branch --show-current) && head=$(git rev-parse HEAD) && remote=$(git rev-parse origin/main) && if [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then dirty=dirty; else dirty=clean; fi && printf "%s\n%s\n%s\n%s\n" "$branch" "$head" "$remote" "$dirty"'
if ($LASTEXITCODE -ne 0 -or $canonicalState.Count -lt 4) {
  throw "Could not verify the canonical WSL checkout before rendering."
}
if ($canonicalState[0] -ne "main" -or $canonicalState[1] -ne $canonicalState[2] -or $canonicalState[3] -ne "clean") {
  throw "Blender rendering requires a clean WSL main checkout at origin/main."
}
$wslImporter = Join-Path $WslRepository "tools\blender\import_motion.py"
$wslLauncher = Join-Path $WslRepository "scripts\render_blender.ps1"
if (-not (Test-Path -LiteralPath $wslImporter -PathType Leaf)) {
  throw "Blender importer not found in the canonical checkout: $wslImporter"
}
if (-not (Test-Path -LiteralPath $wslLauncher -PathType Leaf)) {
  throw "Blender launcher not found in the canonical checkout: $wslLauncher"
}
$localLauncherText = [System.IO.File]::ReadAllText($PSCommandPath).Replace("`r`n", "`n")
$canonicalLauncherText = [System.IO.File]::ReadAllText($wslLauncher).Replace("`r`n", "`n")
if ($localLauncherText -cne $canonicalLauncherText) {
  throw "This Windows launcher is stale. Synchronize the Windows mirror with origin/main and run it again."
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
  "--python", $wslImporter,
  "--",
  "--capture", $capturePath,
  "--description", $descriptionPath,
  "--output", $scenePath,
  "--video-output", $videoPath,
  "--resolution", $Resolution.ToString(),
  "--render-device", $RenderDevice
)
if ($CameraShots.Count -gt 0) {
  $blenderArgs += "--camera-shots"
  $blenderArgs += $CameraShots
}
if ($Fps -gt 0) {
  $blenderArgs += @("--fps", $Fps.ToString([System.Globalization.CultureInfo]::InvariantCulture))
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

Write-Host "Canonical importer checkout: $WslRepository"
Write-Host "Dashboard-visible output: $outputDirectory"
& $blenderCommand.Source @blenderArgs
if ($LASTEXITCODE -ne 0) {
  throw "Blender exited with code $LASTEXITCODE. See the render manifest for the failure state if it was written."
}

Write-Host "Render complete: $(Join-Path $outputDirectory 'scene.manifest.json')"

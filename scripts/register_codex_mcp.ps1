[CmdletBinding()]
param(
    [string]$Name = "ascento",
    [string]$Distro = "Ubuntu",
    [switch]$Replace
)

$ErrorActionPreference = "Stop"

function Convert-ToWslPath([string]$WindowsPath) {
    if ($WindowsPath -notmatch "^([A-Za-z]):\\(.*)$") {
        throw "The checkout must be on a mounted Windows drive, got: $WindowsPath"
    }
    $drive = $Matches[1].ToLowerInvariant()
    $tail = $Matches[2].Replace("\", "/")
    return "/mnt/$drive/$tail"
}

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$serverPath = "$(Convert-ToWslPath $repoRoot)/scripts/ascento_mcp_server.sh"

if (-not (Get-Command codex -ErrorAction SilentlyContinue)) {
    throw "The Codex CLI is not on PATH. Open Codex Desktop, then run this script again."
}
if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    throw "wsl.exe is required because this project uses the Linux/WSL runtime."
}

$available = (& wsl.exe --list --quiet 2>$null | ForEach-Object {
    ($_ -replace ([string][char]0), "").Trim()
}) |
    Where-Object { $_ }
if ($available -notcontains $Distro) {
    throw "WSL distribution '$Distro' is unavailable. Found: $($available -join ', ')"
}

& wsl.exe --distribution $Distro --exec test -f $serverPath
if ($LASTEXITCODE -ne 0) {
    throw "The WSL server launcher is not reachable at $serverPath"
}

& codex mcp get $Name *> $null
$exists = $LASTEXITCODE -eq 0
if ($exists -and -not $Replace) {
    throw "A Codex MCP server named '$Name' already exists. Re-run with -Replace to replace it."
}
if ($exists) {
    & codex mcp remove $Name
    if ($LASTEXITCODE -ne 0) {
        throw "Could not remove the existing '$Name' MCP registration."
    }
}

& codex mcp add $Name -- wsl.exe --distribution $Distro --exec bash $serverPath
if ($LASTEXITCODE -ne 0) {
    throw "Codex could not register the '$Name' MCP server."
}

& codex mcp get $Name
if ($LASTEXITCODE -ne 0) {
    throw "Codex did not persist the '$Name' MCP registration."
}

Write-Host "Registered '$Name'. Open a new Codex Desktop task or restart the app to load its tools."

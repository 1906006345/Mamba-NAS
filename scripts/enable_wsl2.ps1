# Run this bootstrap from an elevated Windows PowerShell, then reboot if requested.
$ErrorActionPreference = "Stop"
$isAdministrator = ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator
)
if (-not $isAdministrator) {
    throw "Administrator privileges are required to enable WSL2 and its services."
}

dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
wsl.exe --set-default-version 2
wsl.exe --install -d Ubuntu-22.04 --no-launch
Write-Host "WSL2 and Ubuntu 22.04 were requested. Reboot Windows, launch Ubuntu once, then run scripts/setup_mamba_nas_wsl.sh."

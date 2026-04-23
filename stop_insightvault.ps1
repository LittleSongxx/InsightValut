[CmdletBinding()]
param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$ScriptArgs
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Get-WslLocation {
  param(
    [Parameter(Mandatory = $true)]
    [string]$WindowsPath
  )

  if ($WindowsPath -match '^[\\/]{2}wsl(?:\.localhost|\$)[\\/](?<Distro>[^\\/]+)(?<LinuxPath>(?:[\\/].*)?)$') {
    $linuxPath = ($Matches.LinuxPath -replace '\\', '/')
    if ([string]::IsNullOrWhiteSpace($linuxPath)) {
      $linuxPath = "/"
    }

    return @{
      Distro = $Matches.Distro
      LinuxPath = $linuxPath
    }
  }

  throw "This wrapper expects the project to be opened from a WSL path like \\\\wsl.localhost\\Ubuntu-22.04\\.... Current path: $WindowsPath"
}

$location = Get-WslLocation -WindowsPath $PSScriptRoot
& wsl.exe -d $location.Distro --cd $location.LinuxPath ./stop_insightvault.sh @ScriptArgs
exit $LASTEXITCODE

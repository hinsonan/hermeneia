param(
  [Parameter(Mandatory = $true)][string]$RepoOwner,
  [Parameter(Mandatory = $true)][string]$RepoName,
  [Parameter(Mandatory = $true)][string]$Tag,
  [Parameter(Mandatory = $true)][string]$Version,
  [Parameter(Mandatory = $true)][string]$AssetPrefix,
  [Parameter(Mandatory = $true)][string]$InstallDir,
  [Parameter(Mandatory = $true)][string]$SevenZipExe,
  [switch]$SkipLaunch
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

function Write-Info {
  param([string]$Message)
  Write-Host "[Hermeneia CUDA Installer] $Message"
}

function Invoke-DownloadWithRetry {
  param(
    [Parameter(Mandatory = $true)][string]$Uri,
    [Parameter(Mandatory = $true)][string]$Destination,
    [int]$MaxAttempts = 5
  )

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Write-Info "Downloading $Uri"
      Invoke-WebRequest -Uri $Uri -OutFile $Destination
      return
    }
    catch {
      if ($attempt -ge $MaxAttempts) {
        throw
      }
      Write-Info "Download failed (attempt $attempt/$MaxAttempts). Retrying in 2 seconds..."
      Start-Sleep -Seconds 2
    }
  }
}

function Get-WebView2Version {
  $clientId = "{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}"
  $registryPaths = @(
    "HKLM:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$clientId",
    "HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$clientId",
    "HKCU:\SOFTWARE\Microsoft\EdgeUpdate\Clients\$clientId",
    "HKCU:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\$clientId"
  )

  foreach ($path in $registryPaths) {
    if (Test-Path $path) {
      $pv = (Get-ItemProperty -Path $path -ErrorAction SilentlyContinue).pv
      if ($pv -and $pv -ne "0.0.0.0") {
        return $pv
      }
    }
  }

  return $null
}

function Ensure-WebView2Runtime {
  $versionDetected = Get-WebView2Version
  if ($versionDetected) {
    Write-Info "WebView2 Runtime detected: $versionDetected"
    return
  }

  Write-Info "WebView2 Runtime not found. Installing Evergreen bootstrapper..."
  $bootstrapperPath = Join-Path $env:TEMP "MicrosoftEdgeWebView2Setup.exe"
  Invoke-DownloadWithRetry -Uri "https://go.microsoft.com/fwlink/p/?LinkId=2124703" -Destination $bootstrapperPath

  $proc = Start-Process -FilePath $bootstrapperPath -ArgumentList "/silent", "/install" -Wait -PassThru
  if ($proc.ExitCode -ne 0) {
    throw "WebView2 installer failed with exit code $($proc.ExitCode)"
  }

  $versionDetected = Get-WebView2Version
  if (-not $versionDetected) {
    throw "WebView2 installation completed but runtime is still not detected"
  }

  Write-Info "WebView2 Runtime installed: $versionDetected"
}

function Read-ChecksumEntries {
  param([Parameter(Mandatory = $true)][string]$ChecksumFilePath)

  $entries = @()
  foreach ($line in (Get-Content -Path $ChecksumFilePath)) {
    if ($line -match "^([A-Fa-f0-9]{64}) \*(.+)$") {
      $entries += [PSCustomObject]@{
        Hash = $Matches[1].ToLowerInvariant()
        Name = $Matches[2]
      }
    }
  }

  if ($entries.Count -eq 0) {
    throw "No valid checksum entries found in $ChecksumFilePath"
  }

  return $entries
}

function Assert-FileHash {
  param(
    [Parameter(Mandatory = $true)][string]$FilePath,
    [Parameter(Mandatory = $true)][string]$ExpectedHash
  )

  $actual = (Get-FileHash -Path $FilePath -Algorithm SHA256).Hash.ToLowerInvariant()
  if ($actual -ne $ExpectedHash.ToLowerInvariant()) {
    throw "SHA256 mismatch for $FilePath. Expected $ExpectedHash but got $actual"
  }
}

function New-AppShortcut {
  param(
    [Parameter(Mandatory = $true)][string]$ShortcutPath,
    [Parameter(Mandatory = $true)][string]$TargetPath,
    [Parameter(Mandatory = $true)][string]$WorkingDirectory
  )

  $shell = New-Object -ComObject WScript.Shell
  $shortcut = $shell.CreateShortcut($ShortcutPath)
  $shortcut.TargetPath = $TargetPath
  $shortcut.WorkingDirectory = $WorkingDirectory
  $shortcut.IconLocation = "$TargetPath,0"
  $shortcut.Save()
}

$tempRoot = Join-Path $env:TEMP ("hermeneia-cuda-installer-" + [Guid]::NewGuid().ToString("N"))
$downloadDir = Join-Path $tempRoot "downloads"
$extractDir = Join-Path $tempRoot "extract"

try {
  Write-Info "Preparing installation for Hermeneia CUDA $Version"
  New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
  New-Item -ItemType Directory -Force -Path $extractDir | Out-Null

  if (-not (Test-Path $SevenZipExe)) {
    throw "Bundled extractor not found: $SevenZipExe"
  }

  $releaseBaseUrl = "https://github.com/$RepoOwner/$RepoName/releases/download/$Tag"
  $checksumFileName = "$AssetPrefix.sha256"
  $checksumFilePath = Join-Path $downloadDir $checksumFileName

  Invoke-DownloadWithRetry -Uri "$releaseBaseUrl/$checksumFileName" -Destination $checksumFilePath
  $checksumEntries = Read-ChecksumEntries -ChecksumFilePath $checksumFilePath

  $archiveEntries = @($checksumEntries | Where-Object { $_.Name -like "$AssetPrefix.7z*" } | Sort-Object Name)
  if ($archiveEntries.Count -eq 0) {
    throw "No archive entries found for prefix $AssetPrefix in $checksumFileName"
  }

  $downloadedFiles = @()
  foreach ($entry in $archiveEntries) {
    $dest = Join-Path $downloadDir $entry.Name
    Invoke-DownloadWithRetry -Uri "$releaseBaseUrl/$($entry.Name)" -Destination $dest
    Assert-FileHash -FilePath $dest -ExpectedHash $entry.Hash
    $downloadedFiles += Get-Item $dest
  }

  Ensure-WebView2Runtime

  $archiveToExtract = $downloadedFiles | Where-Object { $_.Name -match "\.7z\.001$" } | Select-Object -First 1
  if (-not $archiveToExtract) {
    $archiveToExtract = $downloadedFiles | Where-Object { $_.Name -eq "$AssetPrefix.7z" } | Select-Object -First 1
  }
  if (-not $archiveToExtract) {
    throw "Could not determine archive root (.7z or .7z.001)"
  }

  Write-Info "Extracting archive $($archiveToExtract.Name)"
  & $SevenZipExe x "-o$extractDir" "-y" $archiveToExtract.FullName | Out-Host
  if ($LASTEXITCODE -ne 0) {
    throw "Extraction failed with exit code $LASTEXITCODE"
  }

  $stagedRoot = Join-Path $extractDir "hermeneia-cuda"
  $stagedExe = Join-Path $stagedRoot "hermeneia.exe"
  if (-not (Test-Path $stagedExe)) {
    $candidate = Get-ChildItem -Path $extractDir -Recurse -Filter "hermeneia.exe" -File | Select-Object -First 1
    if (-not $candidate) {
      throw "Unable to locate hermeneia.exe after extraction"
    }
    $stagedRoot = $candidate.DirectoryName
    $stagedExe = $candidate.FullName
  }

  Write-Info "Installing to $InstallDir"
  if (Test-Path $InstallDir) {
    Remove-Item $InstallDir -Recurse -Force
  }
  New-Item -ItemType Directory -Force -Path $InstallDir | Out-Null
  Copy-Item (Join-Path $stagedRoot "*") $InstallDir -Recurse -Force

  $installedExe = Join-Path $InstallDir "hermeneia.exe"
  if (-not (Test-Path $installedExe)) {
    throw "Installed executable not found: $installedExe"
  }

  $desktopPath = [Environment]::GetFolderPath("Desktop")
  $programsPath = [Environment]::GetFolderPath("Programs")
  $startMenuDir = Join-Path $programsPath "Hermeneia CUDA"

  New-Item -ItemType Directory -Force -Path $startMenuDir | Out-Null
  New-AppShortcut -ShortcutPath (Join-Path $desktopPath "Hermeneia CUDA.lnk") -TargetPath $installedExe -WorkingDirectory $InstallDir
  New-AppShortcut -ShortcutPath (Join-Path $startMenuDir "Hermeneia CUDA.lnk") -TargetPath $installedExe -WorkingDirectory $InstallDir

  Write-Info "Installation complete."
  if (-not $SkipLaunch.IsPresent) {
    Write-Info "Launching Hermeneia CUDA..."
    Start-Process -FilePath $installedExe -WorkingDirectory $InstallDir
  }
}
finally {
  if (Test-Path $tempRoot) {
    Remove-Item $tempRoot -Recurse -Force -ErrorAction SilentlyContinue
  }
}

Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"

!ifndef OUTPUT_DIR
  !error "OUTPUT_DIR must be defined"
!endif
!ifndef APP_VERSION
  !error "APP_VERSION must be defined"
!endif
!ifndef RELEASE_TAG
  !error "RELEASE_TAG must be defined"
!endif
!ifndef REPO_OWNER
  !error "REPO_OWNER must be defined"
!endif
!ifndef REPO_NAME
  !error "REPO_NAME must be defined"
!endif
!ifndef ASSET_PREFIX
  !error "ASSET_PREFIX must be defined"
!endif

Name "Hermeneia CUDA ${APP_VERSION}"
OutFile "${OUTPUT_DIR}\hermeneia_${APP_VERSION}_windows_x64_cuda_installer.exe"
RequestExecutionLevel user
InstallDir "$LOCALAPPDATA\Programs\Hermeneia-CUDA"
SetCompressor /SOLID lzma

!define MUI_ABORTWARNING
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_LANGUAGE "English"

Section "Install"
  SetOutPath "$PLUGINSDIR"
  File "${__FILEDIR__}\bootstrap-install.ps1"
  File "${__FILEDIR__}\third_party\7z\7z.exe"
  File "${__FILEDIR__}\third_party\7z\7z.dll"

  DetailPrint "Running Hermeneia CUDA bootstrap installer..."
  nsExec::ExecToLog '"$SYSDIR\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "$PLUGINSDIR\bootstrap-install.ps1" -RepoOwner "${REPO_OWNER}" -RepoName "${REPO_NAME}" -Tag "${RELEASE_TAG}" -Version "${APP_VERSION}" -AssetPrefix "${ASSET_PREFIX}" -InstallDir "$INSTDIR" -SevenZipExe "$PLUGINSDIR\7z.exe"'
  Pop $0
  ${If} $0 != 0
    MessageBox MB_ICONSTOP|MB_OK "Hermeneia CUDA installation failed (code $0). Check installer log details."
    Abort
  ${EndIf}
SectionEnd

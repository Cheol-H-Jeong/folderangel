; Inno Setup script — produces a friendly Windows installer for Folder1004.
;
; Build with:  iscc scripts\folder1004.iss
; Output:      dist\Folder1004-Setup.exe
;
; Assumes ``scripts\build_windows.ps1`` has already produced
; ``dist\folder1004\folder1004.exe`` and its supporting bundle.

#define AppName "Folder1004"
#define AppVersion "1.0.0"
#define AppPublisher "Folder1004"
#define AppExeName "folder1004.exe"

[Setup]
AppId={{4E1C2F32-9C7D-4F70-AB10-FA15FA15FA15}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
OutputDir=..\dist
OutputBaseFilename=Folder1004-Setup
Compression=lzma
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "korean";  MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\folder1004\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent

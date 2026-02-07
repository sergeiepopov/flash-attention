# PowerShell script to copy PyTorch DLLs to the build directory

# Find PyTorch installation
$pythonCmd = "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
$torchLibPath = py -c $pythonCmd

Write-Host "PyTorch lib path: $torchLibPath"

# Find the executable
$buildPaths = @(
    "C:\source\flash-attention\build\Debug",
    "C:\source\flash-attention\build\Release",
    "C:\source\flash-attention\build\RelWithDebInfo",
    "C:\source\flash-attention\build\MinSizeRel"
)

foreach ($buildPath in $buildPaths) {
    if (Test-Path "$buildPath\flash-attention.exe") {
        Write-Host "Found executable in: $buildPath"
        Write-Host "Copying DLLs..."
        
        # Copy all DLLs from torch lib directory
        Get-ChildItem "$torchLibPath\*.dll" | ForEach-Object {
            Copy-Item $_.FullName -Destination $buildPath -Force
            Write-Host "  Copied: $($_.Name)"
        }
        
        Write-Host "`nDLL copy complete!"
        Write-Host "You can now run: $buildPath\flash-attention.exe"
    }
}

# Also check for Python DLL
$pythonDllPath = python -c "import sys; import os; print(os.path.join(sys.prefix, 'python314.dll'))"
if (Test-Path $pythonDllPath) {
    foreach ($buildPath in $buildPaths) {
        if (Test-Path "$buildPath\flash-attention.exe") {
            Copy-Item $pythonDllPath -Destination $buildPath -Force
            Write-Host "  Copied: python314.dll"
        }
    }
}

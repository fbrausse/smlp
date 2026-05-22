Add-Type -AssemblyName System.Windows.Forms
$screen = [System.Windows.Forms.Screen]::PrimaryScreen
$source = @"
using System;
using System.Runtime.InteropServices;
public class DPI {
    [DllImport("gdi32.dll")]
    public static extern int GetDeviceCaps(IntPtr hdc, int nIndex);
    [DllImport("user32.dll")]
    public static extern IntPtr GetDC(IntPtr hwnd);
}
"@
Add-Type -TypeDefinition $source
$hdc = [DPI]::GetDC([IntPtr]::Zero)
$width = [DPI]::GetDeviceCaps($hdc, 118)   # DESKTOPHORZRES
$height = [DPI]::GetDeviceCaps($hdc, 117)  # DESKTOPVERTRES
Write-Output "${width}x${height}"
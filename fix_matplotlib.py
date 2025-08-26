#!/usr/bin/env python3
"""
Quick fix for matplotlib compatibility issues on Windows
"""

import subprocess
import sys

def run_command(command):
    """Run a command and return the result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def fix_matplotlib():
    """Fix matplotlib compatibility issues"""
    print("=" * 60)
    print("Fixing matplotlib compatibility issues on Windows...")
    print("=" * 60)
    
    # Step 1: Uninstall problematic matplotlib
    print("\n1. Uninstalling current matplotlib...")
    success, stdout, stderr = run_command("pip uninstall matplotlib -y")
    if success:
        print("✓ matplotlib uninstalled successfully")
    else:
        print(f"⚠ matplotlib uninstall: {stderr}")
    
    # Step 2: Install compatible matplotlib
    print("\n2. Installing matplotlib 3.6.3 (Windows-compatible)...")
    success, stdout, stderr = run_command("pip install matplotlib==3.6.3")
    if success:
        print("✓ matplotlib 3.6.3 installed successfully")
    else:
        print(f"✗ matplotlib installation failed: {stderr}")
        return False
    
    # Step 3: Test matplotlib import
    print("\n3. Testing matplotlib import...")
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"✓ matplotlib {matplotlib.__version__} imported successfully")
        print("✓ matplotlib.pyplot imported successfully")
        return True
    except Exception as e:
        print(f"✗ matplotlib import test failed: {e}")
        return False

def install_stable_requirements():
    """Install the stable Windows requirements"""
    print("\n" + "=" * 60)
    print("Installing stable Windows-compatible packages...")
    print("=" * 60)
    
    # Install stable requirements
    success, stdout, stderr = run_command("pip install -r requirements_windows_stable.txt")
    if success:
        print("✓ All stable packages installed successfully")
        return True
    else:
        print(f"✗ Package installation failed: {stderr}")
        return False

def main():
    """Main fix function"""
    print("Matplotlib Compatibility Fix for Windows")
    print("This script will fix matplotlib import issues on Windows")
    
    # Fix matplotlib
    if not fix_matplotlib():
        print("\n❌ matplotlib fix failed. Please check the error messages above.")
        return
    
    # Ask if user wants to install stable requirements
    print("\n" + "=" * 60)
    response = input("Do you want to install all stable Windows-compatible packages? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        if install_stable_requirements():
            print("\n✅ All fixes applied successfully!")
            print("\nYou can now test your setup with:")
            print("  python test_setup.py")
        else:
            print("\n❌ Package installation failed. Please check the error messages above.")
    else:
        print("\n✅ matplotlib fix completed!")
        print("\nYou can now test your setup with:")
        print("  python test_setup.py")
    
    print("\nIf you continue to have issues, try:")
    print("1. Restart your Python environment")
    print("2. Use the stable requirements: pip install -r requirements_windows_stable.txt")
    print("3. Check the WINDOWS_SETUP_GUIDE.md for more solutions")

if __name__ == "__main__":
    main()



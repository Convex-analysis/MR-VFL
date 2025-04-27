"""
Script to install required dependencies for Mamba scheduler training.
"""

import subprocess
import sys

def install_dependencies():
    """Install required dependencies"""
    dependencies = [
        "torch",
        "numpy",
        "matplotlib",
        "tqdm",
    ]
    
    # Optional dependencies
    optional_dependencies = [
        "mamba_ssm",  # Optional, will use GRU as fallback if not available
    ]
    
    print("Installing required dependencies...")
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
    
    print("\nAttempting to install optional dependencies...")
    for dep in optional_dependencies:
        try:
            print(f"Installing {dep}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", dep])
        except subprocess.CalledProcessError:
            print(f"Failed to install {dep}. Will use fallback implementation.")
    
    print("\nAll dependencies installed successfully!")

if __name__ == "__main__":
    install_dependencies()

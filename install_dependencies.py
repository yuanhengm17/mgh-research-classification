import subprocess
import sys

# List of required packages
required_packages = [
    "firebase-admin",
    "pandas",
    "numpy",
    "matplotlib",
    "torch",
]

def install_package(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"Successfully installed {package}")
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}")

# Install each package
if __name__ == "__main__":
    print("Starting installation of required packages...")
    for package in required_packages:
        install_package(package)
    print("All dependencies installed.")

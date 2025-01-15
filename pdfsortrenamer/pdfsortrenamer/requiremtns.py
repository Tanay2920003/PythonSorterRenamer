
import subprocess
import sys

def install_requirements():
    try:
        print("Installing requirements from requirements.txt...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("All requirements installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

if __name__ == "__main__":
    install_requirements()

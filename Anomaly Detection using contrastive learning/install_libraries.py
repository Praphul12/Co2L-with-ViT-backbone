import subprocess

def install_libraries():
    libraries = [
        "tensorboard_logger",
        "torch",
        "torchvision",
        "numpy",
        "apex",  # Requires NVIDIA Apex, may need manual installation
    ]
    
    for lib in libraries:
        try:
            subprocess.run(["pip", "install", lib], check=True)
        except subprocess.CalledProcessError:
            print(f"Failed to install {lib}")

if __name__ == "__main__":
    install_libraries()

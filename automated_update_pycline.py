import toml
import shutil
import subprocess
import os
import re
from datetime import datetime

# Paths
PYPROJECT_FILE = "pyproject.toml"
DIST_FOLDER = "dist"
OLD_BUILDS_FOLDER = "old_builds"

def update_version():
    """ Reads, increments, and updates the package version in pyproject.toml. """
    # Load pyproject.toml
    with open(PYPROJECT_FILE, "r") as f:
        config = toml.load(f)
    
    # Get current version
    current_version = config["project"]["version"]
    
    # Increment patch version (e.g., 1.2.3 → 1.2.4)
    version_parts = list(map(int, current_version.split(".")))
    version_parts[-1] += 1  # Increment last part
    new_version = ".".join(map(str, version_parts))

    # Update version in pyproject.toml
    config["project"]["version"] = new_version
    with open(PYPROJECT_FILE, "w") as f:
        toml.dump(config, f)

    print(f"🔄 Updated version: {current_version} → {new_version}")
    return new_version

def move_old_builds():
    """ Moves old build files to a subfolder. """
    if not os.path.exists(DIST_FOLDER):
        print("⚠ No old builds found, skipping...")
        return

    # Ensure old_builds/ exists
    if not os.path.exists(OLD_BUILDS_FOLDER):
        os.makedirs(OLD_BUILDS_FOLDER)

    # Create timestamped subfolder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    old_build_dest = os.path.join(OLD_BUILDS_FOLDER, f"build_{timestamp}")
    os.makedirs(old_build_dest)

    # Move old files
    for file in os.listdir(DIST_FOLDER):
        shutil.move(os.path.join(DIST_FOLDER, file), old_build_dest)
    
    print(f"📦 Moved old builds to: {old_build_dest}")

def build_package():
    """ Runs the Python build process. """
    print("🚀 Building package...")
    subprocess.run(["python", "-m", "build"], check=True)

def check_package():
    """ Runs Twine to check if the package is valid. """
    print("🔍 Checking package with Twine...")
    subprocess.run(["twine", "check", os.path.join(DIST_FOLDER, "*")], check=True)

def main():
    """ Main script execution """
    new_version = update_version()
    move_old_builds()
    build_package()
    check_package()
    print(f"✅ Package {new_version} built and ready!")

if __name__ == "__main__":
    main()

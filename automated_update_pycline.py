import toml
import shutil
import subprocess
import os
from datetime import datetime

# Paths
PYPROJECT_FILE = "pyproject.toml"
DIST_FOLDER = "dist"
OLD_BUILDS_FOLDER = "old_builds"
SRC_FOLDER = "src"

def set_pythonpath():
    """ Sets the PYTHONPATH environment variable to the src folder. """
    current_pythonpath = os.environ.get("PYTHONPATH", "")
    new_pythonpath = os.path.abspath(SRC_FOLDER)
    if current_pythonpath:
        new_pythonpath = f"{new_pythonpath}{os.pathsep}{current_pythonpath}"
    os.environ["PYTHONPATH"] = new_pythonpath
    print(f"🔧 PYTHONPATH set to: {os.environ['PYTHONPATH']}")

def run_tests():
    """ Runs the tests using unittest and returns True if all tests pass, False otherwise. """
    print("🧪 Running tests...")
    result = subprocess.run(["python", "-m", "unittest", "discover"], capture_output=True, text=True)
    print(result.stdout)
    return result.returncode == 0

def get_new_version(current_version):
    """ Asks the user how to update the version and returns the new version string. """
    version_parts = list(map(int, current_version.split(".")))

    print(f"\n🔢 Current version: {current_version}")
    print("How do you want to update the version?")
    print("1️⃣ Patch (e.g., 0.0.1 → 0.0.2)")
    print("2️⃣ Minor (e.g., 0.0.1 → 0.1.0)")
    print("3️⃣ Major (e.g., 0.1.0 → 1.0.0)")

    choice = input("Enter 1, 2, or 3: ").strip()
    
    if choice == "1":
        version_parts[2] += 1  # Increment PATCH version
    elif choice == "2":
        version_parts[1] += 1  # Increment MINOR version
        version_parts[2] = 0   # Reset PATCH
    elif choice == "3":
        version_parts[0] += 1  # Increment MAJOR version
        version_parts[1] = 0   # Reset MINOR
        version_parts[2] = 0   # Reset PATCH
    else:
        print("⚠ Invalid input. Keeping current version.")
        return current_version  # Return unchanged version

    return ".".join(map(str, version_parts))

def update_version():
    """ Reads, updates, and writes the package version in pyproject.toml. """
    # Load pyproject.toml
    with open(PYPROJECT_FILE, "r") as f:
        config = toml.load(f)

    # load source/conf.py to update the version
    with open("source/conf.py", "r") as f:
        lines = f.readlines()
    for i, line in enumerate(lines):
        if "release" in line:
            lines[i] = f"release = '{config['project']['version']}'\n"
            break
    with open("source/conf.py", "w") as f:
        f.writelines(lines)
        
    # Get current version
    current_version = config["project"]["version"]

    # Ask user for version update type
    new_version = get_new_version(current_version)

    # Update pyproject.toml
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

def upload_to_pypi():
    """ Uploads the package to PyPI using Twine. """

    print("\nPublishing package to Test-PyPI or PyPI")
    print("Which repository do you want to use?")
    print("1️⃣ Test-PyPI (for testing)")
    print("2️⃣ PyPI (for production)")

    choice = input("Enter 1 or 2: ").strip()
    repository = "testpypi" if choice == "1" else "pypi"

    print(f"📤 Uploading package to {repository}...")
    print("Do you want to proceed?")
    choice = input("Enter 'y' to upload, or anything else to skip: ").strip().lower()
    if choice != "y":
        print("⏹ Skipping upload.")
        return
    
    subprocess.run(["twine", "upload", "--repository", repository, os.path.join(DIST_FOLDER, "*")], check=True)
    
def main():
    """ Main script execution """
    set_pythonpath()
    if run_tests():
        new_version = update_version()
        move_old_builds()
        build_package()
        check_package()
        upload_to_pypi()
        print(f"✅ Package {new_version} built and ready!")
    else:
        print("❌ Tests failed. Version not updated.")

if __name__ == "__main__":
    main()
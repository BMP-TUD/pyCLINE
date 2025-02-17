from . import recovery_methods
from . import model
from . import generate_data
from . import example
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyCLINE")
except PackageNotFoundError:
    __version__ = "0.0.0"  # Default when package is not installed

__all__ = ["recovery_methods", "model", "generate_data", "example"]
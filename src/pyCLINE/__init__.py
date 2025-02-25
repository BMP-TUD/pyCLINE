"""
pyCLINE: Python implementation of the CLINE method introduced by Prokop, Billen, Frolov and Gelens (2025).

Modules:
    recovery_methods: Methods for recovering nullclines.
    model: Model definitions and utilities.
    generate_data: Functions for generating synthetic data.
    example: Example usage and demonstrations.
"""

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

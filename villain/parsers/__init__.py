"""Hand history parsers. Importing this package registers every known format."""

from . import pokernow  # noqa: F401  (registers itself)
from .base import UnknownFormat, detect, parse_file, parse_paths, register  # noqa: F401

__all__ = ["UnknownFormat", "detect", "parse_file", "parse_paths", "register"]

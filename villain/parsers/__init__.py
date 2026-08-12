"""Hand history parsers. Importing this package registers every known format."""

from .base import UnknownFormat, detect, parse_file, parse_paths, register  # noqa: F401
from . import pokernow  # noqa: F401  (registers itself)

__all__ = ["UnknownFormat", "detect", "parse_file", "parse_paths", "register"]

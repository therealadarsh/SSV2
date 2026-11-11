"""SSV2: a retrieval service that measures its own quality."""

from importlib.metadata import version

# Read from the installed package metadata, so pyproject.toml is the only place it is set.
__version__ = version("ssv2")

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("ifermi")
except PackageNotFoundError:
    # package is not installed
    __version__ = ""

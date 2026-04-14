try:
    from importlib.metadata import version, PackageNotFoundError

    __version__ = version("cosmic-foundry")
except PackageNotFoundError:
    __version__ = "0.0.0.dev0"

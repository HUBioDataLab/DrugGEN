import os
import json
import warnings


def _get_current_package_version():
    import importlib.metadata

    return importlib.metadata.version("model_suite")


class BaseConfig:
    model_suite_version: str = _get_current_package_version()

    def to_json(self, path: os.PathLike):
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")

        if os.path.exists(path):
            warnings.warn(f"File {path} already exists. Overwriting.")

        contents = dir(self)
        # filter out callables and private variables
        contents = [
            c
            for c in contents
            if not c.startswith("_") and not callable(getattr(self, c))
        ]

        # make a dictionary of the non-callable, non-private variables
        contents = {c: getattr(self, c) for c in contents}

        with open(path, "w") as f:
            json.dump(contents, f, indent=4)

    @classmethod
    def from_json(cls, path: os.PathLike):
        if os.path.isdir(path):
            path = os.path.join(path, "config.json")
        with open(path, "r") as f:
            config = json.load(f)
        return cls(**config)

from pathlib import Path


def _get_parent_dir() -> Path:
    curr_path = Path(__file__).resolve()
    for directory in curr_path.parents:
            if (directory / "pyproject.toml").is_file():
                return directory
    raise FileNotFoundError("Could not find the parent directory containing pyproject.toml")

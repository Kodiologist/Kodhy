import hy
import pytest

def pytest_collect_file(parent, path):
    if (path.ext == ".hy" and path.basename != "__init__.hy"):
        return pytest.Module.from_parent(parent, fspath=path)

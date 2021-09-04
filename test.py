import os, pathlib
import pytest

if __name__ == "__main__":
    os.chdir( pathlib.Path.cwd() / 'test' )
    pytest.main()

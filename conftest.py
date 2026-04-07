# Ensures the project root is on sys.path so that `src` is importable
# when pytest is invoked from any working directory.
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import os
import importlib

# List all files in the current directory
files = os.listdir(os.path.dirname(__file__))

# Import each Python file
for f in files:
    module_name, ext = os.path.splitext(f)
    if ext == ".py" and module_name != "__init__":
        globals()[module_name] = importlib.import_module("." + module_name, package=__name__)
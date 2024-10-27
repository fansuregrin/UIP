import importlib
import glob
import os


ds_modules = ['data.dataset.' + m.rstrip('.py') for m in 
              glob.glob('*_ds.py', root_dir=os.path.dirname(__file__))]
for ds_module in ds_modules:
    importlib.import_module(ds_module)
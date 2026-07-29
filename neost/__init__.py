""" NEoST: Open-source code for equation of state inference via nested sampling. """
from __future__ import print_function
from importlib.metadata import metadata

meta = metadata(__package__ or __name__)
__version__ = meta.get('Version')
__author__ = meta.get('Author-email')

try:
    __NEOST_SETUP__
except NameError:
    __NEOST_SETUP__ = False

if not __NEOST_SETUP__:

    from . import global_imports

    if global_imports._verbose:
        print("/=============================================\\")
        print("|  NEoST: Nested Equation of State Sampling   |")
        print("|---------------------------------------------|")
        print("|    See the documentation and user guide!    |")
        print("\\=============================================/\n")
        print(f'Imported NEoST version: {__version__}')

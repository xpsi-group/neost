""" NEoST: Open-source code for equation of state inference via nested sampling. """
from __future__ import print_function
__version__ = "0.10.0"
__author__ = "Geert Raaijmakers"

try:
    __NEOST_SETUP__
except NameError:
    __NEOST_SETUP__ = False

if not __NEOST_SETUP__:

    from . import global_imports

    if global_imports._verbose:
        print("/=============================================\\")
        print("|  NEoST: Equation of State Nested Sampling   |")
        print("|---------------------------------------------|")
        print("|    See the documentation and user guide!    |")
        print("\\=============================================/\n")

        print('Imported NEoST version: %s' % __version__)


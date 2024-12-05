# This file specifies how the Cython TOV solvers are to be compiled.
# It is not necessary at all if only Python TOV solvers are to be used.

# Standard libraries
import os
from setuptools import Extension, setup
import subprocess as sub
import sys

# 3rd party
import numpy as np

OS = sys.platform

# Check the operating system
if 'darwin' not in OS and 'linux' not in OS:
    raise ValueError('Unsupported operating system ' + OS)

# Make sure GSL is installed
try:
    gsl_version = sub.check_output(['gsl-config','--version'])[:-1].decode("utf-8")
    gsl_prefix = sub.check_output(['gsl-config','--prefix'])[:-1].decode("utf-8")
except Exception:
    print('GNU Scientific Library cannot be located.')
    raise
else:
    print('GSL version: ' + gsl_version)
    libraries = ['gsl','gslcblas','m']
    library_dirs = [gsl_prefix + '/lib']

# Common includes, linker arguments, compiler arguments
include_dirs = [np.get_include(), gsl_prefix+'/include', '.']
extra_link_args = ['-fopenmp']
extra_compile_args = ['-fopenmp','-Wno-unused-function', '-Wno-uninitialized']

# OS-specific settings
if 'darwin' in OS:
    extra_link_args = []
    extra_compile_args = ['-Wno-unused-function', '-Wno-uninitialized']
    # Using compiler of clang with llvm installed
    # os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
    # os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"
    library_dirs.extend('/usr/local/opt/llvm/lib')
    library_dirs.extend('/opt/local/lib')
    extra_compile_args.append('-Wno-#warnings')
    extra_compile_args.append('-Wno-error=format-security')
    include_dirs.extend(['/usr/local/include', '/usr/local/opt/llvm/include', './neost/tovsolvers/'])

else:
    # point to shared library at compile time so runtime resolution
    # is not affected by environment variables, but is determined
    # by the binary itself
    extra_link_args.append('-Wl,-rpath=%s' % (gsl_prefix + '/lib'))
    extra_compile_args.append('-Wno-cpp')
    include_dirs.append('./neost/tovsolvers/')

# Specify the Cython modules to be compiled
TOVr = Extension(
            name = 'neost.tovsolvers.TOVr',
            sources = ['neost/tovsolvers/TOVr.pyx'],
            libraries = libraries,
            library_dirs = library_dirs,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )

TOVdm = Extension(
            name = 'neost.tovsolvers.TOVdm',
            sources = ['neost/tovsolvers/TOVdm.pyx'],
            libraries = libraries,
            library_dirs = library_dirs,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )

TOVh = Extension(
            name = 'neost.tovsolvers.TOVh',
            sources = ['neost/tovsolvers/TOVh.pyx'],
            libraries = libraries,
            library_dirs = library_dirs,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args,
        )

setup(
    ext_modules=[TOVr, TOVdm, TOVh]
)

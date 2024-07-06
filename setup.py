"""
To compile to C code, and then compile the C code and link libraries, and
install as a Python package:
    --> python setup.py install [--user]
To build extensions in place and not install:
    --> python setup.py build_ext -i
The extension is then constructed in the source directory. To import NEoST,
ensure the sys.path searches the ``src`` directory. If a Python module is
modified, the package then does not need to be reinstalled for usage, but for
use in an interactive environment, the module must be reloaded
(or the kernel must be restarted).
"""

import os

try:
    from setuptools import setup, find_packages, Extension
except ImportError:
    from distutils.core import setup, find_packages, Extension

if __name__ == '__main__':
    import numpy
    import sys
    OS = sys.platform
    print(sys.path)

    if '--nocython' in sys.argv:
        sys.argv.remove('--nocython')
        setup(
            name='NEoST',
            version='0.9.0',
            author='NEoST core team',
            author_email='A.L.Watts@uva.nl',
            url='',
            license='GNU GPLv3',
            description="""NEoST: An open-source code for dense matter equation of state inference via nested sampling.""",
            packages=find_packages(),
            install_requires=[],
            include_package_data=True,
            classifiers=['Intended Audience :: Science/Research',
                         'Operating System :: Mac OS X, Linux',
                         'License :: OSI Approved :: GNU GPLv3 License',
                         'Programming Language :: Python'])

    else:

        if 'darwin' in OS or 'linux' in OS:
            print('Operating system: ' + OS)

            try:
                import subprocess as sub
            except ImportError:
                print('The subprocess module is required to locate the GSL library.')
                raise

            try:
                gsl_version = sub.check_output(['gsl-config','--version'])[:-1].decode("utf-8")
                gsl_prefix = sub.check_output(['gsl-config','--prefix'])[:-1].decode("utf-8")
            except Exception:
                print('GNU Scientific Library cannot be located.')
                raise
            else:
                print('GSL version: ' + gsl_version)
                if '--Cartesius' in sys.argv:
                    sys.argv.remove('--Cartesius')
                    libraries = ['gsl','mkl_intel_lp64','mkl_sequential','mkl_core']
                else:
                    libraries = ['gsl','gslcblas','m']
                library_dirs = [gsl_prefix + '/lib']

            if 'darwin' in OS:
                # Using compiler of clang with llvm installed
                os.environ["CC"] = "/usr/local/opt/llvm/bin/clang"
                os.environ["CXX"] = "/usr/local/opt/llvm/bin/clang++"
                library_dirs.append('/usr/local/opt/llvm/lib')
                library_dirs.append('/opt/local/lib')
                extra_link_args = ['-fopenmp']
                extra_compile_args = ['-fopenmp',
                                      '-Wno-unused-function',
                                      '-Wno-uninitialized',
                                      '-Wno-#warnings',
                                      '-Wno-error=format-security']
                include_dirs = ['/usr/local/include', '/usr/local/opt/llvm/include',
                                gsl_prefix + '/include',
                                './neost',
                                '.',
                                numpy.get_include()]

            else:
                # point to shared library at compile time so runtime resolution
                # is not affected by environment variables, but is determined
                # by the binary itself
                extra_link_args = ['-fopenmp','-Wl,-rpath=%s' % (gsl_prefix + '/lib')]
                extra_compile_args = ['-fopenmp',
                                      '-Wno-unused-function',
                                      '-Wno-uninitialized',
                                      '-Wno-cpp']
                include_dirs = [gsl_prefix + '/include',
                                './neost/tovsolvers/',
                                '.',
                                numpy.get_include()]
        else:
            print('Unsupported operating system.')
            raise Exception

        cmdclass = {}
        try:
            import Cython
            print('Cython.__version__ == %s' % Cython.__version__)
            from Cython.Distutils import build_ext
        except ImportError:
            print('Cannot use Cython. Trying to build extension from C files...')
            try:
                from distutils.command import build_ext
            except ImportError:
                print('Cannot import build_ext from distutils...')
                raise
            else:
                cmdclass['build_ext'] = build_ext
                file_extension = '.c'
        else:
            print('Using Cython to build extension from .pyx files...')
            file_extension = '.pyx'
            cmdclass['build_ext'] = build_ext

        def EXTENSION(modname):

            pathname = modname.replace('.', os.path.sep)

            return Extension(modname,
                             [pathname + file_extension],
                             language='c',
                             libraries=libraries,
                             library_dirs=library_dirs,
                             include_dirs=include_dirs,
                             extra_compile_args=extra_compile_args,
                             extra_link_args=extra_link_args,
                             optional=True)

        modnames = ['neost.tovsolvers.TOVr', 'neost.tovsolvers.TOVh']

        extensions = []

        for mod in modnames:
            extensions.append(EXTENSION(mod))

        setup(
            name='NEoST',
            version='0.9.0',
            author='NEoST core team',
            author_email='A.L.Watts@uva.nl',
            url='',
            license='GNU GPLv3',
            description="""NEoST: An open-source code for dense matter equation of state inference via nested sampling.""",
            packages=find_packages(),
            install_requires=[],
            include_package_data=True,
            include_dirs=include_dirs,
            ext_modules=extensions,
            cmdclass=cmdclass,
            classifiers=['Intended Audience :: Science/Research',
                         'Operating System :: Mac OS X, Linux',
                         'License :: OSI Approved :: GNU GPLv3 License',
                         'Programming Language :: Python'])

else:
    pass

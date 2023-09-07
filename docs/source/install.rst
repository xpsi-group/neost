.. _install:

Installation
============

NEoST is a Python software package that performs neutron star equation 
of state inference process. it is an open source package that was 
and is available on `GitHub`_ and can be cloned as:

.. _GitHub: https://github.com/xpsi-group/neost.git

.. code-block:: bash

	git clone https://github.com/xpsi-group/neost.git </path/to/neost>

On this page we will describe how to install NEoST and it's prerequisites 
on your local machine.

.. note::
	
	For installation on a high-performance computing system, we direct 
	the reader to the alternative instructions at the bottom of this page.

.. _conda_env:

Prerequisite Python Packages
----------------------------

NEoST was developed in Python 3, as such we recommend creating a conda virtual 
environment with anaconda3 according to the instructions below. This will 
ensure installing NEoST won't break any existing software packages.

.. _basic_env:

If Python 3 has been installed using the Anaconda Distribution, a new virtual 
Python environment can then be created by entering the following command 
into the terminal:

.. code-block:: bash

	conda create --name NEoST --clone base

This command will clone the base Python environment into a new environment 
called NEoST. Next, in order to be able to install packages on the new NEoST 
environment enter the following: 

.. code-block:: bash

	conda activate NEoST

This command changes the active virtual environment from the default base 
environment to the new NEoST environment and needs to be entered any time 
NEoST is used.

A next step is to make sure that a number of packages have a specific
version of them installed, these are the following packages and versions:

.. code-block:: bash

	alive_progress v2.4.1
	contextlib v0.6.0.post1
	corner v2.2.1
	GetDist v1.3.4
	Matplotlib v3.3.4
	NumPy v1.21.2
	SciPy v1.6.2
	Seaborn v0.11.1
	
These packages can be installed with the following commands:

.. code-block:: bash

	pip install package_name=version

Where `package_name` is the name of the package and `version` is the specific
version you wish to install

With the new virtual Python environment set up, a number of additional packages 
have to be installed. These packages are GSL, MultiNest, PyMultiNest and Cython. 
Here GSL is the GNU Scientific Library and can be installed on a conda 
environment with the following command: 

.. code-block:: bash

	conda install -c conda-forge gsl

MultiNest can be installed in a manner similar to GSL, the required command 
to do so is as follows: 

.. code-block:: bash

	conda install -c conda-forge multinest

The same goes for PyMultiNest and Cython, both can be installed on a conda 
environment by entering the following commands: 

.. code-block:: bash

	conda install -c conda-forge pymultinest
	pip install Cython~=0.29

.. note::

	The latest versions of Cython (3.*) are not currently compatible with NEoST.

Testing if these packages have installed correctly can be done by entering 
the following command into a terminal: 

.. code-block:: bash

	python -c 'import NAME_OF_PACKAGE'

Here NAME_OF_PACKAGE is the name of the package you wish to test, e.g. 

.. code-block:: bash

	python -c 'import pymultinest'

to test the installation of PyMultiNest.

There is an extra Python packages that can be installed for full NEoST 
functionality, this can be installed by entering the following command 
into a terminal with the NEoST Python environment active:

.. code-block:: bash

	pip install kalepy

This package, KalePy, is an optional package containing a more feature-rich 
implementation of a KDE, it is based on the `SciPy.stats.gaussian_kde()` KDE 
and adds extra functionality which may at times be useful. This extra functionality 
does come at a cost however, it does take up more computational power than 
SciPy's own implementation.

Alternative instructions for prerequisites
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If the provided commands don't work, or if you are installing NEoST on a 
high-performance computing platform, there are alternative ways to install 
GSL, MultiNest and PyMultiNest. To manually install GSL enter the following
into a terminal: 

.. code-block:: bash

	wget -v http://mirror.koddos.net/gnu/gsl/gsl-latest.tar.gz
	tar -zxvf gsl-latest.tar.gz
	cd gsl-latest
	./configure CC=gcc --prefix=$HOME/gsl
	make
	make check
	make install
	make installcheck
	make clean

With this done GSL will have to be added to your path, this is done with the 
following command: 

.. code-block:: bash

	export PATH=$HOME/gsl/bin:$PATH

This command must be given any time GSL is used, therefore it is recommended 
to add this command to your ``~.bashrc`` file

In order to manually install MultiNest and PyMultinest, first install the 
prerequisites. These are mpi4py and compilers for c and fortran and can be 
installed with the following commands:

.. code-block:: bash

	conda install -c conda-forge mpi4py
	sudo apt-get install cmake libblad-dev liblapack-dev libatlas-base-dev

When these have finished installing, clone the MultiNest repository, navigate 
to the cloned repository and install MultiNest using the following commands:

.. code-block:: bash

	git clone https://github.com/farhanferoz/MultiNest.git <path/to/clone>/multinest
	cd <path/to/clone>/multinest/MultiNest_v3.12_CMake/multinest/
	mkdir build
	cd build
	CC=gcc FC=mpif90 CXX=g++ cmake -DCMAKE_{C,CXX}_FLAGS="-O3 -march=native -funroll-loops" -DCMAKE_Fortran_FLAGS="-O3 -march=native -funroll-loops" ..
	make
	ls ../lib/
    
This is the sequence of commands to install MultiNest, the final step now is 
to install the Python interface to MultiNest, PyMultiNest. For this, run the following commands:

.. code-block:: bash

	git clone https://github.com/JohannesBuchner/PyMultiNest.git <path/to/clone>/pymultinest
	cd <path/to/clone>/pymultinest
	python setup.py install [--user]

This will install the package in your NEoST environment if this is the active 
environment. If this is the case, the ``--user`` flag needs 
to be omitted. Next, PyMultiNest needs to be interfaced with multinest itself, 
this is done by using the following single-line command

.. code-block:: bash

	export LD_LIBRARY_PATH=/my/directory/MultiNest/lib/:$LD_LIBRARY_PATH

This command too needs to be given anytime you wish to use PyMultiNest and MultiNest together, 
so it is again recommended to add it to your ``~.bashrc`` file.

Installing NEoST
----------------

With the prerequisites out of the way, NEoST can now be installed. First 
open a new terminal in the directory where NEoST's source code was extracted 
and enter the following command:

.. code-block:: bash

	python build.py install

Alternatively, NEoST can also be installed without cythonizing the TOV solvers, this results 
in much slower performance however. To do this, enter the following command:

.. code-block:: bash

	conda install numba
	python build.py install --nocython

Documentation
-------------

If you wish to compile the documentation you require 
`Sphinx <http://www.sphinx-doc.org/en/master>`_ and extensions. To install
these, run the following commands:

.. code-block:: bash

    conda install sphinx
    conda install -c conda-forge nbsphinx
    conda install decorator
    conda install sphinxcontrib-websupport
    conda install sphinx_rtd_theme

Note, one can also perform these commands using ``pip`` instead of ``conda``. Now the documentation can be compiled using:

.. code-block:: bash

    cd NEoST-main/docs; [make clean;] make html

To rebuild the documentation after a change to source code docstrings:

.. code-block:: bash

    [CC=<path/to/compiler/executable>] python setup.py install [--user]; cd
    docs; make clean; make html; cd ..

The ``.html`` files can then found in ``NEoST-main/docs/build/html``, along with the
notebooks for the tutorials in this documentation. The ``.html`` files can
naturally be opened in a browser, handily via a Jupyter session (this is
particularly useful if the edits are to tutorial notebooks).

Note that if you require links to the source code in the HTML files, you need
to ensure Sphinx imports the ``NEoST`` package from the source directory
instead of from the ``~/.local/lib`` directory of the user. To enforce this,
insert the path to the source directory into ``sys.path`` in the ``conf.py``
script. Then make sure the extension modules are inside the source directory
-- i.e., the package is built in-place (see above).

.. note::

   To build the documentation, all modules need to be imported, and the
   dependencies that are not resolved will print warning messages.


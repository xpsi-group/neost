.. _install:

============
Installation
============

NEoST is a Python software package that performs neutron star equation of state inference process. It is an open source package that is available on `GitHub`_ and can be cloned as:

.. _GitHub: https://github.com/xpsi-group/neost.git

.. code-block:: bash

	git clone https://github.com/xpsi-group/neost.git

On this page we will describe how to install NEoST and its prerequisites.

Creating an environment and installing NEoST's dependencies
===========================================================

NEoST was developed in Python 3. We recommend installing NEoST and its dependencies in a virtual environment. There are two main options for creating an environment and installing the dependencies: Python's first-party package installer, pip, and the third-party Conda package manager. Using conda may be easier because while conda can install MultiNest, pip cannot. The downside is that you will have to install conda.

You need pip in both cases. It is likely already installed. If not, follow the official instructions for installing it.

Using pip
---------

Two of NEoST's dependencies, GSL (GNU Scientific Library) and MultiNest, are not written in Python and are as such not available in the Python package index (PyPI). They need to be installed separately.

GSL is widely available and can almost always be installed with your operating system's package manager, or loaded with something like ``module load gsl`` on a computing cluster.  Alternatively you can compile it yourself using the official instructions, see `Downloading GSL <https://www.gnu.org/software/gsl/#downloading>`_.

However you choose to obtain GSL, make sure it works by executing

.. code-block:: bash

	gsl-config --version

which should print the installed GSL version.

MultiNest, on the other hand, must likely be compiled manually. Follow these steps (which assume that you have ``git``, a ``C++`` compiler, ``cmake``, and ``make`` installed):

.. code-block:: bash

	git clone https://github.com/JohannesBuchner/MultiNest
	cd MultiNest/build
	cmake ..
	make

You then need to add ``<path/to>/MultiNest/lib`` to an environment variable called ``LD_LIBRARY_PATH`` using

.. code-block:: bash

	export LD_LIBRARY_PATH=<path/to>/MultiNest/lib:$LD_LIBRARY_PATH

Of course, exchange ``<path/to>`` for the actual path where you cloned MultiNest. You can add the export command to your ``$HOME/.bashrc`` file if you don't want to execute it in every new terminal in which you're running NEoST.

The next step is to create a virtual environment and install NEoST. In Python, the official way to create a virtual environment is by using the venv module:

.. code-block:: bash

	python -m venv path/to/environment

where you should substitute path/to/environment for a location of your choice. Then activate the environment with

.. code-block:: bash

	source path/to/environment/bin/activate

To install NEoST itself, see Installing NEoST below.

Using Conda
-----------

.. _basic_env:

Assuming a Conda base environment has been installed and activated (see Conda installation instructions, e.g., `Miniconda <https://docs.anaconda.com/miniconda/>`_), a new virtual environment can then be created by navigating to the NEoST base directory and executing

.. code-block:: bash

	conda env create -f environment.yml

This will create a new Conda environment called neost and install all dependencies (including GSL and MultiNest).  Once the environment has been created, activate it with

.. code-block:: bash

	conda activate neost

This changes the active virtual environment from the default base
environment to the new neost environment and needs to be entered any time
NEoST is used.

To install NEoST itself, see Installing NEoST below.

Installing on MAC M-series chips (arm-64)
-----------------------------------------

Errors may occur when installing MultiNest/PyMultiNest on MAC M-series chips as there are no arm-64 coda builds for `MultiNest <https://anaconda.org/conda-forge/multinest>`_. Although we do not have any definite solutions, we can offer a possible troubleshooting method that has worked in the past.

The first step is to remove PyMultiNest from the enviroment.yml file and re-run

.. code-block:: bash

	conda env create -f environment.yml

Then activate the conda enviroment via

.. code-block:: bash
	
	conda activate neost

Once that is complete, try to install PyMultiNest using pip:

.. code-block:: bash

	pip install pymultinest

If this works you may proceed to the Installing NEoST instructions as normal. However, if this fails, or works but installing MultiNest fails, we recomend using the following procedure from `NMMA <https://nuclear-multimessenger-astronomy.github.io/nmma/#for-arm64-macs>`_ to install PyMultiNest/MultiNest. You may also have a glance at the Alternative instructions for prerequisites for installing PyMultiNest/MultiNest as well.

Installing NEoST
================
With the prerequisites out of the way, NEoST can now be installed. First navigate to the NEoST base directory, if you haven't done so already, and install NEoST with

.. code-block:: bash

	make install

or, equivalently,

.. code-block:: bash

	pip install .

NEoST can optionally be installed without cythonizing the TOV solvers, at the expense of much slower performance. If you wish to do this, rename or delete the ``setup.py`` file before running ``make install``.  We only recommend using the Python TOV solvers if the cythonized solvers fail to compile or run.  Note that the unit tests in the ``tests/`` directory fail if the Python solvers are used; this is expected.




Building the documentation
==========================

Building the documentation is completely optional and not required for running NEoST.
If you do wish to compile the documentation locally you will require
`Sphinx <http://www.sphinx-doc.org/en/master>`_ and extensions.

If you have installed NEoST in a conda environment, you can install the documentation build dependencies using

.. code-block:: bash

	conda install sphinx nbsphinx decorator sphinxcontrib-websupport sphinx_rtd_theme pandoc

If you haven't used conda, you can install them using

.. code-block:: bash

	pip install sphinx nbsphinx decorator sphinxcontrib-websupport sphinx_rtd_theme

Unfortunately, the ``pandoc`` version available in pip does not seem to work, so you may have to install pandoc separately using, e.g., your system's package manager. See also `Pandoc <https://pandoc.org/installing.html>`_.

Once the dependencies are installed you can compile the documentation by navigating to the ``docs`` directory and executing

.. code-block:: bash

	make html

The ``.html`` files can then found in ``docs/build/html``, along with the
notebooks for the tutorials in this documentation. The ``.html`` files can
naturally be opened in a browser, handily via a Jupyter session (this is
particularly useful if the edits are to tutorial notebooks).

Alternative instructions for MultiNest
======================================

In case you cannot install MultiNest using either conda or the instructions for manual compilation given above, you could try these older alternative instructions. These have worked in the past, but we are not sure if they still work.

In order to manually install MultiNest and PyMultiNest, first install the prerequisites. These are mpi4py and compilers for C and Fortran, and can be installed with the following commands (assuming you are using a Debian-based distribution):

.. code-block:: bash

	conda install -c conda-forge mpi4py
	sudo apt-get install cmake libblad-dev liblapack-dev libatlas-base-dev

When these have finished installing, clone the MultiNest repository, navigate to the cloned repository and install MultiNest using the following commands:

.. code-block:: bash

	git clone https://github.com/farhanferoz/MultiNest.git <path/to/clone>/multinest
	cd <path/to/clone>/multinest/MultiNest_v3.12_CMake/multinest/
	mkdir build
	cd build
	CC=gcc FC=mpif90 CXX=g++ cmake -DCMAKE_{C,CXX}_FLAGS="-O3 -march=native -funroll-loops" -DCMAKE_Fortran_FLAGS="-O3 -march=native -funroll-loops" ..
	make
	ls ../lib/


The final step now is to install the Python interface to MultiNest, PyMultiNest. For this, simply install it using pip with

.. code-block:: bash

	pip install pymultinest

Alternatively you can clone its git repository and install manually:

.. code-block:: bash

	git clone https://github.com/JohannesBuchner/PyMultiNest.git 
	cd <path/to/clone>/pymultinest
	python setup.py install [--user]

This will install the package in your NEoST environment if this is the active environment. If this is the case, the ``--user`` flag needs to be omitted. Next, PyMultiNest needs to be interfaced with MultiNest itself, achieved by setting an environment variable as follows:

.. code-block:: bash

	export LD_LIBRARY_PATH=/my/directory/MultiNest/lib/:$LD_LIBRARY_PATH

This command too needs to be given anytime you wish to use PyMultiNest and MultiNest together, so it is again recommended to add it to your ``~.bashrc`` file.

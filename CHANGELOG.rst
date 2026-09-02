History
-------

All notable changes to this project will be documented in this file.

The format is based on
`Keep a Changelog <http://keepachangelog.com/en/1.0.0/>`_
and this project adheres to
`Semantic Versioning <http://semver.org/spec/v2.0.0.html>`_.

.. REMOVE THE DOTS BELOW TO UNCOMMENT
.. ..[Unreleased]
.. ~~~~~~~~~~~~

.. Summary
.. ^^^^^^^

.. Fixed
.. ^^^^^

.. Added
.. ^^^^^

.. Changed
.. ^^^^^^^

.. Deprecated
.. ^^^^^^^^^^

.. Removed
.. ^^^^^^^

.. Attribution
.. ^^^^^^^^^^^

[v3.0.0 - 2026-04-01]
~~~~~~~~~~~~~~~~~~~~~
Summary
^^^^^^^
Support for a neutron star with a possible dark energy core described by the Modified Chaplygin Dark Fluid (MCDF) model as described in Rutherford, Prescod-Weinstein, and Watts 2026 (in preperation).


Added
^^^^^
* A pythonic TOV solver for a neutron star with a possible MCDF core (TOVde_python.py). Note, there is a Cython version of this solver (TOVde.pyx), but it requires the step parameter in Star.py to be changed from 0.46 to 0.075 to get the same results as the python version.
This essentially implies that the Cython version is not stable for this application, and thus we have set the Star.py to default to the python version of the TOV solver when dark energy is included. 
That being said, the Cython version is still available for users who wish to experiment with it, but it is not recoommended for use at this time.    
* Included the two-fluid tidal-deformability equations for a neutron star with a possible MCDF core.
* Prior and posterior examples which include the possibility of a MCDF core (see Dark_energy_prior.py, Dark_energy_posterior.py, and DE_MR_Tidal_Tutorial.py).
* A simple tutorial showing how to compute the mass-radius and mass-tidal relations for a neutron star with a possible MCDF core in NEoST.
* MCDF functionality in base.py to compute the maximum central density of a neutron star with a MCDF core (find_max_edsc_de()).
* Updated Star.py to call the TOV solver with the possibility of a MCDF core, and to access the MCDF mass and radius if present. This also required some changes to how Star.py is initialized. In particular, the following parameters have been defined
  eps_plus = 0.0, alpha = 0.0, dark_energy = False, where eps_plus and alpha are the two parameters of the MCDF model, and dark_energy is a boolean that determines whether or not to include the possibility of a MCDF core in the TOV solver.
* MCDF EOS in polytropes.py, speedofsound.py, and tablualted.py.
* MCDF functionality in Likelihood.py with hard-cut offs included to eliminate non-physical parts of the MCDF EOS parameter space.
* MCDF functionality in PosteriorAnalysis.py, thus calling compute_table_data() and compute_auxiliary_data() have an additional boolean argument called "de", which, if True, includes the possibility of a MCDF core in the computed data.
* Central density sampling in Prior.py of a given source is sampled log-uniformly using the maximum central density of a neturon star with a MCDF core as the upper limit.
* Added more detailed descriptions of argument lists/parameter definitions/returns of various functions and parameters in the documentation (see TOVr_python.py, TOVde_python.py, TOVdm_python.py, TOVr.pyx, TOVde.pyx, TOVdm.pyx, and base.py).



Attribution
^^^^^^^^^^^
* NEoST core team

[v2.2.0 - 2025-07-31]
~~~~~~~~~~~~~~~~~~~~~
Summary
^^^^^^^
Support for multicore processing via MPI.

Added
^^^^^
* MPI support in PosteriorAnalysis.py, both for compute_auxiliary_data() and compute_table_data().
* Sampling the cEFT parameter from a standard normal distribution. Requires more work to actually be useful, this is a preparatory step
* Editable install with "make editable"
* Ultranest support for PosteriorAnalysis.py. Although sampling with Ultranest yields incorrect results, but at least the code runs.

Changed
^^^^^^^
* Removed duplicate (and thus error-prone) version specification in neost/__init__.py. This file now reads version and author from pyproject.toml. The version and author information is available via neost.__version__ and neost.__author__.
* Updated examples so that they work with the new PosteriorAnalysis.

Fixed
^^^^^
* A Numpy ragged-arrays issue
* Removed hard-coded limits for the cEFT parameter and replaced with the correct limits, read from the specification of the crust.

Removed
^^^^^^^
* PosteriorAnalysis.compute_prior_auxiliary_data()

Attribution
^^^^^^^^^^^
* NEoST core team


[v2.1.0 - 2024-12-18]
~~~~~~~~~~~~~~~~~~~~~
Summary
^^^^^^^
Changes from JOSS review process and to the publication list.

Added
^^^^^
* Notes for those attempting to compile and run on Mac M-series chips
* Details of Cython vs Python speed-up
* Updated the Rutherford, Prescod-Weinstein, and Watts 2024 Fermionic ADM paper to include publication information

Changed
^^^^^^^
* Small changes to JOSS paper
* Updated install.rst to use python -m pip install

Fixed
^^^^^
* rhoc overwritten in PosteriorAnalysis.compute_auxiliary_data(), which caused a power overflow error in star = Star(10**rhoc).

Attribution
^^^^^^^^^^^
* NEoST core team
* Axel Donath


[v2.0.0 - 2024-10-01]
~~~~~~~~~~~~~~~~~~~~~
Summary
^^^^^^^
Included the functionality of NEoST to allow for the possibility of fermionic or bosonic asymmetric dark matter (ADM) using the Nelson et al. 2018 ADM model.

Added
^^^^^
* Two-fluid TOV solver in both python and cython that allows for an additional ADM component in the GR stellar structure equations. (TOVdm.pyx and TOVdm_python.py)
* Included the two-fluid tidal-deformability equations.
* Prior and posterior examples which include the possibility of ADM
* A simple tutorial showing how to compute the ADM admixed neutron star mass-radius and mass-tidal relations in NEoST.
* ADM functionality in base.py with "fchi_calc" and "find_epsdm_cent" functions.
* Calling functions in Star.py to access ADM mass and radius
* ADM EOS in polytropes.py, speedofsound.py, and tablualted.py
* ADM functionality in Likelihood.py with hard-cut offs included to eliminate non-physical parts of the ADM EOS parameter space.
* ADM functionality in PosteriorAnalysis.py
* ADM sampling in Prior.py such that 'mchi' (ADM particle mass) and 'gchi_over_mphi' (effective ADM self-repulsion strength) are sampled log-uniformly.

Attribution
^^^^^^^^^^^
* NEoST core team

[v1.0.0 - 2024-09-11]
~~~~~~~~~~~~~~~~~~~~~~

Summary
^^^^^^^
Modernized installation, use standard python abbreviations "np" and "plt", updated JOSS paper, minor bug fixes

Fixed
^^^^^
* A numpy ragged-array issue in PosteriorAnalysis.compute_table_data()

Added
^^^^^
* Rutherford 2024 paper to publication list
* New main installation script: pyproject.toml
* Simple makefile that can install NEoST and also clean up generated files to simplify installation troubleshooting
* Reinstated tested instructions for compiling the documentation

Removed
^^^^^^^
* Unused imports and commented-out code

Changed
^^^^^^^
* The content and purpose of setup.py. This file is no longer the main installation script; its only purpose is to compile the Cython TOV solvers. To not compile these in case of issues, simply rename or delete setup.py.
* Installation instructions when not using conda

Attribution
^^^^^^^^^^^
* NEoST core team


[v0.10.0 - 2024-07-10]
~~~~~~~~~~~~~~~~~~~~~~

Summary
^^^^^^^
Updates to the code and documentation for the 2024 ApJL paper, compatibility improvements

Fixed
^^^^^
Compatibility issues:

* A couple of numpy "ragged arrays" problems, which numpy no longer supports. Two in neost/Likelihood.py, and one in the initial_conditions() function in neost/tovsolvers/TOVr_python.py. The ragged arrays were previously constructed similar to np.array([x1, [x2], x3, [x4]]) whereas in this version they are constructed like np.array([x1, x2, x3, x4]).
* Cython 3 compilation issue.
* Use "density" instead of "normed" in numpy.histogramdd.
* seaborn.kdeplot: Use "fill" instead of "shade", "levels" instead of "n_levels", "cmap" instead of "colors", and modify the supplied values accordingly
* Function name change: scipy.integrate.cumtrapz is now called scipy.integrate.cumulative_trapezoid. This was introduced in scipy 1.6 and the old name will become deprecated in scipy 1.14.
* Some minor plotting warnings

Added
^^^^^
* New Keller-N2LO and Keller-N3LO crusts
* neost.PosteriorAnalysis.compute_table_data() function which computes a number of quantities published in tables in the 2024 ApJL paper.
* neost.PosteriorAnalysis.compute_prior_auxiliary_data() function, which is a simplified and faster version of neost.PosteriorAnalysis.compute_auxiliary_data().
* A "likelihood function" for prior samplings, which checks that our imposed constraints are fulfilled. No actual data is used in this "likelihood" so our priors are still priors.
* Use scipy.interpolate.interp1d if scipy.interpolate.UnivariateSpline fails in neost.eos.polytropes

Removed
^^^^^^^
* Dependencies: getdist, alive_progress
* Functionality: the option to plot two distributions at the same time using neost.PosteriorAnalysis.mass_radius_prior_predictive_plot().

Changed
^^^^^^^
* Cython TOV solvers no longer print "using c code". Python TOV solvers, on the other hand, now raise a warning when they are in use.
* Renamed build.py to setup.py to enable installing with pip
* Updated core team membership

Deprecated
^^^^^^^^^^
* Scipy < 1.6 no longer supported due to the name change of scipy.integrate.cumtrapz.

Attribution
^^^^^^^^^^^
* NEoST core team

[v0.9.1] - 2023-09-20
~~~~~~~~~~~~~~~~~~~~~

Summary
^^^^^^^
Minor changes to all documentation and tutorials + submitted version of JOSS paper.

Added
^^^^^

* JOSS paper (submitted version)

Changed
^^^^^^^

* Updates to all tutorials and documentation.

Attribution
^^^^^^^^^^^

* NEoST core team

[v0.9.0] - 2023-09-07
~~~~~~~~~~~~~~~~~~~~~

Summary
^^^^^^^
First public release of repository.

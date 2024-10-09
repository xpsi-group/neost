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

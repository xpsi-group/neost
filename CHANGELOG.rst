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
Scipy < 1.6 no longer supported due to the name change of scipy.integrate.cumtrapz.

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

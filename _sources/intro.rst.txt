.. _intro:

.. image:: _static/neostbanner_large.png


*****************************************
Nested Equation of State Sampling (NEoST)
*****************************************

NEoST is an open-source code for dense matter equation of state (EoS) inference via
nested sampling.

It provides an inference framework that compares pre-existing EoS models (parameterized or tabulated, for both crust and core) to a variety of user-defined input data (real or synthetic), namely mass-radius samples, mass-tidal deformability samples, and mass samples. NEoST uses the
`Multinest <https://ui.adsabs.harvard.edu/abs/2009MNRAS.398.1601F/abstract>`_ package, a Bayesian inference code that computes the evidence for a chosen parameter space, to sample the posterior distributions of the EoS models or compute evidence in the case of a tabulated EoS.  NEoST can also be used as a fast solver for the Tolman-Oppenheimer-Volkoff (TOV) equations. 

.. note::

    What does the T in NEoST stand for, you may well ask?
    It is because the EoS is NesTed!

.. 





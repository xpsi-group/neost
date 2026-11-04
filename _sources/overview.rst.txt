.. _overview:

Overview
========

Matter in the cores of neutron stars can reach several times the nuclear saturation density. The `Equation of State (EoS) <https://ui.adsabs.harvard.edu/abs/2016PhR...621..127L/abstract>`_ of matter under such circumstances is not well understood:  in addition to extreme levels of `neutron-richness <https://ui.adsabs.harvard.edu/abs/2015ARNPS..65..457H/abstract>`_ there could also exist stable states of `strange matter <https://ui.adsabs.harvard.edu/abs/2020PrPNP.11203770T/abstract>`_, in the form of either hyperons or deconfined quarks. Neutron star properties like mass :math:`M`, radius :math:`R` and tidal deformability :math:`\Lambda` depend on the EoS, so measurement of these quantities provides insight. NEoST provides a framework for EoS inference for different types of astrophysical data, and a library of EoS models.   


Framework for Bayesian inference of the EOS
-------------------------------------------

Given EOS parameterizations, governed by parameters :math:`\boldsymbol{\theta}`, we employ Bayes' theorem and write the posterior distributions of the EOS parameters and central energy densities :math:`\boldsymbol{\varepsilon}` as

.. math::

	p(\boldsymbol{\theta}, \boldsymbol{\varepsilon} \,|\, \boldsymbol{d}, \mathbb{M})
	\propto p(\boldsymbol{\theta} \,|\, \mathbb{M})
	~p(\boldsymbol{\varepsilon} \,|\, \boldsymbol{\theta}, \mathbb{M})
	~p(\boldsymbol{d} \,|\, \boldsymbol{\theta}, \mathbb{M}) 

where :math:`\mathbb{M}` denotes the model including all assumed physics and :math:`\boldsymbol{d}` the astrophysical datasets used to constrain the EOS, consisting of, e.g., masses from radio data, masses and radii from NICER, and mass and tidal deformability from gravitational wave data.  Assuming each of these datasets to be independent of each other, we can separate the likelihoods and write 

.. math::

	p(\boldsymbol{\theta}, \boldsymbol{\varepsilon} \,|\, &\boldsymbol{d}, \mathbb{M})
	\propto 
	p(\boldsymbol{\theta} \,|\, \mathbb{M})
	~
	p(\boldsymbol{\varepsilon} \,|\, \boldsymbol{\theta}, \mathbb{M}) \\
	& \times \prod_{i} p(\Lambda_{1,i}, \Lambda_{2,i}, M_{1,i}, M_{2,i} \,|\, 
	\boldsymbol{d}_{\rm GW, i}) \\
	& \times \prod_{j} p(M_j, R_j \,|\, \boldsymbol{d}_{\rm NICER,j}) \\
	& \times \prod_{k} p(M_k \,|\, \boldsymbol{d}_{\rm radio,k}) 

Here the products run over the number of different observed stars, or mergers, in the case of the gravitational wave data.  Note that we have equated the nuisance-marginalized likelihoods to the nuisance-marginalized posterior distributions for the inferred masses, radii etc.  For a discussion of when this is justifiable, we refer the reader to Section 2 of `Raaijmakers et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...918L..29R/abstract>`_.

To speed up convergence we transform the gravitational wave posterior distributions to include the two tidal deformabilities, chirp mass :math:`\mathcal{M}_c = (M_1 M_2)^{3/5}/(M_1 + M_2)^{1/5}` and the mass ratio :math:`q` and re-weight the prior. Then we fix the chirp mass to its median value (see Section 2.2 of `Raaijmakers et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...893L..21R/abstract>`_ for more details). 


.. math::
	p(\boldsymbol{\theta}, \boldsymbol{\varepsilon} \,|\, &\boldsymbol{d}, \mathbb{M})
	\propto p(\boldsymbol{\theta} \,|\, \mathbb{M})
	~ p(\boldsymbol{\varepsilon} \,|\, \boldsymbol{\theta}, \mathbb{M}) \\
	& \times \prod_{i} p(\Lambda_{1,i}, \Lambda_{2,i}, q_i \,|\, \mathcal{M}_c, \boldsymbol{d}_{\rm GW, i}) \\
	& \times \prod_{j} p(M_j, R_j \,|\, \boldsymbol{d}_{\rm NICER,j}) \\
	& \times \prod_{k} p(M_k \,|\, \boldsymbol{d}_{\rm radio,k})

In order to account for the presence of dark matter within this inference procedure, we follow the framework outlined in `Rutherford et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023PhRvD.107j3051R/abstract>`_. The inference methods above sample over the central energy density. We instead sample over the ADM mass-fraction, the fraction of ADM mass inside the neutron star, i.e., we introduce :math:`F_{\chi} = F_{\chi}(\boldsymbol{\theta}, \boldsymbol{\epsilon_{c,B}}, \boldsymbol{\epsilon_{c,ADM}})` and write the above as 

.. math::
    p(\boldsymbol{\theta}, \boldsymbol{\epsilon_{c,B}}, \nonumber \boldsymbol{F_{\chi}} |\mathbf{d}) \propto ~ 
    & p(\boldsymbol{\theta}) p(\boldsymbol{\epsilon_{c, B}}|\boldsymbol{\theta}) p( \boldsymbol{F_{\chi}} |\boldsymbol{\theta}, \boldsymbol{\epsilon_c}) \\ 
    & \times \prod_{i} p(\Lambda_{1,i}, \Lambda_{2,i}, q_i \,|\, \mathcal{M}_c, \boldsymbol{d}_{\rm GW, i}) \\
    & \times \prod_{j} p(M_j, R_j \,|\, \boldsymbol{d}_{\rm NICER,j}) \\
    & \times \prod_{k} p(M_k \,|\, \boldsymbol{d}_{\rm radio,k}),

where :math:`\boldsymbol{\epsilon_{c,B}}` and :math:`\boldsymbol{\epsilon_{c,ADM}}` are the central energy densities of baryonic matter and ADM, respectively. We sample over the mass-fraction because our mass-radius algorithm is structured such that the dark matter energy density is dependent on the mass-fraction.



Astrophysical data 
-------------------------

The astrophysical datasets that NEoST uses come in the form of posterior distributions (e.g. for mass alone, mass-radius, or mass-tidal deformability) that are derived from separate inference analyses.  Mass posteriors come from pulsar timing analysis of pulsars in binary systems:  for an example see `Fonseca et al (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...915L..12F/abstract>`_.   Mass-radius posteriors from NICER data are generated via Pulse Profile Modelling, see e.g. `Riley et al. (2021) <https://ui.adsabs.harvard.edu/abs/2021ApJ...918L..27R/abstract>`_ for a discussion of the method, and the example file ``examples/J0740.npy`` in the NEoST Github repository.   Mass-tidal deformability posteriors are derived from gravitational wave data of neutron star binary mergers, see e.g. `Abbott et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019PhRvX...9a1001A/abstract>`_ for a discussion of the method, and the example file ``examples/GW170817.npy`` in the NEoST Github repository. 


Equation of state models
------------------------

NEoST uses the following general prescription for the EoS models: 

Crust models
^^^^^^^^^^^^

For densities below 0.5 :math:`n_s`, where :math:`n_s = 0.16 \mathrm{fm}^{-3}`,  NEoST uses the BPS EoS for the outer crust (`Baym, Pethick and Sutherland 1971  <https://ui.adsabs.harvard.edu/abs/1971ApJ...170..299B/abstract>`_). 

From this density until a core transition density (typically between 1.1 to 1.5 :math:`n_s`) NEoST assumes a polytrope with a varying normalization that captures the range of allowed EoS from chiral effective field theory (`cEFT  <https://ui.adsabs.harvard.edu/abs/2010PhRvC..82a4314H/abstract>`_). This intermediate part is matched to the parametrized models for core, at higher densities.  Various cEFT prescriptions are available:  `Hebeler et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013ApJ...773...11H/abstract>`_; `Tews et al. (2013) <https://ui.adsabs.harvard.edu/abs/2013PhRvL.110c2504T/abstract>`_; `Lynn et al. (2016) <https://ui.adsabs.harvard.edu/abs/2016PhRvL.116f2501L/abstract>`_; and `Drischler et al. (2019) <https://ui.adsabs.harvard.edu/abs/2019PhRvL.122d2501D/abstract>`_. In the most recent version of NEoST, we have also added `Keller et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023PhRvL.130g2701K/abstract>`_ :math:`N^{2}LO` and :math:`N^{3}LO` cEFT EOS, with calculated error bars up to 1.5 :math:`n_s`.  

Core models
^^^^^^^^^^^

Two parameterized core EoS models are provided:  a three-piece piecewise polytropic (PP) model with varying transition densities between the polytropes (as used in `Hebeler et al. 2013 <https://ui.adsabs.harvard.edu/abs/2013ApJ...773...11H/abstract>`_), and a speed of sound (CS) model based on physical considerations at both nuclear and high densities (`Greif, Raaijmakers et al. 2019 <https://ui.adsabs.harvard.edu/abs/2019MNRAS.485.5363G/abstract>`_).  

For the `PP model <https://ui.adsabs.harvard.edu/abs/2009PhRvD..79l4032R/abstract>`_ pressure is written as :math:`P_i\left(\rho\right)=\rho^\Gamma_i`. The free parameters are the core transition density, the two transition densities between the polytropic segments, and the three polytropic indices.   See :doc:`Piecewise Polytropic Example<PP_example>`.

For the CS model the speed of sound is taken to have the following functional form:

.. math::
    c_s^2/c^2(x) = a_1e^{-\frac{1}{2}(x-a_2)^2}+a_6+\frac{\frac{1}{3}-a_6}{1+e^{-a_5(x-a_4)}}

with :math:`x\equiv\varepsilon(m_nn_s)`, where :math:`m_n` is the neutron mass and :math:`n_s` the nuclear saturation density. The :math:`a_6` parameter is used to match between the crust and the core. Parameters :math:`a_1` through :math:`a_5` are free parameters which are limited by a number of constraints.  Pressure can be obtained via integrating over the density in the following manner

.. math::
    P(\varepsilon)=\int_0^\varepsilon d\varepsilon'c_s^2(\varepsilon')/c^2.
    
See :doc:`Speed of Sound Example<CS_example>`.

The user can alternatively choose to use a tabulated EoS model, in which case there are no free core EoS parameters.  See :doc:`Tabulated Example<Tabulated_example>`, which uses the `AP4 EoS model <https://ui.adsabs.harvard.edu/abs/1997PhRvC..56.2261A/abstract>`_.

Additionally, the user can enable the presence of bosonic/fermionic asymmetric dark matter (ADM) from `Nelson et al. (2018) <https://ui.adsabs.harvard.edu/abs/2019JCAP...07..012N/abstract>`_. The Nelson et al. (2018) ADM model consideres an MeV/GeV mass-scale complex scalar/spin-1/2 dirac spinor particle with repulsive self-interactions mediative by an eV/MeV mass-scale vector gauge boson. These models were considered in the inferences of `Rutherford et al. (2023) <https://ui.adsabs.harvard.edu/abs/2023PhRvD.107j3051R/abstract>`_, which also outlines how the Bayesian inference with ADM is modified to compared to one with only baryonic matter. See :doc:`Piecewise Polytropic with Bosonic ADM Example<PP_with_ADM_example>`.



Sampling
--------

.. image:: _static/NEOST_schematic.png 

NEoST samples from the prior distribution :math:`p(\boldsymbol{\theta} \,|\, \mathbb{M}) p(\boldsymbol{\varepsilon} \,|\, \boldsymbol{\theta}, \mathbb{M})`, computes the corresponding :math:`M`, :math:`R` and :math:`\Lambda`, and then evaluates the likelihood by applying a kernel density estimation (kde, see :doc:`Piecewise Polytropic Example<PP_example>` for more discussion of this aspect) to the posterior distributions of the astrophysical data sets using the nested sampling software `MultiNest <https://github.com/farhanferoz/MultiNest>`_. 

The prior distributions :math:`p(\boldsymbol{\theta} \,|\, \mathbb{M})` used for the EoS models must be set:  the default priors are as described in Section 2.3 of `Raaijmakers et al. (2020) <https://ui.adsabs.harvard.edu/abs/2020ApJ...893L..21R/abstract>`_.



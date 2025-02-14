---
title: 'NEoST: A Python package for nested sampling of the neutron star equation of state'
tags:
    - Python
    - astrostatistics
    - neutron stars
authors:
    - name: Geert Raaijmakers
      orcid: 0000-0002-9397-786X
      affiliation: 1
    - name: Nathan Rutherford
      orcid: 0000-0002-9626-7257
      affiliation: 2
    - name: Patrick Timmerman
      orcid: 0009-0003-2793-1569
      affiliation: 3
    - name: Tuomo Salmi
      orcid: 0000-0001-6356-125X
      affiliation: 3
    - name: Anna L. Watts
      orcid: 0000-0002-1009-2354
      affiliation: 3
    - name:  Chanda Prescod-Weinstein
      orcid: 0000-0002-6742-4532
      affiliation: 2
    - name:  Isak Svensson
      orcid: 0000-0002-9211-5555
      affiliation: "4, 5, 6"
    - name:  Melissa Mendes
      orcid: 0000-0002-5250-0723
      affiliation: "4, 5, 6"
affiliations:
   - name: GRAPPA, Anton Pannekoek Institute for Astronomy and Institute of High-Energy Physics, University of Amsterdam, Science Park 904, 1098 XH Amsterdam, Netherlands
     index: 1
   - name: Department of Physics and Astronomy, University of New Hampshire, Durham, New Hampshire 03824, USA
     index: 2
   - name: Anton Pannekoek Institute for Astronomy, University of Amsterdam, Science Park 904, 1098 XH Amsterdam, Netherlands
     index: 3
   - name: Technische Universität Darmstadt, Department of Physics, 64289 Darmstadt, Germany
     index: 4
   - name: ExtreMe Matter Institute EMMI, GSI Helmholtzzentrum für Schwerionenforschung GmbH, 64291 Darmstadt, Germany
     index: 5
   - name: Max-Planck-Institut für Kernphysik, Saupfercheckweg 1, 69117 Heidelberg, Germany
     index: 6

date: 5 September 2024
bibliography: neostjoss.bib
---


# Summary

The Nested Equation of State Sampling (NEoST) package is an open-source code that allows users to infer the parameters of the dense matter Equation of State (EoS) in neutron stars via nested sampling. It provides a Bayesian inference framework that compares pre-existing EoS models (parameterized or tabulated, for both crust and core) to a variety of user-defined astrophysical input data (real or synthetic), namely mass-radius samples, mass-tidal deformability samples, and mass samples. NEoST can also be used to provide a fast solver for the Tolman--Oppenheimer--Volkoff (TOV) equations for neutron star structure [@TOV1;@TOV2]. Moreover, NEoST is able to fully account for a possible dark matter component inside neutron stars, where the dark matter is described by the @Nelson:2018 model.

# Statement of need

Matter in the cores of neutron stars can reach several times the nuclear saturation density. The EoS of matter under such circumstances is not well understood: in addition to extreme levels of neutron-richness there could also exist stable states of strange matter, in the form of either hyperons or deconfined quarks [@Hebeler:2015;@Lattimer:2016;@Tolos:2020]. Neutron star properties like mass, radius and tidal deformability depend on the EoS, so measurement of these quantities provides insight into the properties of ultradense nuclear matter.

Astrophysical data sets that can be used to constrain the EoS take the form of posterior distributions that are derived from separate inference analyses.  Examples include: mass posteriors from pulsar timing analysis of radio pulsars in binary systems [@Fonseca:2021], joint mass-radius posteriors from Pulse Profile Modeling using NICER data [@Riley:2021;@Miller:2021]; and joint mass-tidal deformability posters from gravitational wave observations of neutron star binary mergers [@GW170817].   NEoST provides a framework for EoS inference that couples these various different types of astrophysical data to either parameterized or tabulated EoS models [e.g. @Keller:2023].

# The NEoST package and science use

NEoST is an open source Python package for Bayesian inference of EoS parameters (for parameterized  models) and/or evidence computation (for parameterized and tabulated models), given astrophysical data sets in the form of posterior distributions.  NEoST samples from the prior distribution of the EoS model parameters and central densities, computes the corresponding mass and radius/tidal deformability and then evaluates the likelihood by applying a kernel density estimation to the posterior distributions of the astrophysical data sets using the nested sampling software MultiNest [@MultiNest_2009; @PyMultiNest].  This workflow is shown in \autoref{fig:neost}.  The full Bayesian inference framework, including notes relating to prior distributions, is described in detail in @Raaijmakers:2020.  It includes a library of existing EoS models for crust and core, and users can easily define their own models.

NEoST also offers various options for post-processing, including generating plots showing the inferred EoS credible regions in pressure-energy density space, and the associated inferred mass-radius relation credible intervals.

![A schematic representation of the inference process using NEoST.
It shows how the track for physical measurements and
the track for theoretical models are fed through the framework, and what the main steps of analysis
are after inference is complete.\label{fig:neost}](fig1.png){width=100%}

NEoST is being used for EoS inference using mass-radius posteriors generated from pulse profile modeling of NICER data [@Raaijmakers:2019;@Raaijmakers:2020;@Raaijmakers:2021;@Rutherford:2024], specifically those generated using the X-PSI package [@Riley:2019;@Riley:2021;@Salmi:2022;@Riley:2023;@Salmi:2023;@Vinciguerra:2024;@Choudhury:2024;@Salmi:2024a;@Salmi:2024b].  It has also been used to study EoS prior sensitivities using synthetic mass-radius posteriors [@Greif:2019] and to study the consequences of a potential dark matter component in neutron stars [@Rutherford:2023;@Rutherford:2024b].

The core routines of NEoST are written in Cython
[@cython2011], and are dependent on the GNU Scientific Library [GSL,
@Gough:2009]. In case the user does not wish to use Cythonised code, there is also an alternative set of routines written purely in Python. High-level object-oriented model construction is performed by a
user in Python.

Release versions of NEoST are freely available on GitHub under the GNU General Public License.  Extensive documentation, step-by-step tutorials, and reproduction
code for existing data analyses, are available
via the GitHub repository, along with a suite of unit tests.  Future plans
include tutorials documenting different types of astrophysical data sets, EoS models that include a dark matter component, and options for coupling to different samplers.

*Software:* Python/C language [@Python2007], GNU Scientific Library [GSL,
@Gough:2009], NumPy [@Numpy2011], Cython [@cython2011], OpenMP [@openmp], MPI
for Python [@mpi4py], Matplotlib [@Hunter:2007; @matplotlibv2], IPython
[@IPython2007], Jupyter [@Kluyver:2016aa], MultiNest [@MultiNest_2009],
PyMultiNest [@PyMultiNest], GetDist [@Lewis19], SciPy [@Scipy], Seaborn [@Seaborn], corner.py [@corner], alive-progress [@aliveprogress].

# Acknowledgements

GR and ALW acknowledge support from ERC Starting Grant No. 639217 CSINEUTRONSTAR.  PT, TS and ALW acknowledge support from ERC Consolidator Grant No. 865768 AEONS (PI ALW).  GR acknowledges support from Nederlandse Organisatie voor Wetenschappelijk Onderzoek VIDI and Projectruimte grants (PI Samaya Nissanke).  NR and CPW are supported by NASA Grant No. 80NSSC22K0092 (PI Chanda Prescod-Weinstein).  IS and MM acknowledge support from ERC Advanced Grant No. 101020842 (PI Achim Schwenk).  More detailed acknowledgements are written in the project
documentation [hosted on GitHub](https://xpsi-group.github.io/neost/acknowledgements.html).

# References

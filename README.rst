.. _readme:


NEoST
=====

**An open-source code for dense matter equation
of state inference via nested sampling.**


NEoST is designed to infer constraints on the dense matter equation of state, 
based on cEFT models for the behaviour of dense matter in the crust of a 
neutron star and core parametrisations for the behaviour of dense matter in 
the core of a neutron star. NEoST allows users to choose from four different 
cEFT models and two different core parametrisations to construct an equation
of state. Users can then use Bayesian analysis techniques to combine this 
equation of state with measurements of a neutron star's mass and radius, 
or measurements of chirp mass in a neutron star merger gravitational wave event,
as well as the tidal deformabilities of the merging stars to obtain constraints
on the parameters of the studied equation of state.

It provides the following functionality:

* Comparison of dense matter physics models to astrophysical measurements.
* Easy-to-use equation of state framework to parametrise equations of state.
* Post-processing functionality to visualise NEoST's results.



For more details on current and planned capabilities, check out the 
`NEoST documentation <https://xpsi-group.github.io/neost/index.html>`_.

Installation and Testing
------------------------

NEoST is best installed from source. The documentation provides
`step-by-step installation instructions <https://xpsi-group.github.io/neost/install.html>`_
for Linux and for limited MacOS systems.

Documentation
-------------

The documentation for NEoST, including a number of tutorials, can be found at `https://xpsi-group.github.io/neost/ <https://xpsi-group.github.io/neost/>`_.

How to get in touch or get involved
-----------------------------------

We always welcome contributions and feedback! We are especially interested in 
hearing from you if
* something breaks,
* you spot bugs, 
* if there is missing functionality, or you have suggestions for future development,

To get in touch, please `open an issue <https://github.com/xpsi-group/neost/issues>`_.
Even better, if you have code you'd be interested in contributing, please send a 
`pull request <https://github.com/xpsi-group/neost/pulls>`_ (or get in touch 
and we'll help guide you through the process!). 

For more information, you can take a look at the documentation's 
`Contributing page <https://xpsi-group.github.io/neost/contributing.html>`_. 

Citing NEoST
-----------
If you find this package useful in your research, please provide the appropriate acknowledgment 
and citation. `Our documentation <https://xpsi-group.github.io/neost/citation.html>`_ provides 
more detail, including BibTeX entries and links to `appropriate papers <https://xpsi-group.github.io/neost/applications.html>`_.

Copyright and Licensing
-----------------------
All content © 2016-2024 the authors.
The code is distributed under the GNU General Public License v3.0; see `LICENSE <LICENSE>`_ for details.

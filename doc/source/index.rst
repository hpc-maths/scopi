.. scopi documentation master file, created by
   sphinx-quickstart on Tue Sep 13 13:01:16 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scopi's documentation!
=================================

Start at :doc:`api/solver` to understand the main steps of SCoPI's algorithm.
Then, read :doc:`api/problems/ProblemBase` to get the notations.

Solvers vs problems
===================

Not all solvers are implemented to solve all problems.
The table below summarizes the different combinations.

+-----------------------------------------+------------+-----------------+----------------------------+---------------------------------------------------+
|                                         | OptimMosek | OptimScs        |  OptimUzawa [#OptimUzawa]_ | OptimProjectedGradient [#OptimProjectedGradient]_ |
+=========================================+============+=================+============================+===================================================+
| DryWithoutFriction                      | YES        | YES             | YES                        | YES                                               |
+-----------------------------------------+------------+-----------------+----------------------------+---------------------------------------------------+
| DryWithFriction                         | YES        | NO [#Friction]_ | NO                         | NO [#Friction]_                                   |
+-----------------------------------------+------------+-----------------+----------------------------+---------------------------------------------------+
| DryWithFrictionFixedPoint               | YES        | NO [#Friction]_ | NO                         | NO [#Friction]_                                   |
+-----------------------------------------+------------+-----------------+----------------------------+---------------------------------------------------+
| ViscousWithoutFriction                  | YES        | YES             | YES                        | YES                                               |
+-----------------------------------------+------------+-----------------+----------------------------+---------------------------------------------------+
| ViscousWithFriction                     | YES        | NO [#Friction]_ | NO                         | NO [#Friction]_                                   |
+-----------------------------------------+------------+-----------------+----------------------------+---------------------------------------------------+
| ViscousWithFrictionFixedPoint           | YES        | NO [#Friction]_ | NO                         | NO [#Friction]_                                   |
+-----------------------------------------+------------+-----------------+----------------------------+---------------------------------------------------+
| ViscousWithFrictionFixedPointProjection | YES        | NO [#Friction]_ | NO                         | NO [#Friction]_                                   |
+-----------------------------------------+------------+-----------------+----------------------------+---------------------------------------------------+

.. [#OptimUzawa] OptimUzawaMkl, OptimUzawaMatrixFreeOmp, and OptimUzawaMatrixFreeTbb.
.. [#OptimProjectedGradient] Template parameter to choose PGD, APGD, APGD-AS, APGD-AR, or APDG-ASR solvers.
.. [#Friction] Could be implemented.

Contents
========
.. toctree::
   :maxdepth: 2

   api/property
   api/params
   api/types
   api/quaternion
   api/minpack
   api/container
   api/utils
   api/solver

Objects
-------
.. toctree::
   :maxdepth: 2

   api/objects/neighbor
   api/objects/types/sphere
   api/objects/types/superellipsoid
   api/objects/types/plane
   api/objects/types/worm

Problems
--------
.. toctree::
   :maxdepth: 2

   api/problems/ProblemBase
   api/problems/DryWithoutFriction
   api/problems/DryWithFriction
   api/problems/DryWithFrictionFixedPoint
   api/problems/ViscousBase
   api/problems/ViscousWithoutFriction
   api/problems/ViscousWithFriction

Solvers
-------
.. toctree::
   :maxdepth: 2

   api/solvers/OptimBase
   api/solvers/OptimMosek
   api/solvers/ConstraintMosek
   api/solvers/OptimProjectedGradient
   api/solvers/gradient/pgd
   api/solvers/gradient/apgd
   api/solvers/gradient/apgd_as
   api/solvers/gradient/apgd_ar
   api/solvers/gradient/apgd_asr
   api/solvers/projection
   api/solvers/OptimScs
   api/solvers/OptimUzawaBase
   api/solvers/OptimUzawaMatrixFreeOmp
   api/solvers/OptimUzawaMatrixFreeTbb
   api/solvers/OptimUzawaMkl

A priori velocities
-------------------
.. toctree::
   :maxdepth: 2

   api/vap/base
   api/vap/vap_fixed
   api/vap/vap_fpd
   api/vap/vap_projection

Contacts
--------
.. toctree::
   :maxdepth: 2

   api/contact/base
   api/contact/contact_brute_force
   api/contact/contact_kdtree

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

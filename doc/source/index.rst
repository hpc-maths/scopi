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

+-------------------------------+------------+----------+-------------------+------------------------------+
|                               | OptimMosek | OptimScs |  OptimUzawa [#0]_ | OptimProjectedGradient [#1]_ |
+===============================+============+==========+===================+==============================+
| DryWithoutFriction            | YES        | YES      | YES               | YES                          |
+-------------------------------+------------+----------+-------------------+------------------------------+
| DryWithFriction               | YES        + NO [#2]_ + NO                | NO [#2]_                     |      
+-------------------------------+------------+----------+-------------------+------------------------------+
| DryWithFrictionFixedPoint     | YES        + NO [#2]_ + NO                | NO [#2]_                     |      
+-------------------------------+------------+----------+-------------------+------------------------------+
| ViscousWithoutFriction        | YES        | YES      | YES               | YES                          |
+-------------------------------+------------+----------+-------------------+------------------------------+
| ViscousWithFriction           | YES        + NO [#2]_ + NO                | NO [#2]_                     |      
+-------------------------------+------------+----------+-------------------+------------------------------+
| ViscousWithFrictionFixedPoint | YES        + NO [#2]_ + NO                | NO [#2]_                     |      
+-------------------------------+------------+----------+-------------------+------------------------------+

.. [#0] OptimUzawaMkl, OptimUzawaMatrixFreeOmp, and OptimUzawaMatrixFreeTbb.
.. [#1] Template parameter to choose PGD, APGD, APGD-AS, APGD-AR, or APDG-ASR solvers.
.. [#2] Could be implemented.

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
   api/objects/types/plan
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
   api/solvers/gradient/uzawa
   api/solvers/gradient/nesterov
   api/solvers/gradient/nesterov_dynrho
   api/solvers/gradient/nesterov_restart
   api/solvers/gradient/nesterov_dynrho_restart
   api/solvers/gradient/projection_max
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

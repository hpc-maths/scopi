.. scopi documentation master file, created by
   sphinx-quickstart on Tue Sep 13 13:01:16 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to scopi's documentation!
=================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api/property
   api/params
   api/types

   api/objects/types/sphere
   api/objects/types/plan

   api/problems/ProblemBase
   api/problems/DryWithoutFriction
   api/problems/DryWithFriction
   api/problems/DryWithFrictionFixedPoint
   api/problems/ViscousBase
   api/problems/ViscousWithoutFriction
   api/problems/ViscousWithFriction

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

   api/vap/base
   api/vap/vap_fixed
   api/vap/vap_fpd
   api/vap/vap_projection

   api/contact/base
   api/contact/contact_brute_force
   api/contact/contact_kdtree

Test
====

Reference to :doc:`api/problems/ProblemBase`
Reference to :doc:`api/solvers/OptimScs`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

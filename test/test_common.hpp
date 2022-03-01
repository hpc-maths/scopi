#pragma once

#include <gtest/gtest.h>

#include <scopi/solver.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/solvers/OptimScs.hpp>
#include <scopi/solvers/OptimUzawaMkl.hpp>
#include <scopi/solvers/OptimUzawaMatrixFreeOmp.hpp>
#include <scopi/solvers/OptimUzawaMatrixFreeTbb.hpp>

#include <scopi/contact/contact_kdtree.hpp>
#include <scopi/contact/contact_brute_force.hpp>

#include <scopi/solvers/MatrixOptimSolverFriction.hpp>

#include <scopi/vap/vap_fixed.hpp>

namespace scopi
{
    #define SOLVER_WITH_CONTACT(dim, contact, vap) \
        ScopiSolver<dim, OptimMosek<>, contact, vap>, \
        ScopiSolver<dim, OptimScs, contact, vap>, \
        ScopiSolver<dim, OptimUzawaMkl, contact, vap>, \
        ScopiSolver<dim, OptimUzawaMatrixFreeTbb, contact, vap>, \
        ScopiSolver<dim, OptimUzawaMatrixFreeOmp, contact, vap>, \
        ScopiSolver<dim, OptimMosek<MatrixOptimSolverFriction>, contact, vap> // friction

    #define SOLVER_WITH_CONTACT_FRICTION(dim, contact, vap) \
        ScopiSolver<dim, OptimMosek<MatrixOptimSolverFriction>, contact, vap> // friction

    template<std::size_t dim, class vap_t = vap_fixed>
    using solver_types = ::testing::Types<
                            SOLVER_WITH_CONTACT(dim, contact_kdtree, vap_t)
                             >;

    template<std::size_t dim, class vap_t = vap_fixed>
    using solver_with_contact_types = ::testing::Types<
                            SOLVER_WITH_CONTACT(dim, contact_kdtree, vap_t),
                            SOLVER_WITH_CONTACT(dim, contact_brute_force, vap_t)
                             >;

    template<std::size_t dim>
    using solver_with_contact_types_friction = ::testing::Types<
                            SOLVER_WITH_CONTACT_FRICTION(dim, contact_kdtree),
                            SOLVER_WITH_CONTACT_FRICTION(dim, contact_brute_force)
                             >;
}

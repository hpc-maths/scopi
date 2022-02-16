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

namespace scopi
{
    #define SOLVER_WITH_CONTACT(dim, contact) \
        ScopiSolver<dim, OptimMosek<>, contact>, \
        ScopiSolver<dim, OptimScs, contact>, \
        ScopiSolver<dim, OptimUzawaMkl, contact>, \
        ScopiSolver<dim, OptimUzawaMatrixFreeTbb, contact>, \
        ScopiSolver<dim, OptimUzawaMatrixFreeOmp, contact>, \
        ScopiSolver<dim, OptimMosek<MatrixOptimSolverFriction>, contact> // friction

    #define SOLVER_WITH_CONTACT_FRICTION(dim, contact) \
        ScopiSolver<dim, OptimMosek<MatrixOptimSolverFriction>, contact> // friction

    template<std::size_t dim>
    using solver_types = ::testing::Types<
                            SOLVER_WITH_CONTACT(dim, contact_kdtree)
                             >;

    template<std::size_t dim>
    using solver_with_contact_types = ::testing::Types<
                            SOLVER_WITH_CONTACT(dim, contact_kdtree),
                            SOLVER_WITH_CONTACT(dim, contact_brute_force)
                             >;

    template<std::size_t dim>
    using solver_with_contact_types_friction = ::testing::Types<
                            SOLVER_WITH_CONTACT_FRICTION(dim, contact_kdtree),
                            SOLVER_WITH_CONTACT_FRICTION(dim, contact_brute_force)
                             >;
}

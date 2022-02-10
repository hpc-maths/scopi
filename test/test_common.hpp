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

namespace scopi
{
    // #define SOLVER_WITH_CONTACT(dim, contact) \
    //     ScopiSolver<dim, OptimMosek, contact>, \
    //     ScopiSolver<dim, OptimScs, contact>, \
    //     ScopiSolver<dim, OptimUzawaMkl, contact>, \
    //     ScopiSolver<dim, OptimUzawaMatrixFreeTbb, contact>, \
    //     ScopiSolver<dim, OptimUzawaMatrixFreeOmp, contact>

    #define SOLVER_WITH_CONTACT(dim, contact) \
        ScopiSolver<dim, OptimScs, contact>, \
        ScopiSolver<dim, OptimMosek, contact>, \
        ScopiSolver<dim, OptimUzawaMatrixFreeTbb, contact>, \
        ScopiSolver<dim, OptimUzawaMatrixFreeOmp, contact>

    template<std::size_t dim>
    using solver_types = ::testing::Types<
                            SOLVER_WITH_CONTACT(dim, contact_kdtree)
                             >;

    template<std::size_t dim>
    using solver_with_contact_types = ::testing::Types<
                            SOLVER_WITH_CONTACT(dim, contact_kdtree),
                            SOLVER_WITH_CONTACT(dim, contact_brute_force)
                             >;
}

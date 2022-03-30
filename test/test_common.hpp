#pragma once

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
        ScopiSolver<dim, OptimMosek<MatrixOptimSolverFriction>, contact, vap> // friction with mu = 0

    #define SOLVER_WITH_CONTACT_FRICTION(dim, contact, vap) \
        ScopiSolver<dim, OptimMosek<MatrixOptimSolverFriction>, contact, vap> // friction

}

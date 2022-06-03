#pragma once
#include "doctest/doctest.h"

#include <scopi/solver.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/solvers/OptimScs.hpp>
#ifdef SCOPI_USE_MKL
#include <scopi/solvers/OptimUzawaMkl.hpp>
#endif
#include <scopi/solvers/OptimUzawaMatrixFreeOmp.hpp>
#include <scopi/solvers/OptimUzawaMatrixFreeTbb.hpp>

#include <scopi/contact/contact_kdtree.hpp>
#include <scopi/contact/contact_brute_force.hpp>

#include <scopi/problems/DryWithoutFriction.hpp>
#include <scopi/problems/DryWithFriction.hpp>
#include <scopi/problems/ViscousWithoutFriction.hpp>
#include <scopi/problems/ViscousWithFriction.hpp>

#include <scopi/vap/vap_fixed.hpp>
#include <scopi/vap/vap_fpd.hpp>

namespace scopi
{
    template <class first, class second, class third>
    struct SolverWithParams
    {
        using SolverType = first;
        using OptimParamsType = second;
        using ProblemParamsType = third;
    };

#define SET_SOLVER_AND_PARAMS(solver, problem, dim, contact, vap) \
    SolverWithParams<ScopiSolver<dim, problem, solver, contact, vap>, OptimParams<solver>, ProblemParams<problem>>

#ifdef SCOPI_USE_MKL
#define SOLVER_DRY_WITHOUT_FRICTION(dim, contact, vap) \
    SET_SOLVER_AND_PARAMS(OptimMosek, DryWithoutFriction, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimScs, DryWithoutFriction, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMkl, DryWithoutFriction, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMatrixFreeTbb, DryWithoutFriction, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMatrixFreeOmp, DryWithoutFriction, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimMosek, DryWithFriction, dim, contact, vap) // friction with mu = 0
#else
#define SOLVER_DRY_WITHOUT_FRICTION(dim, contact, vap) \
    SET_SOLVER_AND_PARAMS(OptimMosek, DryWithoutFriction, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimScs, DryWithoutFriction, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMatrixFreeTbb, DryWithoutFriction, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMatrixFreeOmp, DryWithoutFriction, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimMosek, DryWithFriction, dim, contact, vap) // friction with mu = 0
#endif

#define SOLVER_DRY_WITH_FRICTION(dim, contact, vap) \
    SET_SOLVER_AND_PARAMS(OptimMosek, DryWithFriction, dim, contact, vap)

#ifdef SCOPI_USE_MKL
#define SOLVER_VISCOUS_WITHOUT_FRICTION(dim, contact, vap) \
    SET_SOLVER_AND_PARAMS(OptimMosek, ViscousWithoutFriction<dim>, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMkl, ViscousWithoutFriction<dim>, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMatrixFreeTbb, ViscousWithoutFriction<dim>, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMatrixFreeOmp, ViscousWithoutFriction<dim>, dim, contact, vap)
#else
#define SOLVER_VISCOUS_WITHOUT_FRICTION(dim, contact, vap) \
    SET_SOLVER_AND_PARAMS(OptimMosek, ViscousWithoutFriction<dim>, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMatrixFreeTbb, ViscousWithoutFriction<dim>, dim, contact, vap), \
    SET_SOLVER_AND_PARAMS(OptimUzawaMatrixFreeOmp, ViscousWithoutFriction<dim>, dim, contact, vap)
#endif

#define SOLVER_VISCOUS_WITH_FRICTION(dim, contact, vap) \
    SET_SOLVER_AND_PARAMS(OptimMosek, ViscousWithFriction<dim>, dim, contact, vap)

#define DOCTEST_VALUE_PARAMETERIZED_DATA(data, data_container) \
    static size_t _doctest_subcase_idx = 0; \
    std::for_each(data_container.begin(), data_container.end(), [&](const auto& in) {  \
            DOCTEST_SUBCASE((std::string(#data_container "[") + \
                        std::to_string(_doctest_subcase_idx++) + "]").c_str()) { data = in; } \
                        }); \
    _doctest_subcase_idx = 0;

}

// does not compile if this is inside the namespace
// TODO to be automated
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fpd>);

TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimScs, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimScs, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimScs, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimScs, scopi::contact_brute_force, scopi::vap_fpd>);

#ifdef SCOPI_USE_MKL
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMkl, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMkl, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMkl, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMkl, scopi::contact_brute_force, scopi::vap_fpd>);
#endif

TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMatrixFreeTbb, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMatrixFreeTbb, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMatrixFreeTbb, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMatrixFreeTbb, scopi::contact_brute_force, scopi::vap_fpd>);

TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMatrixFreeOmp, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMatrixFreeOmp, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMatrixFreeOmp, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithoutFriction, scopi::OptimUzawaMatrixFreeOmp, scopi::contact_brute_force, scopi::vap_fpd>);

TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithFriction, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithFriction, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithFriction, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::DryWithFriction, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fpd>);

TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fpd>);

#ifdef SCOPI_USE_MKL 
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMkl, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMkl, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMkl, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMkl, scopi::contact_brute_force, scopi::vap_fpd>);
#endif

TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMatrixFreeTbb, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMatrixFreeTbb, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMatrixFreeTbb, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMatrixFreeTbb, scopi::contact_brute_force, scopi::vap_fpd>);

TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMatrixFreeOmp, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMatrixFreeOmp, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMatrixFreeOmp, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithoutFriction<2>, scopi::OptimUzawaMatrixFreeOmp, scopi::contact_brute_force, scopi::vap_fpd>);

TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithFriction<2>, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithFriction<2>, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithFriction<2>, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithFriction<2>, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fpd>);

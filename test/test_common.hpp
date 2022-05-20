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

#include <scopi/problems/DryWithFriction.hpp>

#include <scopi/vap/vap_fixed.hpp>
#include <scopi/vap/vap_fpd.hpp>

namespace scopi
{
#ifdef SCOPI_USE_MKL
    #define SOLVER_WITH_CONTACT(dim, contact, vap) \
        ScopiSolver<dim, DryWithoutFriction, OptimMosek, contact, vap>, \
        ScopiSolver<dim, DryWithoutFriction, OptimScs, contact, vap>, \
        ScopiSolver<dim, DryWithoutFriction, OptimUzawaMkl, contact, vap>, \
        ScopiSolver<dim, DryWithoutFriction, OptimUzawaMatrixFreeTbb, contact, vap>, \
        ScopiSolver<dim, DryWithoutFriction, OptimUzawaMatrixFreeOmp, contact, vap>, \
        ScopiSolver<dim, DryWithFriction, OptimMosek, contact, vap> // friction with mu = 0
#else
    #define SOLVER_WITH_CONTACT(dim, contact, vap) \
        ScopiSolver<dim, DryWithoutFriction, OptimMosek, contact, vap>, \
        ScopiSolver<dim, DryWithoutFriction, OptimScs, contact, vap>, \
        ScopiSolver<dim, DryWithoutFriction, OptimUzawaMatrixFreeTbb, contact, vap>, \
        ScopiSolver<dim, DryWithoutFriction, OptimUzawaMatrixFreeOmp, contact, vap>, \
        ScopiSolver<dim, DryWithFriction, OptimMosek, contact, vap> // friction with mu = 0
#endif

    #define SOLVER_WITH_CONTACT_FRICTION(dim, contact, vap) \
        ScopiSolver<dim, DryWithFriction, OptimMosek, contact, vap>

    #define SOLVER_WITH_CONTACT_VISCOSITY(dim, contact, vap) \
        ScopiSolver<dim, ViscousWithFriction<dim>, OptimMosek, contact, vap>
                                                                              
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

TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithFriction<2>, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithFriction<2>, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fpd>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithFriction<2>, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fixed>);
TYPE_TO_STRING(scopi::ScopiSolver<2, scopi::ViscousWithFriction<2>, scopi::OptimMosek, scopi::contact_brute_force, scopi::vap_fpd>);

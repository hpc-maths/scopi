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
#ifdef SCOPI_USE_MKL
#define SOLVER_DRY_WITHOUT_FRICTION(dim, contact, vap) \
    ScopiSolver<dim, OptimMosek<DryWithoutFriction>, contact, vap>, \
    ScopiSolver<dim, OptimScs<DryWithoutFriction>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMkl<DryWithoutFriction>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMatrixFreeTbb<DryWithoutFriction>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMatrixFreeOmp<DryWithoutFriction>, contact, vap>, \
    ScopiSolver<dim, OptimMosek<DryWithFriction>, contact, vap> // friction with mu = 0
#else
#define SOLVER_DRY_WITHOUT_FRICTION(dim, contact, vap) \
    ScopiSolver<dim, OptimMosek<DryWithoutFriction>, contact, vap>, \
    ScopiSolver<dim, OptimScs<DryWithoutFriction>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMatrixFreeTbb<DryWithoutFriction>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMatrixFreeOmp<DryWithoutFriction>, contact, vap>, \
    ScopiSolver<dim, OptimMosek<DryWithFriction>, contact, vap> // friction with mu = 0
#endif

#ifdef SCOPI_USE_MKL
#define SOLVER_VISCOUS_GLOBULE(dim, contact, vap) \
    ScopiSolver<dim, OptimMosek<ViscousGlobule>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMkl<ViscousGlobule>, contact, vap>
#else
#define SOLVER_VISCOUS_GLOBULE(dim, contact, vap) \
    ScopiSolver<dim, OptimMosek<ViscousGlobule>, contact, vap>
#endif

#define SOLVER_DRY_WITH_FRICTION(dim, contact, vap) \
    ScopiSolver<dim, OptimMosek<DryWithFriction>, contact, vap> \

#ifdef SCOPI_USE_MKL
#define SOLVER_VISCOUS_WITHOUT_FRICTION(dim, contact, vap) \
    ScopiSolver<dim, OptimMosek<ViscousWithoutFriction<dim>>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMkl<ViscousWithoutFriction<dim>>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMatrixFreeTbb<ViscousWithoutFriction<dim>>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMatrixFreeOmp<ViscousWithoutFriction<dim>>, contact, vap>
#else
#define SOLVER_VISCOUS_WITHOUT_FRICTION(dim, contact, vap) \
    ScopiSolver<dim, OptimMosek<ViscousWithoutFriction<dim>>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMatrixFreeTbb<ViscousWithoutFriction<dim>>, contact, vap>, \
    ScopiSolver<dim, OptimUzawaMatrixFreeOmp<ViscousWithoutFriction<dim>>, contact, vap>
#endif

#define SOLVER_VISCOUS_WITH_FRICTION(dim, contact, vap) \
    ScopiSolver<dim, OptimMosek<ViscousWithFriction<dim>>, contact, vap>

#define DOCTEST_VALUE_PARAMETERIZED_DATA(data, data_container) \
    static size_t _doctest_subcase_idx = 0; \
    std::for_each(data_container.begin(), data_container.end(), [&](const auto& in) {  \
            DOCTEST_SUBCASE((std::string(#data_container "[") + \
                        std::to_string(_doctest_subcase_idx++) + "]").c_str()) { data = in; } \
                        }); \
    _doctest_subcase_idx = 0;

}

#define TYPE_TO_STRING_ONE_SOLVER(solver, problem, dim, contact, vap) \
    TYPE_TO_STRING(scopi::ScopiSolver<dim, scopi::solver<scopi::problem>, scopi::contact, scopi::vap>)
#define TYPE_TO_STRING_CONTACTS_VAP(solver, problem, dim)\
    TYPE_TO_STRING_ONE_SOLVER(solver, problem, dim, contact_kdtree, vap_fixed); \
    TYPE_TO_STRING_ONE_SOLVER(solver, problem, dim, contact_kdtree, vap_fpd); \
    TYPE_TO_STRING_ONE_SOLVER(solver, problem, dim, contact_brute_force, vap_fixed); \
    TYPE_TO_STRING_ONE_SOLVER(solver, problem, dim, contact_brute_force, vap_fpd);

TYPE_TO_STRING_CONTACTS_VAP(OptimMosek, DryWithoutFriction, 2)
TYPE_TO_STRING_CONTACTS_VAP(OptimScs, DryWithoutFriction, 2)
#ifdef SCOPI_USE_MKL
TYPE_TO_STRING_CONTACTS_VAP(OptimUzawaMkl, DryWithoutFriction, 2)
#endif
TYPE_TO_STRING_CONTACTS_VAP(OptimUzawaMatrixFreeTbb, DryWithoutFriction, 2)
TYPE_TO_STRING_CONTACTS_VAP(OptimUzawaMatrixFreeOmp, DryWithoutFriction, 2)

TYPE_TO_STRING_CONTACTS_VAP(OptimMosek, DryWithFriction, 2)

TYPE_TO_STRING_CONTACTS_VAP(OptimMosek, ViscousWithoutFriction<2>, 2)
#ifdef SCOPI_USE_MKL
TYPE_TO_STRING_CONTACTS_VAP(OptimUzawaMkl, ViscousWithoutFriction<2>, 2)
#endif
TYPE_TO_STRING_CONTACTS_VAP(OptimUzawaMatrixFreeTbb, ViscousWithoutFriction<2>, 2)
TYPE_TO_STRING_CONTACTS_VAP(OptimUzawaMatrixFreeOmp, ViscousWithoutFriction<2>, 2)

TYPE_TO_STRING_CONTACTS_VAP(OptimMosek, ViscousWithFriction<2>, 2)

TYPE_TO_STRING_CONTACTS_VAP(OptimMosek, ViscousGlobule, 2)
#ifdef SCOPI_USE_MKL
TYPE_TO_STRING_CONTACTS_VAP(OptimUzawaMkl, ViscousGlobule, 2)
#endif

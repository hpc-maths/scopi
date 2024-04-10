#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <xtensor/xtensor.hpp>

#include <scopi/types.hpp>

namespace scopi
{
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_velocity_no_friction(double alpha, double t, double r, double g, double y0);
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_no_friction(double alpha,  double t, double r, double g, double y0);

    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_velocity_viscous(double alpha, double t, double r, double g, double y0, double gamma_min, double t_inv);
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_viscous(double alpha,  double t, double r, double g, double y0, double gamma_min, double t_inv);

    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_friction(double alpha, double mu, double t, double r, double g, double y0);
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_velocity_friction(double alpha, double mu, double t, double r, double g, double y0);

    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_velocity_viscous_friction(double alpha, double mu, double t, double r, double g, double y0, double gamma_min, double t_inv);
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_viscous_friction(double alpha, double mu, double t, double r, double g, double y0, double gamma_min, double t_inv);
}

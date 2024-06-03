#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <xtensor/xtensor.hpp>

#include <scopi/types.hpp>

namespace scopi
{
    static constexpr double PI        = xt::numeric_constants<double>::PI;
    static constexpr double tolerance = 1e-6;
    bool diffFile(std::string filenameRef, std::string filenameResult, double tol = 1e-12);
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan(double alpha, double mu, double t, double r, double g, double y0);
    std::pair<type::position_t<2>, double>
    analytical_solution_sphere_plan_velocity(double alpha, double mu, double t, double r, double g, double y0);
}

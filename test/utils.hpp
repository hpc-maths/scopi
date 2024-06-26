#pragma once

#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <xtensor/xtensor.hpp>

namespace fs = std::filesystem;

#include <scopi/types.hpp>

namespace scopi
{
    static constexpr double PI        = xt::numeric_constants<double>::PI;
    static constexpr double tolerance = 1e-6;
    bool diffFile(const fs::path filenameResult, const std::filesystem::path filenameRef, double tol = 1e-12);
    bool check_reference_file(const std::filesystem::path path, const std::string_view filename, std::size_t it, double tolerance = 1e-12);
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plane(double alpha, double mu, double t, double r, double g, double y0);
    std::pair<type::position_t<2>, double>
    analytical_solution_sphere_plane_velocity(double alpha, double mu, double t, double r, double g, double y0);
}

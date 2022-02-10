#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <xtensor/xtensor.hpp>

namespace scopi
{
    static constexpr double PI = xt::numeric_constants<double>::PI;
    static constexpr double tolerance = 1e-6;
    bool diffFile(std::string filenameRef, std::string filenameResult, double tol=1e-12);
}

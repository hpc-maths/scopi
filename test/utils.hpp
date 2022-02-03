#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <xtensor/xtensor.hpp>

namespace scopi
{
    static constexpr double PI = xt::numeric_constants<double>::PI;

    bool diffFile(std::string filenameRef, std::string filenameResult);
}

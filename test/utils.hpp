#pragma once

#include <fstream>
#include <nlohmann/json.hpp>
#include <xtensor/xtensor.hpp>

namespace scopi
{
    constexpr double PI = xt::numeric_constants<double>::PI;

    bool diffFile(std::string filenameRef, std::string filenameResult)
    {
        std::ifstream fileRef(filenameRef);
        std::ifstream fileResult(filenameResult);
        if(fileRef && fileResult)
        {
            nlohmann::json jsonRef = nlohmann::json::parse(fileRef);
            nlohmann::json jsonResult = nlohmann::json::parse(fileResult);
            nlohmann::json diff = nlohmann::json::diff(jsonRef, jsonResult);
            return diff.empty();
        }
        else
        {
            return false;
        }
    }

}

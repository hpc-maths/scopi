#include "utils.hpp"

namespace scopi
{
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

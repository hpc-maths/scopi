#include "utils.hpp"

namespace scopi
{
    bool diffFile(std::string filenameRef, std::string filenameResult, double tol)
    {
        std::ifstream fileRef(filenameRef);
        std::ifstream fileResult(filenameResult);
        if(fileRef && fileResult)
        {
            nlohmann::json jsonRef = nlohmann::json::parse(fileRef);
            nlohmann::json jsonResult = nlohmann::json::parse(fileResult);
            nlohmann::json diff = nlohmann::json::diff(jsonRef, jsonResult);
            if (!diff.empty())
            {
                for(auto& p: diff)
                {
                    std::string path_ = p["path"];
                    nlohmann::json::json_pointer path(path_);
                    if (jsonRef[path].is_number_float() && jsonResult[path].is_number_float())
                    {
                        if (std::abs(static_cast<double>(jsonRef[path]) - static_cast<double>(jsonResult[path])) > tol)
                        {
                            return false;
                        }
                    }
                    else
                    {
                        return false;
                    }
                }
            }
            return true;
        }
        else
        {
            return false;
        }
    }

}

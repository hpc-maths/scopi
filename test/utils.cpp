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
            nlohmann::json diff = nlohmann::json::diff(jsonRef["objects"], jsonResult["objects"]);
            if (!diff.empty())
            {
                for(auto& p: diff)
                {
                    std::string path_ = p["path"];
                    nlohmann::json::json_pointer path(path_);
                    if (jsonRef["objects"][path].is_number_float() && jsonResult["objects"][path].is_number_float())
                    {
                        if (std::abs(static_cast<double>(jsonRef["objects"][path]) - static_cast<double>(jsonResult["objects"][path])) > tol)
                        {
                            std::cerr << path << " " << jsonRef["objects"][path] << " " << jsonResult["objects"][path] << std::endl;
                            return false;
                        }
                    }
                    else
                    {
                        std::cerr << path << std::endl;
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

    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan(double alpha, double mu, double t, double r, double g, double y0)
    {
        double x_normal, theta;
        double t_impact = std::sqrt(2*(y0-r)/(g*std::cos(alpha)));
        type::position_t<2> x;
        if (t > t_impact)
        {
            double v_t_m = g*t_impact*std::sin(alpha);
            double v_n_m = -g*t_impact*std::cos(alpha);
            double t2 = (t - t_impact );
            double x_impact = g*std::sin(alpha)*t_impact*t_impact/2.;
            if(std::tan(alpha) <= 3*mu)
            {
                x_normal = g*std::sin(alpha)*t2*t2/3. + 2.*v_t_m*t2/3. + x_impact;
                theta = -g*std::sin(alpha)*t2*t2/(3.*r) - 2*v_t_m*t2/(3.*r);
            }
            else
            {
                x_normal = g*(std::sin(alpha) - mu*std::cos(alpha))*t2*t2/2. + (v_t_m + mu*v_n_m)*t2 + x_impact;
                theta = -mu*g*std::cos(alpha)*t2*t2/r + 2*mu*v_n_m*t2/r;
            }
            x(0) =  x_normal*std::cos(alpha) + r*std::sin(alpha);
            x(1) = -x_normal*std::sin(alpha) + r*std::cos(alpha);
            return std::make_pair(x, theta);
        }
        else
        {
            x(0) = y0*std::sin(alpha);
            x(1) = y0*std::cos(alpha) - g*t*t/2.;
            return std::make_pair(x, 0.);
        }
    }

    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_velocity(double alpha, double mu, double t, double r, double g, double y0)
    {
        double v_normal, omega;
        double t_impact = std::sqrt(2*(y0-r)/(g*std::cos(alpha)));
        type::position_t<2> x;
        if (t > t_impact)
        {
            double v_t_m = g*t_impact*std::sin(alpha);
            double v_n_m = -g*t_impact*std::cos(alpha);
            double t2 = (t - t_impact );
            if(std::tan(alpha) <= 3*mu)
            {
                v_normal = 2.*g*std::sin(alpha)*t2/3. + 2.*v_t_m/3.;
                omega = -2.*g*std::sin(alpha)*t2/(3.*r) - 2*v_t_m/(3.*r);
            }
            else
            {
                v_normal = g*(std::sin(alpha) - mu*std::cos(alpha))*t2 + (v_t_m + mu*v_n_m);
                omega = -2.*mu*g*std::cos(alpha)*t2/r + 2*mu*v_n_m/r;
            }
            x[0] = v_normal*std::cos(alpha);
            x[1] = -v_normal*std::sin(alpha);
            return std::make_pair(x, omega);
        }
        else
        {
            x[0] = 0.;
            x[1] = -g*t;
            return std::make_pair(x, 0.);
        }
    }
}

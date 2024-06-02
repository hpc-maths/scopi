#include "scopi/solvers/OptimUzawaBase.hpp"

namespace scopi
{
    OptimParamsUzawaBase::OptimParamsUzawaBase(const OptimParamsUzawaBase& params)
        : tol(params.tol)
        , max_iter(params.max_iter)
        , rho(params.rho)
    {
    }

    OptimParamsUzawaBase::OptimParamsUzawaBase()
        : tol(1e-9)
        , max_iter(40000)
        , rho(2000.)
    {
    }

    void OptimParamsUzawaBase::init_options(CLI::App& app)
    {
        auto opt = app.add_option_group("Uzawa options");
        opt->add_option("--tol", tol, "Maximum distance between two neighboring particles")->capture_default_str();
        opt->add_option("--max-ite", max_iter, "Maximal number of iterations")->capture_default_str();
        opt->add_option("--rho", rho, "Step for the gradient descent")->capture_default_str();
    }
}

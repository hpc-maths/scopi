#pragma once

#include <CLI/CLI.hpp>

#include <plog/Log.h>

#include <xtensor/xnoalias.hpp>
#include <xtensor/xnorm.hpp>
#include <xtensor/xtensor.hpp>

namespace scopi
{

    template <class Problem, class Contacts, class Particles>
    class minimization_problem;

    struct pgd_params
    {
        void init_options(CLI::App& app)
        {
            auto opt = app.add_option_group("PGD options");
            opt->add_option("--pgd-alpha", alpha, "descent coefficient")->capture_default_str();
            opt->add_option("--pgd-max-ite", max_ite, "Maximum number of iterations")->capture_default_str();
            opt->add_option("--pgd-tolerance", tolerance, "Tolerance")->capture_default_str();
        }

        double alpha     = 0.05;
        double max_ite   = 10000;
        double tolerance = 1e-6;
    };

    class pgd
    {
      public:

        using params_t = pgd_params;

        pgd(const params_t& params = params_t())
            : m_params(params)
        {
        }

        void init_options(CLI::App& app)
        {
            m_params.init_options(app);
        }

        params_t& get_params()
        {
            return m_params;
        }

        template <class Problem, class Contacts, class Particles>
        auto operator()(const minimization_problem<Problem, Contacts, Particles>& min_p)
        {
            std::size_t ite = 0;

            xt::xtensor<double, 1> lambda_n   = xt::zeros<double>({min_p.size()});
            xt::xtensor<double, 1> lambda_np1 = xt::zeros<double>({min_p.size()});

            while (ite < m_params.max_ite)
            {
                ++ite;

                xt::xtensor<double, 1> dG = min_p.gradient(lambda_n);
                lambda_np1                = lambda_n - m_params.alpha * dG;
                min_p.projection(lambda_np1);

                // PLOG_INFO << fmt::format("pgd -> ite: {} residual: {}", ite, xt::norm_l2(lambda_np1 - lambda_n)[0]) << std::endl;
                // PLOG_INFO << fmt::format("alpha: {}", m_params.alpha) << std::endl;
                // PLOG_INFO << fmt::format("dG: {}", xt::norm_linf(dG)) << std::endl;
                // PLOG_INFO << fmt::format("lambda_n: {}", xt::norm_linf(lambda_np1)) << std::endl;

                if (xt::norm_l2(lambda_np1 - lambda_n)[0] < m_params.tolerance)
                // if (xt::norm_linf(dG)[0] < m_params.tolerance || xt::norm_linf(lambda_np1)[0] < m_params.tolerance)
                {
                    std::swap(lambda_n, lambda_np1);
                    break;
                }

                std::swap(lambda_n, lambda_np1);
            }
            PLOG_INFO << fmt::format("pgd converged in {} iterations.", ite) << std::endl;
            return lambda_n;
        }

      private:

        params_t m_params;
    };

    struct apgd_params
    {
        void init_options(CLI::App& app)
        {
            auto opt = app.add_option_group("APGD options");
            opt->add_option("--apgd-alpha", alpha, "descent coefficient")->capture_default_str();
            opt->add_option("--apgd-max-ite", max_ite, "Maximum number of iterations")->capture_default_str();
            opt->add_option("--apgd-tolerance", tolerance, "Tolerance")->capture_default_str();
            opt->add_flag("--apgd-dynamic", dynamic_descent, "Adaptive descent coefficient")->capture_default_str();
        }

        double alpha         = 0.05;
        double max_ite       = 10000;
        double tolerance     = 1e-6;
        bool dynamic_descent = false;
    };

    class apgd
    {
      public:

        using params_t = apgd_params;

        apgd(const params_t& params = params_t())
            : m_params(params)
        {
        }

        void init_options(CLI::App& app)
        {
            m_params.init_options(app);
        }

        params_t& get_params()
        {
            return m_params;
        }

        template <class Problem, class Contacts, class Particles>
        auto operator()(const minimization_problem<Problem, Contacts, Particles>& min_p)
        {
            std::size_t ite = 0;

            xt::xtensor<double, 1> lambda_n   = xt::zeros<double>({min_p.size()});
            xt::xtensor<double, 1> lambda_np1 = xt::zeros<double>({min_p.size()});

            xt::xtensor<double, 1> theta_n   = xt::ones<double>({min_p.size()});
            xt::xtensor<double, 1> theta_np1 = xt::ones<double>({min_p.size()});

            xt::xtensor<double, 1> y_n   = xt::zeros<double>({min_p.size()});
            xt::xtensor<double, 1> y_np1 = xt::zeros<double>({min_p.size()});

            double alpha  = m_params.alpha;
            double lipsch = 1. / alpha; // used only if dynamic_descent = true

            while (ite < m_params.max_ite)
            {
                ++ite;

                xt::xtensor<double, 1> dG = min_p.gradient(y_n);
                xt::noalias(lambda_np1)   = y_n - alpha * dG;
                min_p.projection(lambda_np1);

                if (m_params.dynamic_descent)
                {
                    while (min_p(lambda_np1)
                           >= min_p(y_n) + xt::linalg::dot(dG, lambda_np1 - y_n)[0] + 0.5 * lipsch * xt::norm_l2(lambda_np1 - y_n)[0])
                    {
                        lipsch *= 2;
                        alpha                   = 1. / lipsch;
                        xt::noalias(lambda_np1) = y_n - alpha * dG;
                        min_p.projection(lambda_np1);
                    }
                }

                // PLOG_INFO << fmt::format("apgd -> ite: {} residual: {}", ite, xt::norm_l2(lambda_np1 - lambda_n)[0]) << std::endl;
                // PLOG_INFO << fmt::format("alpha: {}", m_params.alpha) << std::endl;
                // PLOG_INFO << fmt::format("dG: {}", xt::norm_linf(dG)) << std::endl;
                // PLOG_INFO << fmt::format("lambda_n: {}", xt::norm_linf(lambda_np1)) << std::endl;

                if (xt::norm_l2(lambda_np1 - lambda_n)[0] < m_params.tolerance)
                // if (xt::norm_linf(dG)[0] < m_params.tolerance || xt::norm_linf(lambda_np1)[0] < m_params.tolerance)
                {
                    std::swap(lambda_n, lambda_np1);
                    break;
                }

                xt::noalias(theta_np1) = 0.5 * (theta_n * xt::sqrt(4 + theta_n * theta_n) - theta_n * theta_n);
                auto beta              = theta_n * (1 - theta_n) / (theta_n * theta_n + theta_np1);
                xt::noalias(y_np1)     = lambda_np1 + beta * (lambda_np1 - lambda_n);

                if (m_params.dynamic_descent)
                {
                    if (xt::linalg::dot(dG, lambda_np1 - lambda_n)[0] > 0)
                    {
                        y_np1 = lambda_np1;
                        theta_np1.fill(1.);
                    }
                    lipsch *= 0.97;
                    alpha = 1. / lipsch;
                }
                std::swap(lambda_n, lambda_np1);
                std::swap(theta_n, theta_np1);
                std::swap(y_n, y_np1);
            }
            PLOG_INFO << fmt::format("apgd converged in {} iterations.", ite) << std::endl;
            return lambda_n;
        }

      private:

        params_t m_params;
    };

}
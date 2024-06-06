#include "analytical_solution.hpp"

namespace scopi
{
    /// Dry without friction
    std::pair<type::position_t<2>, double>
    analytical_solution_sphere_plan_velocity_no_friction(double alpha, double t, double r, double g, double y0)
    {
        const double t_impact = std::sqrt(2 * (y0 - r) / (g * std::cos(alpha)));
        type::position_t<2> v;
        if (t < t_impact)
        {
            v[1] = -g * t * std::cos(alpha);
            v[0] = g * t * std::sin(alpha);
            return std::make_pair(v, 0.);
        }

        v[1] = 0.;
        v[0] = g * t * std::sin(alpha);
        return std::make_pair(v, 0.);
    }

    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_no_friction(double alpha, double t, double r, double g, double y0)
    {
        const double t_impact = std::sqrt(2 * (y0 - r) / (g * std::cos(alpha)));
        type::position_t<2> x;
        if (t < t_impact)
        {
            x[1] = -g * std::pow(t, 2) * std::cos(alpha) / 2 + y0;
            x[0] = g * std::pow(t, 2) * std::sin(alpha) / 2;
            return std::make_pair(x, 0.);
        }

        x[1] = r;
        x[0] = g * std::pow(t, 2) * std::sin(alpha) / 2;
        return std::make_pair(x, 0.);
    }

    /// Visous without friction
    std::pair<type::position_t<2>, double>
    analytical_solution_sphere_plan_velocity_viscous(double alpha, double t, double r, double g, double y0, double gamma_min, double t_inv)
    {
        type::position_t<2> v;
        const double t_impact    = std::sqrt(2 * (y0 - r) / (g * std::cos(alpha)));
        const double t_gamma_min = -gamma_min / (g * std::cos(alpha));
        double gamma_inv;
        if (t_inv > t_gamma_min)
        {
            gamma_inv = -gamma_min;
        }
        else
        {
            gamma_inv = -g * t_inv * std::cos(alpha);
        }
        const double t_unstick = t_inv + gamma_inv / (g * std::cos(alpha));
        if (t < t_impact)
        {
            v[1] = -g * t * std::cos(alpha);
            v[0] = g * t * std::sin(alpha);
            return std::make_pair(v, 0.);
        }

        if (t < t_inv)
        {
            v[1] = 0.;
            v[0] = g * t * std::sin(alpha);
            return std::make_pair(v, 0.);
        }

        if (t < t_unstick)
        {
            v[1] = 0.;
            v[0] = g * t_inv * std::sin(alpha) - g * (t - t_inv) * std::sin(alpha);
            return std::make_pair(v, 0.);
        }

        v[1] = g * (t - t_unstick) * std::cos(alpha);
        v[0] = g * t_inv * std::sin(alpha) - g * (t - t_inv) * std::sin(alpha);
        return std::make_pair(v, 0.);
    }

    std::pair<type::position_t<2>, double>
    analytical_solution_sphere_plan_viscous(double alpha, double t, double r, double g, double y0, double gamma_min, double t_inv)
    {
        type::position_t<2> x;
        const double t_impact    = std::sqrt(2 * (y0 - r) / (g * std::cos(alpha)));
        const double t_gamma_min = -gamma_min / (g * std::cos(alpha));
        double gamma_inv;

        if (t_inv > t_gamma_min)
        {
            gamma_inv = gamma_min;
        }
        else
        {
            gamma_inv = -g * t_inv * std::cos(alpha);
        }

        const double t_unstick = t_inv - gamma_inv / (g * std::cos(alpha));

        if (t < t_impact)
        {
            x[1] = -g * std::pow(t, 2) * std::cos(alpha) / 2 + y0;
            x[0] = g * std::pow(t, 2) * std::sin(alpha) / 2;
            return std::make_pair(x, 0.);
        }

        if (t < t_inv)
        {
            x[1] = r;
            x[0] = g * std::pow(t, 2) * std::sin(alpha) / 2;
            return std::make_pair(x, 0.);
        }

        if (t < t_unstick)
        {
            x[1] = r;
            x[0] = g * std::pow(t_inv, 2) * std::sin(alpha) / 2 + g * t_inv * std::sin(alpha) * (t - t_inv)
                 - g * pow((t - t_inv), 2) * std::sin(alpha) / 2;
            return std::make_pair(x, 0.);
        }

        x[1] = r + g * pow((t - t_unstick), 2) * std::cos(alpha) / 2;
        x[0] = g * std::pow(t_inv, 2) * std::sin(alpha) / 2 + g * t_inv * std::sin(alpha) * (t - t_inv)
             - g * pow((t - t_inv), 2) * std::sin(alpha) / 2;
        return std::make_pair(x, 0.);
    }

    /// Friction
    std::pair<type::position_t<2>, double>
    analytical_solution_sphere_plan_friction(double alpha, double mu, double t, double r, double g, double y0)
    {
        const double t_impact = std::sqrt(2 * (y0 - r) / (g * std::cos(alpha)));
        type::position_t<2> x;
        if (t > t_impact)
        {
            double x_tangent;
            double theta;
            const double v_t_m     = g * t_impact * std::sin(alpha);
            const double v_n_m     = -g * t_impact * std::cos(alpha);
            const double t2        = (t - t_impact);
            const double xt_impact = g * std::pow(t_impact, 2) * std::sin(alpha) / 2;
            if (std::tan(alpha) <= 3 * mu)
            {
                x_tangent = g * std::sin(alpha) * t2 * t2 / 3. + 2. * v_t_m * t2 / 3. + xt_impact;
                theta     = -g * std::sin(alpha) * t2 * t2 / (3. * r) - 2 * v_t_m * t2 / (3. * r);
            }
            else
            {
                x_tangent = g * (std::sin(alpha) - mu * std::cos(alpha)) * t2 * t2 / 2. + (v_t_m + mu * v_n_m) * t2 + xt_impact;
                theta     = -mu * g * std::cos(alpha) * t2 * t2 / r + 2 * mu * v_n_m * t2 / r;
            }
            x[1] = r;
            x[0] = x_tangent;
            return std::make_pair(x, theta);
        }
        x[1] = -g * std::pow(t, 2) * std::cos(alpha) / 2 + y0;
        x[0] = g * std::pow(t, 2) * std::sin(alpha) / 2;
        return std::make_pair(x, 0.);
    }

    std::pair<type::position_t<2>, double>
    analytical_solution_sphere_plan_velocity_friction(double alpha, double mu, double t, double r, double g, double y0)
    {
        const double t_impact = std::sqrt(2 * (y0 - r) / (g * std::cos(alpha)));
        type::position_t<2> v;
        if (t > t_impact)
        {
            double v_tangent;
            double omega;
            const double v_t_m = g * t_impact * std::sin(alpha);
            const double v_n_m = -g * t_impact * std::cos(alpha);
            const double t2    = (t - t_impact);
            if (std::tan(alpha) <= 3 * mu)
            {
                v_tangent = 2. * g * std::sin(alpha) * t2 / 3. + 2. * v_t_m / 3.;
                omega     = -2. * g * std::sin(alpha) * t2 / (3. * r) - 2 * v_t_m / (3. * r);
            }
            else
            {
                v_tangent = g * (std::sin(alpha) - mu * std::cos(alpha)) * t2 + (v_t_m + mu * v_n_m);
                omega     = -2. * mu * g * std::cos(alpha) * t2 / r + 2 * mu * v_n_m / r;
            }
            v[1] = 0;
            v[0] = v_tangent;
            return std::make_pair(v, omega);
        }
        v[1] = -g * t * std::cos(alpha);
        v[0] = g * t * std::sin(alpha);
        return std::make_pair(v, 0.);
    }

    /// Viscous with friction
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_velocity_viscous_friction(double alpha,
                                                                                                     double mu,
                                                                                                     double t,
                                                                                                     double r,
                                                                                                     double g,
                                                                                                     double y0,
                                                                                                     double gamma_min,
                                                                                                     double t_inv)
    {
        double t_impact    = std::sqrt(2 * (y0 - r) / (g * std::cos(alpha)));
        double t_gamma_min = -gamma_min / (g * std::cos(alpha));
        double gamma_inv;
        type::position_t<2> v;
        if (t_inv > t_gamma_min)
        {
            gamma_inv = gamma_min;
        }
        else
        {
            gamma_inv = -g * t_inv * std::cos(alpha);
        }
        if (t < t_impact)
        {
            v[1] = -g * t * std::cos(alpha);
            v[0] = g * t * std::sin(alpha);
            return std::make_pair(v, 0.);
        }

        if (t < t_gamma_min)
        {
            v[1] = 0.;
            v[0] = g * t * std::sin(alpha);
            return std::make_pair(v, 0.);
        }

        if (t < t_inv)
        {
            double omega;
            double v_t_gamma_min = -gamma_min * std::tan(alpha);
            double t2            = (t - t_gamma_min);
            if (std::tan(alpha) <= 3 * mu)
            {
                double t_no_slide = -std::tan(alpha) * t_gamma_min / (g * (std::tan(alpha) - 3 * mu)) + t_gamma_min;
                if (t > t_no_slide)
                {
                    double vt_no_slide    = g * (std::sin(alpha) - mu * std::cos(alpha)) * (t_no_slide - t_gamma_min) + v_t_gamma_min;
                    double omega_no_slide = -2. * mu * g * std::cos(alpha) * (t_no_slide - t_gamma_min) / r;
                    v[0]                  = 2. * g * std::sin(alpha) * (t - t_no_slide) / 3. + vt_no_slide;
                    omega                 = -2. * g * std::sin(alpha) * (t - t_no_slide) / (3. * r) + omega_no_slide;
                }
                else
                {
                    v[0]  = g * (std::sin(alpha) - mu * std::cos(alpha)) * t2 + v_t_gamma_min;
                    omega = -2. * mu * g * std::cos(alpha) * t2 / r;
                }
            }
            else
            {
                v[0]  = g * (std::sin(alpha) - mu * std::cos(alpha)) * t2 + v_t_gamma_min;
                omega = -2. * mu * g * std::cos(alpha) * t2 / r;
            }
            v[1] = 0.;
            return std::make_pair(v, omega);
        }

        double t_unstick     = t_inv - gamma_inv / (g * std::cos(alpha));
        double v_t_gamma_min = -gamma_min * std::tan(alpha);
        double t2            = (t_inv - t_gamma_min);
        double vt_tinv;
        double omega_tinv;
        if (std::tan(alpha) <= 3 * mu)
        {
            double t_no_slide = -std::tan(alpha) * t_gamma_min / (g * (std::tan(alpha) - 3 * mu)) + t_gamma_min;
            if (t_inv > t_no_slide)
            {
                double vt_no_slide    = g * (std::sin(alpha) - mu * std::cos(alpha)) * (t_no_slide - t_gamma_min) + v_t_gamma_min;
                double omega_no_slide = -2. * mu * g * std::cos(alpha) * (t_no_slide - t_gamma_min) / r;
                vt_tinv               = 2. * g * std::sin(alpha) * (t_inv - t_no_slide) / 3. + vt_no_slide;
                omega_tinv            = -2. * g * std::sin(alpha) * (t_inv - t_no_slide) / (3. * r) + omega_no_slide;
            }
            else
            {
                vt_tinv    = g * (std::sin(alpha) - mu * std::cos(alpha)) * t2 + v_t_gamma_min;
                omega_tinv = -2. * mu * g * std::cos(alpha) * t2 / r;
            }
        }
        else
        {
            vt_tinv    = g * (std::sin(alpha) - mu * std::cos(alpha)) * t2 + v_t_gamma_min;
            omega_tinv = -2. * mu * g * std::cos(alpha) * t2 / r;
        }
        v[0] = vt_tinv - g * std::sin(alpha) * (t - t_inv);
        if (t < t_unstick)
        {
            v[1] = 0;
        }
        else
        {
            v[1] = g * std::cos(alpha) * (t - t_unstick);
        }
        return std::make_pair(v, omega_tinv);
    }

    std::pair<type::position_t<2>, double>
    analytical_solution_sphere_plan_viscous_friction(double alpha, double mu, double t, double r, double g, double y0, double gamma_min, double t_inv)
    {
        double angle;
        double t_impact    = std::sqrt(2 * (y0 - r) / (g * std::cos(alpha)));
        double t_gamma_min = -gamma_min / (g * std::cos(alpha));
        type::position_t<2> x;
        double gamma_inv;
        if (t_inv > t_gamma_min)
        {
            gamma_inv = gamma_min;
        }
        else
        {
            gamma_inv = -g * t_inv * std::cos(alpha);
        }

        if (t < t_impact)
        {
            x[1] = -g * std::pow(t, 2) * std::cos(alpha) / 2 + y0;
            x[0] = g * std::pow(t, 2) * std::sin(alpha) / 2;
            return std::make_pair(x, 0.);
        }

        if (t < t_gamma_min)
        {
            x[1] = r;
            x[0] = g * std::pow(t, 2) * std::sin(alpha) / 2;
            return std::make_pair(x, 0.);
        }

        if (t < t_inv)
        {
            double v_t_gamma_min = -gamma_min * std::tan(alpha);
            double x_t_gamma_min = g * (std::pow(t_gamma_min, 2)) * std::sin(alpha) / 2;
            double t2            = (t - t_gamma_min);
            if (std::tan(alpha) <= 3 * mu)
            {
                double t_no_slide = -v_t_gamma_min / (g * std::cos(alpha) * (std::tan(alpha) - 3 * mu)) + t_gamma_min;
                if (t > t_no_slide)
                {
                    const double vt_no_slide    = g * (std::sin(alpha) - mu * std::cos(alpha)) * (t_no_slide - t_gamma_min) + v_t_gamma_min;
                    const double omega_no_slide = -2. * mu * g * std::cos(alpha) * (t_no_slide - t_gamma_min) / r;
                    const double xt_no_slide    = g * (std::sin(alpha) - mu * std::cos(alpha)) * (std::pow(t_no_slide - t_gamma_min, 2)) / 2
                                             + v_t_gamma_min * (t_no_slide - t_gamma_min) + x_t_gamma_min;
                    const double angle_no_slide = -2. * mu * g * std::cos(alpha) * (std::pow(t_no_slide - t_gamma_min, 2)) / (2 * r);
                    x[0]  = 2. * g * std::sin(alpha) * (std::pow(t - t_no_slide, 2)) / 6. + vt_no_slide * (t - t_no_slide) + xt_no_slide;
                    angle = -2. * g * std::sin(alpha) * (std::pow(t - t_no_slide, 2)) / (6. * r) + omega_no_slide * (t - t_no_slide)
                          + angle_no_slide;
                }
                else
                {
                    x[0]  = g * (std::sin(alpha) - mu * std::cos(alpha)) * std::pow(t2, 2) / 2 + v_t_gamma_min * t2 + x_t_gamma_min;
                    angle = -2. * mu * g * std::cos(alpha) * std::pow(t2, 2) / (2 * r);
                }
            }
            else
            {
                x[0]  = g * (std::sin(alpha) - mu * std::cos(alpha)) * std::pow(t2, 2) / 2 + v_t_gamma_min * t2 + x_t_gamma_min;
                angle = -2. * mu * g * std::cos(alpha) * std::pow(t2, 2) / (2 * r);
            }
            x[1] = r;
        }
        else
        {
            const double t_unstick     = t_inv - gamma_inv / (g * std::cos(alpha));
            const double v_t_gamma_min = -gamma_min * std::tan(alpha);
            const double x_t_gamma_min = g * (std::pow(t_gamma_min, 2)) * std::sin(alpha) / 2;
            const double t2            = (t_inv - t_gamma_min);
            double omega_tinv;
            double x_tinv;
            double angle_tinv;
            double vt_tinv;
            if (std::tan(alpha) <= 3 * mu)
            {
                double t_no_slide = -v_t_gamma_min / (g * std::cos(alpha) * (std::tan(alpha) - 3 * mu)) + t_gamma_min;
                if (t_inv > t_no_slide)
                {
                    const double vt_no_slide    = g * (std::sin(alpha) - mu * std::cos(alpha)) * (t_no_slide - t_gamma_min) + v_t_gamma_min;
                    const double omega_no_slide = -2. * mu * g * std::cos(alpha) * (t_no_slide - t_gamma_min) / r;
                    const double xt_no_slide    = g * (std::sin(alpha) - mu * std::cos(alpha)) * (std::pow(t_no_slide - t_gamma_min, 2)) / 2
                                             + v_t_gamma_min * (t_no_slide - t_gamma_min) + x_t_gamma_min;
                    const double angle_no_slide = -2. * mu * g * std::cos(alpha) * (std::pow(t_no_slide - t_gamma_min, 2)) / (2 * r);
                    vt_tinv                     = 2. * g * std::sin(alpha) * (t_inv - t_no_slide) / 3. + vt_no_slide;
                    x_tinv = 2. * g * std::sin(alpha) * (std::pow(t_inv - t_no_slide, 2)) / 6. + vt_no_slide * (t_inv - t_no_slide)
                           + xt_no_slide;
                    omega_tinv = -2. * g * std::sin(alpha) * (t_inv - t_no_slide) / (3. * r) + omega_no_slide;
                    angle_tinv = -2. * g * std::sin(alpha) * (std::pow(t_inv - t_no_slide, 2)) / (6. * r)
                               + omega_no_slide * (t_inv - t_no_slide) + angle_no_slide;
                }
                else
                {
                    vt_tinv    = g * (std::sin(alpha) - mu * std::cos(alpha)) * t2 + v_t_gamma_min;
                    x_tinv     = g * (std::sin(alpha) - mu * std::cos(alpha)) * std::pow(t2, 2) / 2 + v_t_gamma_min * t2 + x_t_gamma_min;
                    omega_tinv = -2. * mu * g * std::cos(alpha) * t2 / r;
                    angle_tinv = -2. * mu * g * std::cos(alpha) * std::pow(t2, 2) / (2 * r);
                }
            }
            else
            {
                vt_tinv    = g * (std::sin(alpha) - mu * std::cos(alpha)) * t2 + v_t_gamma_min;
                x_tinv     = g * (std::sin(alpha) - mu * std::cos(alpha)) * std::pow(t2, 2) / 2 + v_t_gamma_min * t2 + x_t_gamma_min;
                omega_tinv = -2. * mu * g * std::cos(alpha) * t2 / r;
                angle_tinv = -2. * mu * g * std::cos(alpha) * std::pow(t2, 2) / (2 * r);
            }

            x[0]  = vt_tinv * (t - t_inv) - g * std::sin(alpha) * std::pow((t - t_inv), 2) / 2 + x_tinv;
            angle = omega_tinv * (t - t_inv) + angle_tinv;
            if (t < t_unstick)
            {
                x[1] = r;
            }
            else
            {
                x[1] = g * std::cos(alpha) * pow(t - t_unstick, 2) / 2 + r;
            }
        }

        return std::make_pair(x, angle);
    }
}

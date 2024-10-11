#pragma once

#include <nlohmann/json.hpp>

namespace nl = nlohmann;

#include "../scopi.hpp"
#include "../utils.hpp"

namespace scopi
{

    // Property type
    class NoFriction
    {
    };

    class Friction
    {
    };

    class FixedPoint
    {
    };

    class FrictionFixedPoint
    {
    };

    class Viscous
    {
    };

    class ViscousFriction
    {
    };

    template <class problem_t>
    struct contact_property;

    template <>
    struct contact_property<NoFriction>
    {
        static auto to_json()
        {
            return nl::json{
                {"type", "no friction"},
            };
        }
    };

    template <>
    struct contact_property<Friction>
    {
        contact_property()
        {
            auto& sub = get_app();
            if (!check_option(sub, "--mu"))
            {
                sub.add_option("--mu", mu, "Friction coefficient")->capture_default_str();
            }
        }

        auto to_json() const
        {
            return nl::json{
                {"type", "friction"},
                {"mu",   mu        },
            };
        }

        double mu = 0.1;
    };

    template <>
    struct contact_property<FixedPoint>
    {
        contact_property()
        {
            auto& sub = get_app();
            if (!check_option(sub, "--fixed-point-tol"))
            {
                sub.add_option("--fixed-point-tol", fixed_point_tol, "Fixed point tolerance")->capture_default_str();
                sub.add_option("--fixed-point-max-iter", fixed_point_max_iter, "Fixed point max iterations")->capture_default_str();
            }
        }

        auto to_json() const
        {
            return nl::json{
                {"type",                 "fixed point"       },
                {"fixed_point_tol",      fixed_point_tol     },
                {"fixed_point_max_iter", fixed_point_max_iter},
            };
        }

        double fixed_point_tol      = 1e-6;
        double fixed_point_max_iter = 1000;
    };

    template <>
    struct contact_property<Viscous>
    {
        contact_property()
        {
            auto& sub = get_app();
            if (!check_option(sub, "--gamma"))
            {
                sub.add_option("--gamma", gamma, "Adhesion potential")->capture_default_str();
                sub.add_option("--gamma-min", gamma_min, "Adhesion potential threshold")->capture_default_str();
                sub.add_option("--gamma-tol", gamma_tol, "Adhesion potential tolerance")->capture_default_str();
            }
        }

        auto to_json() const
        {
            return nl::json{
                {"type",      "viscous"},
                {"gamma",     gamma    },
                {"gamma_min", gamma_min},
                {"gamma_tol", gamma_tol},
            };
        }

        double gamma     = 0;
        double gamma_min = -2.;
        double gamma_tol = 1e-6;
    };

    template <>
    struct contact_property<FrictionFixedPoint> : public contact_property<Friction>,
                                                  contact_property<FixedPoint>
    {
        // contact_property()
        //     : contact_property<Friction>()
        //     , contact_property<FixedPoint>()
        // {
        // }

        auto to_json() const
        {
            return nl::json{
                {"type",                 "friction fixed point"},
                {"mu",                   mu                    },
                {"fixed_point_tol",      fixed_point_tol       },
                {"fixed_point_max_iter", fixed_point_max_iter  },
            };
        }
    };

    template <>
    struct contact_property<ViscousFriction> : public contact_property<Viscous>,
                                               contact_property<Friction>,
                                               contact_property<FixedPoint>
    {
        // contact_property()
        //     : contact_property<Viscous>()
        //     , contact_property<Friction>()
        //     , contact_property<FixedPoint>()
        // {
        // }

        auto to_json() const
        {
            return nl::json{
                {"type",                 "viscous friction"  },
                {"gamma",                gamma               },
                {"gamma_min",            gamma_min           },
                {"gamma_tol",            gamma_tol           },
                {"mu",                   mu                  },
                {"fixed_point_tol",      fixed_point_tol     },
                {"fixed_point_max_iter", fixed_point_max_iter},
            };
        }
    };

    template <class ostream>
    void to_stream(ostream& out, int indent, const contact_property<NoFriction>&)
    {
        print_indented(out, indent, "{:<12} : no friction", "type");
    }

    template <class ostream>
    void to_stream(ostream& out, int indent, const contact_property<Friction>& prop)
    {
        print_indented(out, indent, "{:<12} : friction", "type");
        print_indented(out, indent + 4, "{:<12} : {}", "mu", prop.mu);
    }

    template <class ostream>
    void to_stream(ostream& out, int indent, const contact_property<FixedPoint>& prop)
    {
        print_indented(out, indent, "{:<12} : fixed point", "type");
        print_indented(out, indent + 4, "{:<12} : {}", "fixed_point_tol", prop.fixed_point_tol);
        print_indented(out, indent + 4, "{:<12} : {}", "fixed_point_max_iter", prop.fixed_point_max_iter);
    }

    template <class ostream>
    void to_stream(ostream& out, int indent, const contact_property<Viscous>& prop)
    {
        print_indented(out, indent, "{:<12} : viscous", "type");
        print_indented(out, indent + 4, "{:<12} : {}", "gamma", prop.gamma);
        print_indented(out, indent + 4, "{:<12} : {}", "gamma_min", prop.gamma_min);
        print_indented(out, indent + 4, "{:<12} : {}", "gamma_tol", prop.gamma_tol);
    }

    template <class ostream>
    void to_stream(ostream& out, int indent, const contact_property<ViscousFriction>& prop)
    {
        to_stream(out, indent, static_cast<const contact_property<Viscous>&>(prop));
        to_stream(out, indent, static_cast<const contact_property<Friction>&>(prop));
        to_stream(out, indent, static_cast<const contact_property<FixedPoint>&>(prop));
    }
}

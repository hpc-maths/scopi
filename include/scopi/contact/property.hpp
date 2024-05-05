#pragma once

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
        double gamma     = 0;//Pour l'affichage
        double gamma_min = -2.;//Pour l'affichage
    };

    template <>
    struct contact_property<Friction>
    {
        double mu = 0.1;
        double gamma     = 0;//Pour l'affichage
        double gamma_min = -2.;//Pour l'affichage
    };

    template <>
    struct contact_property<FrictionFixedPoint>
    {
        double mu = 0.5;
        double fixed_point_tol = 1e-6;
        double fixed_point_max_iter = 1000;
        double gamma     = 0;//Pour l'affichage
        double gamma_min = -2.;//Pour l'affichage
    };

    template <>
    struct contact_property<Viscous>
    {
        double gamma     = 0;
        double gamma_min = -2.;
        double gamma_tol = 1e-6;
    };

    template <>
    struct contact_property<ViscousFriction>
    {
        double mu = 0.5;
        double fixed_point_tol = 1e-3;
        double fixed_point_max_iter = 1000;
        double gamma     = 0;
        double gamma_min = -1.4;
        double gamma_tol = 1e-6;
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
        print_indented(out, indent, "{:<12} : {}", "mu", prop.mu);
    }

    template <class ostream>
    void to_stream(ostream& out, int indent, const contact_property<Viscous>& prop)
    {
        print_indented(out, indent, "{:<12} : viscous", "type");
        print_indented(out, indent, "{:<12} : {}", "gamma", prop.gamma);
        print_indented(out, indent, "{:<12} : {}", "gamma_min", prop.gamma_min);
        print_indented(out, indent, "{:<12} : {}", "gamma_tol", prop.gamma_tol);
    }
}

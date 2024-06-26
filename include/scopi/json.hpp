#pragma once

#include <nlohmann/json.hpp>

#include <xtensor/xjson.hpp>
#include <xtensor/xtensor.hpp>

#include "property.hpp"

namespace nl = nlohmann;

namespace nlohmann
{
    template <std::size_t dim>
    struct adl_serializer<scopi::property<dim>>
    {
        using xt_t       = xt::xtensor<double, 1>;
        using xt_t2      = xt::xtensor<double, 2>;
        using moment_t   = typename std::conditional<dim == 2, double, xt_t>::type;
        using rotation_t = typename std::conditional<dim == 2, double, xt_t>::type;

        static void to_json(json&, const scopi::property<dim>&)
        {
        }

        static void from_json(const json& j, scopi::property<dim>& p)
        {
            p.velocity(j["velocity"].get<xt_t>())
                .desired_velocity(j["desired_velocity"].get<xt_t>())
                .omega(j["omega"].get<rotation_t>())
                .desired_omega(j["desired_omega"].get<rotation_t>())
                .force(j["force"].get<xt_t>())
                .mass(j["mass"].get<double>())
                .moment_inertia(j["moment_inertia"].get<moment_t>());
            if (j["active"].get<bool>())
            {
                p.activate();
            }
            else
            {
                p.deactivate();
            }
        }
    };
}

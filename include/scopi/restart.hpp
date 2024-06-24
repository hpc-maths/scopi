#pragma once

#include <nlohmann/json.hpp>
#include <xtensor/xjson.hpp>

#include "container.hpp"
#include "json.hpp"
#include "objects/types/plane.hpp"
#include "objects/types/segment.hpp"
#include "objects/types/sphere.hpp"
#include "objects/types/superellipsoid.hpp"
#include "objects/types/worm.hpp"
#include "property.hpp"

namespace nl = nlohmann;

namespace scopi
{

    template <std::size_t dim>
    auto from_json(const nl::json& j)
    {
        using xt_t = xt::xtensor<double, 1>;
        scopi_container<dim> container;

        for (const auto& o : j["objects"])
        {
            std::string type = o["type"];

            if (type == "sphere")
            {
                sphere<dim> s({o["position"].get<xt_t>()}, {o["quaternion"].get<xt_t>()}, o["radius"].get<double>());
                container.push_back(s, o["properties"]);
            }
            else if (type == "superellipsoid")
            {
                superellipsoid<dim> s({o["position"].get<xt_t>()},
                                      {o["quaternion"].get<xt_t>()},
                                      o["radius"].get<xt_t>(),
                                      o["squareness"].get<xt_t>());
                container.push_back(s, o["properties"]);
            }
            else if (type == "segment")
            {
                segment<dim> s(o["p1"].get<xt_t>(), o["p2"].get<xt_t>());
                container.push_back(s, o["properties"]);
            }
            else if (type == "plane")
            {
                plane<dim> s({o["position"].get<xt_t>()}, {o["quaternion"].get<xt_t>()});
                container.push_back(s, o["properties"]);
            }
            else if (type == "worm")
            {
                std::vector<type::position_t<dim>> pos;
                std::vector<type::quaternion_t> q;
                for (auto& p : o["worm"])
                {
                    pos.push_back(p["position"].get<xt_t>());
                    q.push_back(p["quaternion"].get<xt_t>());
                }
                worm<dim> w(pos, q, o["worm"][0]["radius"], pos.size());
                container.push_back(w, o["properties"]);
            }
        }
        return container;
    }

    // template <std::size_t dim, class Contact>
    // auto contact_from_json(const nl::json& j)
    // {
    //     using xt_t = xt::xtensor<double, 1>;
    //     std::vector<std::pair<std::size_t, std::size_t>> contacts;

    //     for (const auto& c : j["contacts"])
    //     {
    //         contacts.push_back({c["id1"], c["id2"]});
    //     }
    //     return contacts;
    // }
}

#pragma once

#include <cstddef>
#include <iostream>
#include <iterator>
#include <regex>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xjson.hpp>

#include "../dispatch.hpp"
#include "../neighbor.hpp"
#include "../types/plane.hpp"
#include "../types/segment.hpp"
#include "../types/sphere.hpp"
#include "../types/superellipsoid.hpp"
#include "../types/worm.hpp"

#include <nlohmann/json.hpp>

namespace nl = nlohmann;

namespace scopi
{

    // SORTIES PREVUES POUR LE FORMAT JSON

    // SPHERE
    /**
     * @brief Write the elements of a sphere in json format.
     *
     * @tparam dim Dimension (2 or 3).
     * @param s [in] Sphere.
     *
     * @return nlohmann json object.
     */
    template <std::size_t dim>
    nl::json write_objects(const sphere<dim, false>& s, [[maybe_unused]] std::size_t id)
    {
        nl::json object;

        object["type"]       = "sphere";
        object["id"]         = id;
        object["position"]   = xt::flatten(s.pos());
        object["radius"]     = s.radius();
        object["rotation"]   = xt::flatten(s.rotation());
        object["quaternion"] = xt::flatten(s.q());

        return object;
        // std::cout << "write_objects : SPHERE" << std::endl;
        // std::stringstream ss;
        // std::regex xp1("\\{|\\}");
        // std::regex xp2("\\.,");
        // std::regex xp3("\\.\\]");
        // ss << std::scientific << std::setprecision(14) <<
        //   "\"type\": \"sphere\", " <<
        //   "\"position\": [" << s.pos() << "], " <<
        //   "\"radius\": " << s.radius() << ", " <<
        //   "\"rotation\": ["  << xt::flatten(s.rotation()) <<
        //   "]";
        // auto ssss = std::regex_replace(std::regex_replace(std::regex_replace(ss.str(), xp1, ""), xp2, ".0,"), xp3, ".0]");
        // return ssss;
    }

    // SUPERELLIPSOID
    /**
     * @brief Write the elements of a superellipsoid in json format.
     *
     * @tparam dim Dimension (2 or 3).
     * @param s [in] Superellipsoid.
     *
     * @return nlohmann json object.
     */
    template <std::size_t dim>
    nl::json write_objects(const superellipsoid<dim, false>& s, std::size_t id)
    {
        nl::json object;

        object["type"]       = "superellipsoid";
        object["id"]         = id;
        object["position"]   = xt::flatten(s.pos());
        object["radius"]     = s.radius();
        object["squareness"] = s.squareness();
        object["rotation"]   = xt::flatten(s.rotation());
        object["quaternion"] = xt::flatten(s.q());

        return object;

        // std::cout << "write_objects : SUPERELLIPSOID" << std::endl;
        // std::stringstream ss;
        // std::regex xp1("\\{|\\}");
        // std::regex xp2("\\.,");
        // std::regex xp3("\\.\\]");
        // ss << std::scientific << std::setprecision(14) <<
        //   "\"type\": \"superellipsoid\", " <<
        //   "\"position\": [" << s.pos() << "], " <<
        //   "\"radius\": [" << s.radius() << "], " <<
        //   "\"squareness\": [" << s.squareness() << "], " <<
        //   "\"rotation\": ["  << xt::flatten(s.rotation()) <<
        //   "]";
        // auto ssss = std::regex_replace(std::regex_replace(std::regex_replace(ss.str(), xp1, ""), xp2, ".0,"), xp3, ".0]");
        // return ssss;
    }

    // PLANE
    /**
     * @brief
     * @brief Write the elements of a plane in json format.
     *
     * @tparam dim Dimension (2 or 3).
     * @param p [in] Plane.
     *
     * @return nlohmann json object.
     */
    template <std::size_t dim>
    nl::json write_objects(const plane<dim, false>& p, [[maybe_unused]] std::size_t id)
    {
        nl::json object;

        object["type"]       = "plane";
        object["id"]         = id;
        object["position"]   = xt::flatten(p.pos());
        object["normal"]     = p.normal();
        object["rotation"]   = xt::flatten(p.rotation());
        object["quaternion"] = xt::flatten(p.q());

        return object;

        // std::cout << "write_objects : PLANE" << std::endl;
        // std::stringstream ss;
        // std::regex xp1("\\{|\\}");
        // std::regex xp2("\\.,");
        // std::regex xp3("\\.\\]");
        // ss << std::scientific << std::setprecision(14) <<
        //   "\"type\": \"plane\", " <<
        //   "\"position\": [" << p.pos() << "], " <<
        //   "\"normal\": [" << p.normal() << "], " <<
        //   "\"rotation\": ["  << xt::flatten(p.rotation()) <<
        //   "]";
        // auto ssss = std::regex_replace(std::regex_replace(std::regex_replace(ss.str(), xp1, ""), xp2, ".0,"), xp3, ".0]");
        // return ssss;
    }

    // SEGMENT
    /**
     * @brief
     * @brief Write the elements of a segment in json format.
     *
     * @tparam dim Dimension (2 or 3).
     * @param p [in] Plane.
     *
     * @return nlohmann json object.
     */
    template <std::size_t dim>
    nl::json write_objects(const segment<dim, false>& seg, std::size_t id)
    {
        nl::json object;

        object["type"]       = "segment";
        object["id"]         = id;
        auto extrema         = seg.extrema();
        object["p1"]         = extrema[0];
        object["p2"]         = extrema[1];
        object["normal"]     = seg.normal();
        object["tangent"]    = seg.tangent();
        object["quaternion"] = xt::flatten(seg.q());

        return object;
    }

    // WORM
    /**
     * @brief Write the elements of a worm in json format.
     *
     * @tparam dim Dimension (2 or 3).
     * @param w [in] Worm.
     *
     * @return nlohmann json object.
     */
    template <std::size_t dim>
    nl::json write_objects(const worm<dim, false>& w, std::size_t id)
    {
        nl::json object;
        object["type"] = "worm";
        for (std::size_t i = 0; i < w.size(); ++i)
        {
            nl::json json_worm;
            json_worm["position"]   = xt::flatten(w.pos(i));
            json_worm["id"]         = id + i;
            json_worm["radius"]     = w.radius();
            json_worm["quaternion"] = xt::flatten(w.q(i));
            object["worm"].push_back(json_worm);
        }
        return object;
        // std::cout << "write_objects : WORM" << std::endl;
        // std::stringstream ss;
        // ss << "redcell; ";
        // return ss.str();
    }

    /**
     * @brief
     *
     * \todo Write documentation.
     *
     * @tparam dim Dimension (2 or 3).
     */
    template <std::size_t dim>
    struct write_objects_functor
    {
        /**
         * @brief Alias for nlohmann json object.
         */
        using return_type = nl::json;

        /**
         * @brief
         *
         * \todo Write documentation.
         *
         * @tparam T1
         * @param obj1
         *
         * @return
         */
        template <class T1>
        return_type run(const T1& obj1, std::size_t id) const
        {
            return write_objects(obj1, id);
        }

        /**
         * @brief
         *
         * \todo Write documentation.
         *
         * @param object
         *
         * @return
         */
        return_type on_error(const object<dim, false>&, std::size_t) const
        {
            return nl::json::object();
        }
    };

    /**
     * @brief
     *
     * \todo Write documentation.
     *
     * @tparam dim Dimension (2 or 3).
     */
    template <std::size_t dim>
    using write_objects_dispatcher = unit_static_dispatcher<
        write_objects_functor<dim>,
        const object<dim, false>,
        mpl::vector<const sphere<dim, false>, const superellipsoid<dim, false>, const worm<dim, false>, const plane<dim, false>, const segment<dim, false>>,
        typename write_objects_functor<dim>::return_type>;
}

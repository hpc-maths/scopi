#pragma once

#include <iostream>
#include <iterator>
#include <regex>
#include <string>

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include "sphere.hpp"
#include "superellipsoid.hpp"
#include "globule.hpp"
#include "plan.hpp"
#include "neighbor.hpp"

#include "nlohmann/json.hpp"

namespace nl = nlohmann;

namespace scopi
{

    // SORTIES PREVUES POUR LE FORMAT JSON

    // SPHERE
    template<std::size_t dim>
    nl::json write_objects(const sphere<dim, false>& s)
    {
        nl::json object;

        object["type"] = "sphere";
        object["position"] = s.pos();
        object["radius"] = s.radius();
        object["rotation"] = xt::flatten(s.rotation());
        object["quaternion"] = s.q();

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
    template<std::size_t dim>
    nl::json write_objects(const superellipsoid<dim, false>& s)
    {
        nl::json object;

        object["type"] = "superellipsoid";
        object["position"] = s.pos();
        object["radius"] = s.radius();
        object["squareness"] = s.squareness();
        object["rotation"] = xt::flatten(s.rotation());
        object["quaternion"] = s.q();

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

    // PLAN
    template<std::size_t dim>
    nl::json write_objects(const plan<dim, false> p)
    {
        nl::json object;

        object["type"] = "plan";
        object["position"] = p.pos();
        object["normal"] = p.normal();
        object["rotation"] = xt::flatten(p.rotation());
        object["quaternion"] = p.q();

        return object;

      // std::cout << "write_objects : PLAN" << std::endl;
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

    // GLOBULE
    template<std::size_t dim>
    nl::json write_objects(const globule<dim, false>)
    {
      return {};
      // std::cout << "write_objects : GLOBULE" << std::endl;
      // std::stringstream ss;
      // ss << "redcell; ";
      // return ss.str();
    }

}

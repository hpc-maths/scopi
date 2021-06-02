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


namespace scopi
{

    // SORTIES PREVUES POUR LE FORMAT JSON

    // SPHERE
    template<std::size_t dim>
    auto write_objects(const sphere<dim, false>& s)
    {
        std::cout << "write_objects : SPHERE" << std::endl;
        std::stringstream ss;
        std::regex xp1("\\{|\\}");
        std::regex xp2("\\.,");
        std::regex xp3("\\.\\]");
        ss << std::scientific << std::setprecision(14) <<
          "\"type\": \"sphere\", " <<
          "\"position\": [" << s.pos() << "], " <<
          "\"radius\": " << s.radius() << ", " <<
          "\"rotation\": ["  << xt::flatten(s.rotation()) <<
          "]";
        auto ssss = std::regex_replace(std::regex_replace(std::regex_replace(ss.str(), xp1, ""), xp2, ".0,"), xp3, ".0]");
        return ssss;
    }


    // SUPERELLIPSOID
    template<std::size_t dim>
    auto write_objects(const superellipsoid<dim, false>& s)
    {
        std::cout << "write_objects : SUPERELLIPSOID" << std::endl;
        std::stringstream ss;
        std::regex xp1("\\{|\\}");
        std::regex xp2("\\.,");
        std::regex xp3("\\.\\]");
        ss << std::scientific << std::setprecision(14) <<
          "\"type\": \"superellipsoid\", " <<
          "\"position\": [" << s.pos() << "], " <<
          "\"radius\": [" << s.radius() << "], " <<
          "\"squareness\": [" << s.squareness() << "], " <<
          "\"rotation\": ["  << xt::flatten(s.rotation()) <<
          "]";
        auto ssss = std::regex_replace(std::regex_replace(std::regex_replace(ss.str(), xp1, ""), xp2, ".0,"), xp3, ".0]");
        return ssss;
    }

    // PLAN
    template<std::size_t dim>
    auto write_objects(const plan<dim, false> p)
    {
      std::cout << "write_objects : PLAN" << std::endl;
      std::stringstream ss;
      std::regex xp1("\\{|\\}");
      std::regex xp2("\\.,");
      std::regex xp3("\\.\\]");
      ss << std::scientific << std::setprecision(14) <<
        "\"type\": \"plane\", " <<
        "\"position\": [" << p.pos() << "], " <<
        "\"normal\": [" << p.normal() << "], " <<
        "\"rotation\": ["  << xt::flatten(p.rotation()) <<
        "]";
      auto ssss = std::regex_replace(std::regex_replace(std::regex_replace(ss.str(), xp1, ""), xp2, ".0,"), xp3, ".0]");
      return ssss;
    }

    // GLOBULE
    template<std::size_t dim>
    auto write_objects(const globule<dim, false>)
    {
      std::cout << "write_objects : GLOBULE" << std::endl;
      std::stringstream ss;
      ss << "redcell; ";
      return ss.str();
    }

}

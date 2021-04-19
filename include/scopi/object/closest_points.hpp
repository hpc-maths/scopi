#pragma once

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>

#include "sphere.hpp"
#include "superellipsoid.hpp"
#include "globule.hpp"
#include "plan.hpp"

namespace scopi
{
    // SPHERE - SPHERE
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>& s1, const sphere<dim, false>& s2)
    {
        std::cout << "closest_points : SPHERE - SPHERE" << std::endl;
        xt::xtensor_fixed<double, xt::xshape<2, dim>> pts;
        auto s1_pos = xt::view(s1.pos(), 0);
        auto s2_pos = xt::view(s2.pos(), 0);
        auto s1_to_s2 = s2_pos - s1_pos;
        xt::view(pts, 0) = s1_pos + s1.radius()*s1_to_s2;
        xt::view(pts, 1) = s2_pos - s2.radius()*s1_to_s2;
        return pts;
    }

    // SUPERELLIPSOID - SUPERELLIPSOID
    template<std::size_t dim>
    auto closest_points(const superellipsoid<dim, false>& s1, const superellipsoid<dim, false>& s2)
    {
        std::cout << "closest_points : SUPERELLIPSOID - SUPERELLIPSOID" << std::endl;
        xt::xtensor_fixed<double, xt::xshape<2, dim>> pts;
        xt::xtensor_fixed<double, xt::xshape<2*(2*dim+dim-1+dim*dim)>> args = xt::hstack(xt::xtuple(
          xt::view(s1.pos(), 0), s1.radius(), s1.squareness(), xt::flatten(s1.rotation()),
          xt::view(s2.pos(), 0), s2.radius(), s2.squareness(), xt::flatten(s2.rotation())
        ));
        // std::cout << "args = " << args << std::endl;
        if (dim==2) {
            auto newton_F = [](auto u, auto args)
            {
                double b1 = u(0);
                double b2 = u(1);
                double s1xc = args(0);
                double s1yc = args(1);
                double s1rx = args(2);
                double s1ry = args(3);
                double s1e = args(4);
                double Q1 = args(5);
                double Q2 = args(8);
                double Q3 = args(7);
                double Q4 = args(6);
                double s2xc = args(9);
                double s2yc = args(10);
                double s2rx = args(11);
                double s2ry = args(12);
                double s2e = args(13);
                double R1 = args(14);
                double R2 = args(17);
                double R3 = args(16);
                double R4 = args(15);
                double sb2e1 = std::pow(std::fabs(std::sin(b1)),(2 - s1e)) * sign(std::sin(b1));
                double sbe1  = std::pow(std::fabs(std::sin(b1)),s1e) * sign(std::sin(b1));
                double cb2e1 = std::pow(std::fabs(std::cos(b1)),(2 - s1e)) * sign(std::cos(b1));
                double cbe1  = std::pow(std::fabs(std::cos(b1)),s1e) * sign(std::cos(b1));
                double sb2e2 = std::pow(std::fabs(std::sin(b2)),(2 - s2e)) * sign(std::sin(b2));
                double sbe2  = std::pow(std::fabs(std::sin(b2)),s2e) * sign(std::sin(b2));
                double cb2e2 = std::pow(std::fabs(std::cos(b2)),(2 - s2e)) * sign(std::cos(b2));
                double cbe2  = std::pow(std::fabs(std::cos(b2)),s2e) * sign(std::cos(b2));
                xt::xtensor_fixed<double, xt::xshape<2>> res;
                res(0) = -( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1 ) *
                    ( -s1rx*Q3*cbe1 - s1ry*Q2*sbe1 - s1yc + s2rx*R3*cbe2 + s2ry*R2*sbe2 + s2yc ) +
                    ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1) *
                    ( -s1rx*Q1*cbe1 - s1ry*Q4*sbe1 - s1xc + s2rx*R1*cbe2 + s2ry*R4*sbe2 + s2xc );
                res(1) = ( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1 ) *
                    ( s2rx*R4*sb2e2 + s2ry*R1*cb2e2 ) +
                    ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1 ) *
                    ( s2rx*R2*sb2e2 + s2ry*R3*cb2e2 ) +
                    std::sqrt( std::pow(s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1, 2) + std::pow(s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1, 2) ) *
                    std::sqrt( std::pow(s2rx*R4*sb2e2 + s2ry*R1*cb2e2, 2) + std::pow(s2rx*R2*sb2e2 + s2ry*R3*cb2e2, 2) );
                return res;
            };
            auto newton_GradF = [](auto u, auto args)
            {
                double b1 = u(0);
                double b2 = u(1);
                double s1xc = args(0);
                double s1yc = args(1);
                double s1rx = args(2);
                double s1ry = args(3);
                double s1e = args(4);
                double Q1 = args(5);
                double Q2 = args(8);
                double Q3 = args(7);
                double Q4 = args(6);
                double s2xc = args(9);
                double s2yc = args(10);
                double s2rx = args(11);
                double s2ry = args(12);
                double s2e = args(13);
                double R1 = args(14);
                double R2 = args(17);
                double R3 = args(16);
                double R4 = args(15);
                double sb2e1 = std::pow(std::fabs(std::sin(b1)),(2 - s1e)) * sign(std::sin(b1));
                double sbe1  = std::pow(std::fabs(std::sin(b1)),s1e) * sign(std::sin(b1));
                double cb2e1 = std::pow(std::fabs(std::cos(b1)),(2 - s1e)) * sign(std::cos(b1));
                double cbe1  = std::pow(std::fabs(std::cos(b1)),s1e) * sign(std::cos(b1));
                double sb2e2 = std::pow(std::fabs(std::sin(b2)),(2 - s2e)) * sign(std::sin(b2));
                double sbe2  = std::pow(std::fabs(std::sin(b2)),s2e) * sign(std::sin(b2));
                double cb2e2 = std::pow(std::fabs(std::cos(b2)),(2 - s2e)) * sign(std::cos(b2));
                double cbe2  = std::pow(std::fabs(std::cos(b2)),s2e) * sign(std::cos(b2));
                double scb1e1 = std::sin(b1) * std::pow(std::fabs(std::cos(b1)),s1e - 1);
                double csb1e1 = std::cos(b1) * std::pow(std::fabs(std::sin(b1)),s1e - 1);
                double iscb1e1 = std::sin(b1) * std::pow(std::fabs(std::cos(b1)),1 - s1e);
                double icsb1e1 = std::cos(b1) * std::pow(std::fabs(std::sin(b1)),1 - s1e);
                double scb1e2 = std::sin(b2) * std::pow(std::fabs(std::cos(b2)),s2e - 1);
                double csb1e2 = std::cos(b2) * std::pow(std::fabs(std::sin(b2)),s2e - 1);
                double iscb1e2 = std::sin(b2) * std::pow(std::fabs(std::cos(b2)),1 - s2e);
                double icsb1e2 = std::cos(b2) * std::pow(std::fabs(std::sin(b2)),1 - s2e);
                xt::xtensor_fixed<double, xt::xshape<2,2>> res;
                res(0,0) =  ( -s1rx*Q4*sb2e1 - s1ry*Q1*cb2e1 ) *
                    ( s1e*s1rx*Q3*scb1e1 - s1e*s1ry*Q2*csb1e1 ) +
                    ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1) *
                    ( s1e*s1rx*Q1*scb1e1 - s1e*s1ry*Q4*csb1e1 ) +
                    ( -s1rx*(2 - s1e)*Q4*icsb1e1 + s1ry*(2 - s1e)*Q1*iscb1e1 ) *
                    ( -s1rx*Q3*cbe1 - s1ry*Q2*sbe1 - s1yc + s2rx*R3*cbe2 + s2ry*R2*sbe2 + s2yc) +
                    ( s1rx*(2 - s1e)*Q2*icsb1e1 - s1ry*(2 - s1e)*Q3*iscb1e1 ) *
                    ( -s1rx*Q1*cbe1 - s1ry*Q4*sbe1 - s1xc + s2rx*R1*cbe2 + s2ry*R4*sbe2 + s2xc );
                res(0,1) = ( -s1rx*Q4*sb2e1 - s1ry*Q1*cb2e1 ) *
                    ( -s2e*s2rx*R3*scb1e2 + s2e*s2ry*R2*csb1e2 ) +
                    ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1 ) *
                    ( -s2e*s2rx*R1*scb1e2 + s2e*s2ry*R4*csb1e2 );
                res(1,0) =  (
                    ( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1 ) *
                    ( 2*s1rx*(2 - s1e)*Q4*icsb1e1 - 2*s1ry*(2 - s1e)*Q1*iscb1e1 )/2 +
                    ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1) *
                    ( 2*s1rx*(2 - s1e)*Q2*icsb1e1 - 2*s1ry*(2 - s1e)*Q3*iscb1e1 )/2
                    ) *
                    std::sqrt(
                      std::pow( s2rx*R4*sb2e2 + s2ry*R1*cb2e2, 2) +
                      std::pow( s2rx*R2*sb2e2 + s2ry*R3*cb2e2, 2)
                    ) / std::sqrt(
                      std::pow( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1, 2) +
                      std::pow( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1, 2)
                    ) +
                    ( s2rx*R4*sb2e2 + s2ry*R1*cb2e2 ) *
                    ( s1rx*(2 - s1e)*Q4*icsb1e1 - s1ry*(2 - s1e)*Q1*iscb1e1 ) +
                    ( s2rx*R2*sb2e2 + s2ry*R3*cb2e2) *
                    ( s1rx*(2 - s1e)*Q2*icsb1e1 - s1ry*(2 - s1e)*Q3*iscb1e1 );
                res(1,1) = (
                    (s2rx*R4*sb2e2 + s2ry*R1*cb2e2) *
                    (2*s2rx*(2 - s2e)*R4*icsb1e2 - 2*s2ry*(2 - s2e)*R1*iscb1e2 )/2 +
                    (s2rx*R2*sb2e2 + s2ry*R3*cb2e2) *
                    (2*s2rx*(2 - s2e)*R2*icsb1e2 - 2*s2ry*(2 - s2e)*R3*iscb1e2 )/2) *
                    std::sqrt(
                      std::pow( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1, 2) +
                      std::pow( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1, 2)
                    ) / std::sqrt(
                      std::pow( s2rx*R4*sb2e2 + s2ry*R1*cb2e2, 2) +
                      std::pow( s2rx*R2*sb2e2 + s2ry*R3*cb2e2, 2)
                    ) +
                    ( s1rx*Q4*sb2e1 + s1ry*Q1*cb2e1 ) *
                    ( s2rx*(2 - s2e)*R4*icsb1e2 - s2ry*(2 - s2e)*R1*iscb1e2 ) +
                    ( s1rx*Q2*sb2e1 + s1ry*Q3*cb2e1 ) *
                    ( s2rx*(2 - s2e)*R2*icsb1e2 - s2ry*(2 - s2e)*R3*iscb1e2 );
                return res;
            };
            // std::cout << "newton_F = " << newton_F(u0,args) << "\n" << std::endl;
            // std::cout << "newton_GradF = " << newton_GradF(u0,args) << "\n" << std::endl;
            double pi = 4*std::atan(1);
            xt::xtensor_fixed<double, xt::xshape<4>> binit = { -pi, -pi/2, 0.001, pi/2 };
            xt::xtensor_fixed<double, xt::xshape<4,4>> dinit;
            for (std::size_t i = 0; i < binit.size(); i++) {
                for (std::size_t j = 0; j < binit.size(); j++) {
                    dinit(i,j) = xt::linalg::norm(s1.point(binit(i))-s2.point(binit(j)),2);
                }
            }
            // std::cout << "initialization : b = " << binit << " distances = " << dinit << std::endl;
            auto dmin = xt::amin(dinit);
            // std::cout << "initialization : dmin = " << dmin << std::endl;
            auto indmin = xt::from_indices(xt::where(xt::equal(dinit, dmin)));
            // std::cout << "initialization : indmin = " << indmin << std::endl;
            // std::cout << "initialization : imin = " << indmin(0,0) << " jmin = " << indmin(1,0) << std::endl;
            xt::xtensor_fixed<double, xt::xshape<2>> u0 = {binit(indmin(0,0)), binit(indmin(1,0))};
            // std::cout << "newton_GradF(u0,args) = " << newton_GradF(u0,args) << " newton_F(u0,args) = " << newton_F(u0,args) << std::endl;
            auto u = newton_method(u0,newton_F,newton_GradF,args,200,1.0e-10,1.0e-7);
            xt::view(pts, 0, xt::all()) = s1.point(u(0));
            xt::view(pts, 1, xt::all()) = s2.point(u(1));
            std::cout << "closest_points : pts = " << pts << std::endl;
        }
        else { // dim == 3
          // xt::xtensor_fixed<double, xt::xshape<18>> args = {
          //   s1_pos(0), s1_pos(1), s1_pos(2), s1_rad(0), s1_rad(1), s1_rad(2),
          //   s1_sqr(0), s1_sqr(1),
          //   s1_rot(0,0), s1_rot(1,1),
          //   s1_rot(1,0), s1_rot(0,1),
          //   s2_pos(0), s2_pos(1), s2_pos(2), s2_rad(0), s2_rad(1), s2_rad(2), s2_sqr(0), s2_rot(0,0), s2_rot(1,1), s2_rot(1,0), s2_rot(0,1)
          // };


        }
        return pts;
    }

    // PLAN - PLAN
    template<std::size_t dim>
    auto closest_points(const plan<dim, false>, const plan<dim, false>)
    {
        std::cout << "closest_points : PLAN - PLAN" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // GLOBULE - GLOBULE
    template<std::size_t dim>
    auto closest_points(const globule<dim, false>, const globule<dim, false>)
    {
        std::cout << "closest_points : GLOBULE - GLOBULE" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // SPHERE - GLOBULE
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>, const globule<dim, false>)
    {
        std::cout << "closest_points : SPHERE - GLOBULE" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // SPHERE - PLAN
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>& s, const plan<dim, false>& p)
    {
        std::cout << "closest_points : SPHERE - PLAN" << std::endl;
        xt::xtensor_fixed<double, xt::xshape<2, dim>> pts;
        auto s_pos = s.pos(0);
        auto p_pos = p.pos(0);

        auto normal = p.normal();

        // plan2sphs.n
        auto plan_to_sphere = xt::eval(xt::linalg::dot(s_pos - p_pos, normal));
        auto sign = xt::sign(plan_to_sphere);

        xt::view(pts, 0) = s_pos - sign*s.radius()*normal;
        xt::view(pts, 1) = s_pos - plan_to_sphere*normal;
        return pts;
    }

    // SPHERE - SUPERELLIPSOID
    template<std::size_t dim>
    auto closest_points(const sphere<dim, false>, const superellipsoid<dim, false>)
    {
        std::cout << "closest_points : SPHERE - SUPERELLIPSOID" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // SUPERELLIPSOID - PLAN
    template<std::size_t dim>
    auto closest_points(const superellipsoid<dim, false>, const plan<dim, false>)
    {
        std::cout << "closest_points : SUPERELLIPSOID - PLAN" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // SUPERELLIPSOID - GLOBULE
    template<std::size_t dim>
    auto closest_points(const superellipsoid<dim, false>, const globule<dim, false>)
    {
        std::cout << "closest_points : SUPERELLIPSOID - GLOBULE" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }

    // GLOBULE - PLAN
    template<std::size_t dim>
    auto closest_points(const globule<dim, false>, const plan<dim, false>)
    {
        std::cout << "closest_points : GLOBULE - PLAN" << std::endl;
        return xt::xtensor_fixed<double, xt::xshape<2, dim>>();
    }
}

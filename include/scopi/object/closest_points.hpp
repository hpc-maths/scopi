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
        double pi = 4*std::atan(1);
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
            auto newton_F = [](auto u, auto args)
            {
                double a1 = u(0);
                double b1 = u(1);
                double a2 = u(2);
                double b2 = u(3);
                double s1xc = args(0);
                double s1yc = args(1);
                double s1zc = args(2);
                double s1rx = args(3);
                double s1ry = args(4);
                double s1rz = args(5);
                double s1e = args(6);
                double s1n = args(7);
                double Q1 = args(8);
                double Q2 = args(12);
                double Q3 = args(11);
                double Q4 = args(9);
                double Q5 = args(13);
                double Q6 = args(14);
                double Q7 = args(15);
                double Q8 = args(16);
                double Q9 = args(10);
                double s2xc = args(17);
                double s2yc = args(18);
                double s2zc = args(19);
                double s2rx = args(20);
                double s2ry = args(21);
                double s2rz = args(22);
                double s2e = args(23);
                double s2n = args(24);
                double R1 = args(25);
                double R2 = args(29);
                double R3 = args(28);
                double R4 = args(26);
                double R5 = args(30);
                double R6 = args(31);
                double R7 = args(32);
                double R8 = args(33);
                double R9 = args(27);
                double sb2e1 = std::pow(std::fabs(std::sin(b1)),(2 - s1e)) * sign(std::sin(b1));
                double sbe1  = std::pow(std::fabs(std::sin(b1)),s1e) * sign(std::sin(b1));
                double cb2e1 = std::pow(std::fabs(std::cos(b1)),(2 - s1e)) * sign(std::cos(b1));
                double cbe1  = std::pow(std::fabs(std::cos(b1)),s1e) * sign(std::cos(b1));
                double sb2e2 = std::pow(std::fabs(std::sin(b2)),(2 - s2e)) * sign(std::sin(b2));
                double sbe2  = std::pow(std::fabs(std::sin(b2)),s2e) * sign(std::sin(b2));
                double cb2e2 = std::pow(std::fabs(std::cos(b2)),(2 - s2e)) * sign(std::cos(b2));
                double cbe2  = std::pow(std::fabs(std::cos(b2)),s2e) * sign(std::cos(b2));
                double CA1 = std::pow(std::fabs(std::cos(a1)),s1n) * sign(std::cos(a1));
                double CA2 = std::pow(std::fabs(std::cos(a2)),s2n) * sign(std::cos(a2));
                double CB1 = std::pow(std::fabs(std::cos(b1)),s1e) * sign(std::cos(b1));
                double CB2 = std::pow(std::fabs(std::cos(b2)),s2e) * sign(std::cos(b2));
                double SA1 = std::pow(std::fabs(std::sin(a1)),s1n) * sign(std::sin(a1));
                double SA2 = std::pow(std::fabs(std::sin(a2)),s2n) * sign(std::sin(a2));
                double SB1 = std::pow(std::fabs(std::sin(b1)),s1e) * sign(std::sin(b1));
                double SB2 = std::pow(std::fabs(std::sin(b2)),s2e) * sign(std::sin(b2));
                double H01 = sign(std::cos(a1)) * std::pow(std::sin(a1),2) / SA1;
                double H02 = CA1*CB1;
                double H03 = std::pow(std::sin(b1),2) * std::cos(a1) * std::fabs(std::cos(a1)) / (SB1*CA1);
                double H04 = std::pow(std::cos(b1),2) * std::cos(a1) * std::fabs(std::cos(a1)) / (CB1*CA1);
                double H6 = std::tan(a1) * CB1 * CA1;
                double H7 = std::tan(a1) * SB1 * CA1;
                double H8 = std::cos(b1) * std::sin(b1) * std::cos(a1) * std::fabs(std::cos(a1)) / (CA1*SB1);
                double H9 = SB1 * CA1;
                double H10 = CA2 * CB2;
                double H11 = SB2 * CA2;
                double H12 = std::sin(a1) * std::pow(std::sin(b1),2) * std::fabs(std::cos(a1)) / (SB1*CA1);
                double H13 = std::sin(a1) * std::fabs(std::cos(a1)) * std::pow(std::cos(b1),2) / (CA1*CB1);
                double H14 = std::fabs(std::cos(a1)) * std::sin(a1) / SA1;
                double H15 = SA1 / std::tan(a1);
                double H16 = std::sin(b1) * std::cos(b1) * std::cos(a1) * std::fabs(std::cos(a1)) / (CB1*CA1);
                double H17 = std::pow(std::sin(b2),2) * std::cos(a2) * std::fabs(std::cos(a2)) / (SB2*CA2);
                double H18 = std::pow(std::sin(a2),2) * sign(std::cos(a2)) / SA2;
                double H19 = std::cos(a2) * std::fabs(std::cos(a2)) / CA2;
                double H20 = CA1 * CB1 * std::tan(b1);
                double H21 = SB1 / std::tan(b1);
                double H22 = SB2 * CA2 * std::tan(a2);
                double H23 = CA2 * std::tan(a2);
                double H24 = CA2 * CB2 * std::tan(b2);
                double H25 = SB2 / std::tan(b2);
                double H26 = SA2 / std::tan(a2);
                double H27 = std::fabs(std::cos(a2)) * std::sin(a2) / SA2;
                double H28 = std::cos(b2) * std::sin(b2) / SB2;
                double H29 = std::sin(a2) * std::pow(std::sin(b2),2) * std::fabs(std::cos(a2)) / (SB2*CA2);
                double H30 = std::sin(b2) * std::cos(a2) * std::fabs(std::cos(a2)) * std::cos(b2) / (CA2*CB2);
                double H31 = std::sin(a2) * std::fabs(std::cos(a2)) * std::pow(std::cos(b2),2) / (CA2*CB2);
                double H32 = SB2 * CA2;
                double H33 = std::cos(a2) * std::fabs(std::cos(a2)) * std::pow(std::cos(b2),2) / (CA2*CB2);
                double H34 = std::cos(b2) * std::sin(b2) * std::cos(a2) * std::fabs(std::cos(a2)) / (SB2*CA2);
                double H35 = SB1 * CA1 / std::tan(b1);
                double H36 = CA2 * CB2 * std::tan(a2);
                double H37 = SB2 * CA2 / std::tan(b2);
                xt::xtensor_fixed<double, xt::xshape<4>> res;
                res(0) = -(s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                          (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc) +
                          (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04) *
                          (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc);
                res(1) = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04) *
                         (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc) -
                         (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)*
                         (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc);
                res(2) = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                         (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc) -
                         (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04) *
                         (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc);
                res(3) = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                         (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33) +
                         (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)*
                         (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33) +
                         (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04)*
                         (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33) +
                         std::sqrt(std::pow(s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04, 2) +
                                   std::pow(s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04, 2) +
                                   std::pow(s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04, 2) ) *
                         std::sqrt(std::pow(s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33, 2) +
                                   std::pow(s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33, 2) +
                                   std::pow(s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33, 2) );
                return res;
            };
            auto newton_GradF = [](auto u, auto args)
            {
                double a1 = u(0);
                double b1 = u(1);
                double a2 = u(2);
                double b2 = u(3);
                double s1xc = args(0);
                double s1yc = args(1);
                double s1zc = args(2);
                double s1rx = args(3);
                double s1ry = args(4);
                double s1rz = args(5);
                double s1e = args(6);
                double s1n = args(7);
                double Q1 = args(8);
                double Q2 = args(12);
                double Q3 = args(11);
                double Q4 = args(9);
                double Q5 = args(13);
                double Q6 = args(14);
                double Q7 = args(15);
                double Q8 = args(16);
                double Q9 = args(10);
                double s2xc = args(17);
                double s2yc = args(18);
                double s2zc = args(19);
                double s2rx = args(20);
                double s2ry = args(21);
                double s2rz = args(22);
                double s2e = args(23);
                double s2n = args(24);
                double R1 = args(25);
                double R2 = args(29);
                double R3 = args(28);
                double R4 = args(26);
                double R5 = args(30);
                double R6 = args(31);
                double R7 = args(32);
                double R8 = args(33);
                double R9 = args(27);
                double sb2e1 = std::pow(std::fabs(std::sin(b1)),(2 - s1e)) * sign(std::sin(b1));
                double sbe1  = std::pow(std::fabs(std::sin(b1)),s1e) * sign(std::sin(b1));
                double cb2e1 = std::pow(std::fabs(std::cos(b1)),(2 - s1e)) * sign(std::cos(b1));
                double cbe1  = std::pow(std::fabs(std::cos(b1)),s1e) * sign(std::cos(b1));
                double sb2e2 = std::pow(std::fabs(std::sin(b2)),(2 - s2e)) * sign(std::sin(b2));
                double sbe2  = std::pow(std::fabs(std::sin(b2)),s2e) * sign(std::sin(b2));
                double cb2e2 = std::pow(std::fabs(std::cos(b2)),(2 - s2e)) * sign(std::cos(b2));
                double cbe2  = std::pow(std::fabs(std::cos(b2)),s2e) * sign(std::cos(b2));
                double CA1 = std::pow(std::fabs(std::cos(a1)),s1n) * sign(std::cos(a1));
                double CA2 = std::pow(std::fabs(std::cos(a2)),s2n) * sign(std::cos(a2));
                double CB1 = std::pow(std::fabs(std::cos(b1)),s1e) * sign(std::cos(b1));
                double CB2 = std::pow(std::fabs(std::cos(b2)),s2e) * sign(std::cos(b2));
                double SA1 = std::pow(std::fabs(std::sin(a1)),s1n) * sign(std::sin(a1));
                double SA2 = std::pow(std::fabs(std::sin(a2)),s2n) * sign(std::sin(a2));
                double SB1 = std::pow(std::fabs(std::sin(b1)),s1e) * sign(std::sin(b1));
                double SB2 = std::pow(std::fabs(std::sin(b2)),s2e) * sign(std::sin(b2));
                double H01 = sign(std::cos(a1)) * std::pow(std::sin(a1),2) / SA1;
                double H02 = CA1*CB1;
                double H03 = std::pow(std::sin(b1),2) * std::cos(a1) * std::fabs(std::cos(a1)) / (SB1*CA1);
                double H04 = std::pow(std::cos(b1),2) * std::cos(a1) * std::fabs(std::cos(a1)) / (CB1*CA1);
                double H6 = std::tan(a1) * CB1 * CA1;
                double H7 = std::tan(a1) * SB1 * CA1;
                double H8 = std::cos(b1) * std::sin(b1) * std::cos(a1) * std::fabs(std::cos(a1)) / (CA1*SB1);
                double H9 = SB1 * CA1;
                double H10 = CA2 * CB2;
                double H11 = SB2 * CA2;
                double H12 = std::sin(a1) * std::pow(std::sin(b1),2) * std::fabs(std::cos(a1)) / (SB1*CA1);
                double H13 = std::sin(a1) * std::fabs(std::cos(a1)) * std::pow(std::cos(b1),2) / (CA1*CB1);
                double H14 = std::fabs(std::cos(a1)) * std::sin(a1) / SA1;
                double H15 = SA1 / std::tan(a1);
                double H16 = std::sin(b1) * std::cos(b1) * std::cos(a1) * std::fabs(std::cos(a1)) / (CB1*CA1);
                double H17 = std::pow(std::sin(b2),2) * std::cos(a2) * std::fabs(std::cos(a2)) / (SB2*CA2);
                double H18 = std::pow(std::sin(a2),2) * sign(std::cos(a2)) / SA2;
                double H19 = std::cos(a2) * std::fabs(std::cos(a2)) / CA2;
                double H20 = CA1 * CB1 * std::tan(b1);
                double H21 = SB1 / std::tan(b1);
                double H22 = SB2 * CA2 * std::tan(a2);
                double H23 = CA2 * std::tan(a2);
                double H24 = CA2 * CB2 * std::tan(b2);
                double H25 = SB2 / std::tan(b2);
                double H26 = SA2 / std::tan(a2);
                double H27 = std::fabs(std::cos(a2)) * std::sin(a2) / SA2;
                double H28 = std::cos(b2) * std::sin(b2) / SB2;
                double H29 = std::sin(a2) * std::pow(std::sin(b2),2) * std::fabs(std::cos(a2)) / (SB2*CA2);
                double H30 = std::sin(b2) * std::cos(a2) * std::fabs(std::cos(a2)) * std::cos(b2) / (CA2*CB2);
                double H31 = std::sin(a2) * std::fabs(std::cos(a2)) * std::pow(std::cos(b2),2) / (CA2*CB2);
                double H32 = SB2 * CA2;
                double H33 = std::cos(a2) * std::fabs(std::cos(a2)) * std::pow(std::cos(b2),2) / (CA2*CB2);
                double H34 = std::cos(b2) * std::sin(b2) * std::cos(a2) * std::fabs(std::cos(a2)) / (SB2*CA2);
                double H35 = SB1 * CA1 / std::tan(b1);
                double H36 = CA2 * CB2 * std::tan(a2);
                double H37 = SB2 * CA2 / std::tan(b2);
                xt::xtensor_fixed<double, xt::xshape<4,4>> res;
                res(0,0) = (-s1rx*s1ry*Q5*H01 - s1rx*s1rz*Q2*H03 - s1ry*s1rz*Q3*H04) *
                           (s1n*s1rx*Q6*H6 + s1n*s1ry*Q7*H7 - s1n*s1rz*Q8*H15) +
                           (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04) *
                           (s1n*s1rx*Q3*H6 + s1n*s1ry*Q2*H7 - s1n*s1rz*Q5*H15) +
                           (-s1rx*s1ry*(2 - s1n)*Q5*H14 + s1rx*s1rz*(2 - s1n)*Q2*H12 + s1ry*s1rz*(2 - s1n)*Q3*H13) *
                           (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc) +
                           (s1rx*s1ry*(2 - s1n)*Q8*H14 - s1rx*s1rz*(2 - s1n)*Q7*H12 - s1ry*s1rz*(2 - s1n)*Q6*H13)*
                           (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc);
                res(0,1) = (-s1rx*s1ry*Q5*H01 - s1rx*s1rz*Q2*H03 - s1ry*s1rz*Q3*H04) *
                           (s1e*s1rx*Q6*H20 - s1e*s1ry*Q7*H35) +
                           (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04) *
                           (s1e*s1rx*Q3*H20 - s1e*s1ry*Q2*H35) +
                           (s1rx*s1rz*(2 - s1e)*Q7*H8 - s1ry*s1rz*(2 - s1e)*Q6*H16) *
                           (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc) +
                           (-s1rx*s1rz*(2 - s1e)*Q2*H8 + s1ry*s1rz*(2 - s1e)*Q3*H16) *
                           (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc);
                res(0,2) = (-s1rx*s1ry*Q5*H01 - s1rx*s1rz*Q2*H03 - s1ry*s1rz*Q3*H04) *
                           (-s2n*s2rx*R6*H36 - s2n*s2ry*R7*H22 + s2n*s2rz*R8*H26) +
                           (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04) *
                           (-s2n*s2rx*R3*H36 - s2n*s2ry*R2*H22 + s2n*s2rz*R5*H26);
                res(0,3) = (-s1rx*s1ry*Q5*H01 - s1rx*s1rz*Q2*H03 - s1ry*s1rz*Q3*H04) *
                           (-s2e*s2rx*R6*H24 + s2e*s2ry*R7*H37) +
                           (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04) *
                           (-s2e*s2rx*R3*H24 + s2e*s2ry*R2*H37);
                res(1,0) = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04) *
                           (s1n*s1rx*Q6*H6 + s1n*s1ry*Q7*H7 - s1n*s1rz*Q8*H15) +
                           (-s1rx*s1ry*Q8*H01 - s1rx*s1rz*Q7*H03 - s1ry*s1rz*Q6*H04) *
                           (s1n*s1rx*Q1*H6 + s1n*s1ry*Q4*H7 - s1n*s1rz*Q9*H15) +
                           (s1rx*s1ry*(2 - s1n)*Q9*H14 - s1rx*s1rz*(2 - s1n)*Q4*H12 - s1ry*s1rz*(2 - s1n)*Q1*H13) *
                           (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc) +
                           (-s1rx*s1ry*(2 - s1n)*Q8*H14 + s1rx*s1rz*(2 - s1n)*Q7*H12 + s1ry*s1rz*(2 - s1n)*Q6*H13) *
                           (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc);
                res(1,1) = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04) *
                           (s1e*s1rx*Q6*H20 - s1e*s1ry*Q7*H35) +
                           (-s1rx*s1ry*Q8*H01 - s1rx*s1rz*Q7*H03 - s1ry*s1rz*Q6*H04) *
                           (s1e*s1rx*Q1*H20 - s1e*s1ry*Q4*H35) +
                           (-s1rx*s1rz*(2 - s1e)*Q7*H8 + s1ry*s1rz*(2 - s1e)*Q6*H16) *
                           (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc) +
                           (s1rx*s1rz*(2 - s1e)*Q4*H8 - s1ry*s1rz*(2 - s1e)*Q1*H16) *
                           (-s1rx*Q6*H02 - s1ry*Q7*H9 - s1rz*Q8*SA1 - s1zc + s2rx*R6*H10 + s2ry*R7*H11 + s2rz*R8*SA2 + s2zc);
                res(1,2) = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04)*
                           (-s2n*s2rx*R6*H36 - s2n*s2ry*R7*H22 + s2n*s2rz*R8*H26) +
                           (-s1rx*s1ry*Q8*H01 - s1rx*s1rz*Q7*H03 - s1ry*s1rz*Q6*H04) *
                           (-s2n*s2rx*R1*H36 - s2n*s2ry*R4*H22 + s2n*s2rz*R9*H26);
                res(1,3) = (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04) *
                           (-s2e*s2rx*R6*H24 + s2e*s2ry*R7*H37) +
                           (-s1rx*s1ry*Q8*H01 - s1rx*s1rz*Q7*H03 - s1ry*s1rz*Q6*H04) *
                           (-s2e*s2rx*R1*H24 + s2e*s2ry*R4*H37);
                res(2,0) = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                           (s1n*s1rx*Q1*H6 + s1n*s1ry*Q4*H7 - s1n*s1rz*Q9*H15) +
                           (-s1rx*s1ry*Q9*H01 - s1rx*s1rz*Q4*H03 - s1ry*s1rz*Q1*H04) *
                           (s1n*s1rx*Q3*H6 + s1n*s1ry*Q2*H7 - s1n*s1rz*Q5*H15) +
                           (s1rx*s1ry*(2 - s1n)*Q5*H14 - s1rx*s1rz*(2 - s1n)*Q2*H12 - s1ry*s1rz*(2 - s1n)*Q3*H13) *
                           (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H11 + s2rz*R9*SA2 + s2xc) +
                           (-s1rx*s1ry*(2 - s1n)*Q9*H14 + s1rx*s1rz*(2 - s1n)*Q4*H12 + s1ry*s1rz*(2 - s1n)*Q1*H13) *
                           (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc);
                res(2,1) = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                           (s1e*s1rx*Q1*H20 - s1e*s1ry*Q4*H35) +
                           (-s1rx*s1ry*Q9*H01 - s1rx*s1rz*Q4*H03 - s1ry*s1rz*Q1*H04) *
                           (s1e*s1rx*Q3*H20 - s1e*s1ry*Q2*H35) +
                           (-s1rx*s1rz*(2 - s1e)*Q4*H8 + s1ry*s1rz*(2 - s1e)*Q1*H16) *
                           (-s1rx*Q3*H02 - s1ry*Q2*H9 - s1rz*Q5*SA1 - s1yc + s2rx*R3*H10 + s2ry*R2*H11 + s2rz*R5*SA2 + s2yc) +
                           (s1rx*s1rz*(2 - s1e)*Q2*H8 - s1ry*s1rz*(2 - s1e)*Q3*H16) *
                           (-s1rx*Q1*H02 - s1ry*Q4*H9 - s1rz*Q9*SA1 - s1xc + s2rx*R1*H10 + s2ry*R4*H32 + s2rz*R9*SA2 + s2xc);
                res(2,2) = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                           (-s2n*s2rx*R1*H36 - s2n*s2ry*R4*H22 + s2n*s2rz*R9*H26) +
                           (-s1rx*s1ry*Q9*H01 - s1rx*s1rz*Q4*H03 - s1ry*s1rz*Q1*H04) *
                           (-s2n*s2rx*R3*H36 - s2n*s2ry*R2*H22 + s2n*s2rz*R5*H26);
                res(2,3) = (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                           (-s2e*s2rx*R1*H24 + s2e*s2ry*R4*H37) +
                           (-s1rx*s1ry*Q9*H01 - s1rx*s1rz*Q4*H03 - s1ry*s1rz*Q1*H04) *
                           (-s2e*s2rx*R3*H24 + s2e*s2ry*R2*H37);
                res(3,0) = ( (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                             (2*s1rx*s1ry*(2 - s1n)*Q5*H14 - 2*s1rx*s1rz*(2 - s1n)*Q2*H12 - 2*s1ry*s1rz*(2 - s1n)*Q3*H13)/2 +
                             (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04) *
                             (2*s1rx*s1ry*(2 - s1n)*Q9*H14 - 2*s1rx*s1rz*(2 - s1n)*Q4*H12 - 2*s1ry*s1rz*(2 - s1n)*Q1*H13)/2 +
                             (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04) *
                             (2*s1rx*s1ry*(2 - s1n)*Q8*H14 - 2*s1rx*s1rz*(2 - s1n)*Q7*H12 - 2*s1ry*s1rz*(2 - s1n)*Q6*H13)/2
                           )*std::sqrt(
                             std::pow(s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33, 2) +
                             std::pow(s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33, 2) +
                             std::pow(s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33, 2)
                           ) / std::sqrt(
                             std::pow(s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04, 2) +
                             std::pow(s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04, 2) +
                             std::pow(s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04, 2)
                           ) +
                             (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33) *
                             (s1rx*s1ry*(2 - s1n)*Q5*H14 - s1rx*s1rz*(2 - s1n)*Q2*H12 - s1ry*s1rz*(2 - s1n)*Q3*H13) +
                             (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33) *
                             (s1rx*s1ry*(2 - s1n)*Q9*H14 - s1rx*s1rz*(2 - s1n)*Q4*H12 - s1ry*s1rz*(2 - s1n)*Q1*H13) +
                             (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33) *
                             (s1rx*s1ry*(2 - s1n)*Q8*H14 - s1rx*s1rz*(2 - s1n)*Q7*H12 - s1ry*s1rz*(2 - s1n)*Q6*H13);
                res(3,1) = ( (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                             (2*s1rx*s1rz*(2 - s1e)*Q2*H8 - 2*s1ry*s1rz*(2 - s1e)*Q3*H16)/2 +
                             (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04) *
                             (2*s1rx*s1rz*(2 - s1e)*Q4*H8  - 2*s1ry*s1rz*(2 - s1e)*Q1*H16)/2 +
                             (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04) *
                             (2*s1rx*s1rz*(2 - s1e)*Q7*H8 - 2*s1ry*s1rz*(2 - s1e)*Q6*H16)/2
                           )*std::sqrt(
                             std::pow(s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33, 2) +
                             std::pow(s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33, 2) +
                             std::pow(s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33, 2)
                           )/std::sqrt(
                              std::pow(s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04, 2) +
                              std::pow(s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04, 2) +
                              std::pow(s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04, 2)
                            ) +
                            (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33) *
                            (s1rx*s1rz*(2 - s1e)*Q2*H8 - s1ry*s1rz*(2 - s1e)*Q3*H16) +
                            (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33) *
                            (s1rx*s1rz*(2 - s1e)*Q4*H8 - s1ry*s1rz*(2 - s1e)*Q1*H16) +
                            (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33) *
                            (s1rx*s1rz*(2 - s1e)*Q7*H8 - s1ry*s1rz*(2 - s1e)*Q6*H16);
                res(3,2) = ( (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33) *
                             (2*s2rx*s2ry*(2 - s2n)*R5*H27 - 2*s2rx*s2rz*(2 - s2n)*R2*H29 - 2*s2ry*s2rz*(2 - s2n)*R3*H31)/2 +
                             (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33) *
                             (2*s2rx*s2ry*(2 - s2n)*R9*H27 - 2*s2rx*s2rz*(2 - s2n)*R4*H29 - 2*s2ry*s2rz*(2 - s2n)*R1*H31)/2 +
                             (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33) *
                             (2*s2rx*s2ry*(2 - s2n)*R8*H27 - 2*s2rx*s2rz*(2 - s2n)*R7*H29 - 2*s2ry*s2rz*(2 - s2n)*R6*H31)/2
                           )*std::sqrt(
                             std::pow(s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04, 2) +
                             std::pow(s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04, 2) +
                             std::pow(s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04, 2)
                           )/std::sqrt(
                             std::pow(s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33, 2) +
                             std::pow(s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33, 2) +
                             std::pow(s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33, 2)
                           ) +
                           (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                           (s2rx*s2ry*(2 - s2n)*R5*H27 - s2rx*s2rz*(2 - s2n)*R2*H29 - s2ry*s2rz*(2 - s2n)*R3*H31) +
                           (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04) *
                           (s2rx*s2ry*(2 - s2n)*R9*H27 - s2rx*s2rz*(2 - s2n)*R4*H29 - s2ry*s2rz*(2 - s2n)*R1*H31) +
                           (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04) *
                           (s2rx*s2ry*(2 - s2n)*R8*H27 - s2rx*s2rz*(2 - s2n)*R7*H29 - s2ry*s2rz*(2 - s2n)*R6*H31);
                res(3,3) = (  (s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33) *
                              (2*s2rx*s2rz*(2 - s2e)*R2*H34 - 2*s2ry*s2rz*(2 - s2e)*R3*H30)/2 +
                              (s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33) *
                              (2*s2rx*s2rz*(2 - s2e)*R4*H34 - 2*s2ry*s2rz*(2 - s2e)*R1*H30)/2 +
                              (s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33) *
                              (2*s2rx*s2rz*(2 - s2e)*R7*H34 - 2*s2ry*s2rz*(2 - s2e)*R6*H30)/2
                            )*std::sqrt(
                              std::pow(s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04, 2) +
                              std::pow(s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04, 2) +
                              std::pow(s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04, 2)
                            )/std::sqrt(
                              std::pow(s2rx*s2ry*R5*H18 + s2rx*s2rz*R2*H17 + s2ry*s2rz*R3*H33, 2) +
                              std::pow(s2rx*s2ry*R9*H18 + s2rx*s2rz*R4*H17 + s2ry*s2rz*R1*H33, 2) +
                              std::pow(s2rx*s2ry*R8*H18 + s2rx*s2rz*R7*H17 + s2ry*s2rz*R6*H33, 2)
                            ) +
                            (s1rx*s1ry*Q5*H01 + s1rx*s1rz*Q2*H03 + s1ry*s1rz*Q3*H04) *
                            (s2rx*s2rz*(2 - s2e)*R2*H34 - s2ry*s2rz*(2 - s2e)*R3*H30) +
                            (s1rx*s1ry*Q9*H01 + s1rx*s1rz*Q4*H03 + s1ry*s1rz*Q1*H04) *
                            (s2rx*s2rz*(2 - s2e)*R4*H34 - s2ry*s2rz*(2 - s2e)*R1*H30) +
                            (s1rx*s1ry*Q8*H01 + s1rx*s1rz*Q7*H03 + s1ry*s1rz*Q6*H04) *
                            (s2rx*s2rz*(2 - s2e)*R7*H34 - s2ry*s2rz*(2 - s2e)*R6*H30);
                return res;
            };
            const int num = 10;
            auto ainit = xt::linspace<double>(-pi/2, pi/2, num);
            auto binit = xt::linspace<double>(-pi, pi, num);
            xt::xtensor_fixed<double, xt::xshape<num,num>> dinit;
            for (std::size_t i = 0; i < binit.size(); i++) {
                for (std::size_t j = 0; j < binit.size(); j++) {
                    dinit(i,j) = xt::linalg::norm(s1.point(ainit(i),binit(i))-s2.point(ainit(j),binit(j)),2);
                }
            }
            // std::cout << "initialization : b = " << binit << " distances = " << dinit << std::endl;
            auto dmin = xt::amin(dinit);
            // std::cout << "initialization : dmin = " << dmin << std::endl;
            auto indmin = xt::from_indices(xt::where(xt::equal(dinit, dmin)));
            // std::cout << "initialization : indmin = " << indmin << std::endl;
            // std::cout << "initialization : imin = " << indmin(0,0) << " jmin = " << indmin(1,0) << std::endl;
            xt::xtensor_fixed<double, xt::xshape<4>> u0 = { ainit(indmin(0,0)), binit(indmin(0,0)), ainit(indmin(1,0)), binit(indmin(1,0)) };
            // std::cout << "u0 = "<< u0 << std::endl;
            // std::cout << "newton_GradF(u0,args) = " << newton_GradF(u0,args) << " newton_F(u0,args) = " << newton_F(u0,args) << std::endl;
            auto u = newton_method(u0,newton_F,newton_GradF,args,200,1.0e-10,1.0e-7);
            xt::view(pts, 0, xt::all()) = s1.point(u(0),u(1));
            xt::view(pts, 1, xt::all()) = s2.point(u(2),u(3));
            std::cout << "closest_points : pts = " << pts << std::endl;
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

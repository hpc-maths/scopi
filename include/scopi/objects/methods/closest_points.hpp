#pragma once

#include <cstddef>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xsort.hpp>
#include <xtensor/xio.hpp>

#include "../../minpack.hpp"

#include "../types/sphere.hpp"
#include "../types/superellipsoid.hpp"
#include "../types/globule.hpp"
#include "../types/plan.hpp"
#include "../neighbor.hpp"
#include "../dispatch.hpp"

namespace scopi
{
    // SPHERE - SPHERE
    template<std::size_t dim, bool owner>
    auto closest_points(const sphere<dim, owner>& si, const sphere<dim, owner>& sj)
    {
        // std::cout << "closest_points : SPHERE - SPHERE" << std::endl;
        auto si_pos = xt::view(si.pos(), 0);
        auto sj_pos = xt::view(sj.pos(), 0);
        auto si_to_sj = (sj_pos - si_pos)/xt::linalg::norm(sj_pos - si_pos);

        neighbor<dim> neigh;
        neigh.pi = si_pos + si.radius()*si_to_sj;
        neigh.pj = sj_pos - sj.radius()*si_to_sj;
        neigh.nij = (neigh.pj - sj_pos)/xt::linalg::norm(neigh.pj - sj_pos);
        neigh.dij = xt::linalg::dot(neigh.pi - neigh.pj, neigh.nij)[0];
        return neigh;
    }

    // SUPERELLIPSOID 2D - SUPERELLIPSOID 2D
    template<bool owner>
    auto closest_points(const superellipsoid<2, owner>& s1, const superellipsoid<2, owner>& s2)
    {
        // std::cout << "closest_points : SUPERELLIPSOID - SUPERELLIPSOID" << std::endl;
        neighbor<2> neigh;
        // xt::xtensor_fixed<double, xt::xshape<2*(2*dim+dim-1+dim*dim)>> args = xt::hstack(xt::xtuple(
        xt::xtensor_fixed<double, xt::xshape<18>> args = xt::hstack(xt::xtuple(
          xt::view(s1.pos(), 0), s1.radius(), s1.squareness(), xt::flatten(s1.rotation()),
          xt::view(s2.pos(), 0), s2.radius(), s2.squareness(), xt::flatten(s2.rotation())
        ));

        auto newton_F = [](auto u, auto args)
        {
            double b1 = u(0);
            double b2 = u(1);
            double s1xc = args(0);
            double s1yc = args(1);
            double s1rx = args(2);
            double s1ry = args(3);
            double s1e = args(4);
            double M00 = args(5);
            double M01 = args(6);
            double M10 = args(7);
            double M11 = args(8);
            double s2xc = args(9);
            double s2yc = args(10);
            double s2rx = args(11);
            double s2ry = args(12);
            double s2e = args(13);
            double N00 = args(14);
            double N01 = args(15);
            double N10 = args(16);
            double N11 = args(17);
            double cb1 = std::cos(b1);
            double cb1e = sign(cb1) * std::pow(std::abs(cb1), s1e);
            double cb1e2 = sign(cb1) * std::pow(std::abs(cb1), 2 - s1e);
            double sb1 = std::sin(b1);
            double sb1e = sign(sb1) * std::pow(std::abs(sb1),s1e);
            double sb1e2 = sign(sb1) * std::pow(std::abs(sb1), 2 - s1e);
            double F1 = s1rx * sb1e2;
            double F2 = s1ry * sb1e;
            double F3 = s1ry * cb1e2;
            double F4 = s1rx * cb1e;
            double cb2 = std::cos(b2);
            double cb2e = sign(cb2) * std::pow(std::abs(cb2),s2e);
            double cb2e2 = sign(cb2) * std::pow(std::abs(cb2),2 - s2e);
            double sb2 = std::sin(b2);
            double sb2e = sign(sb2) * std::pow(std::abs(sb2), s2e);
            double sb2e2 = sign(sb2) * std::pow(std::abs(sb2),2 - s2e);
            double F5 = s2rx * sb2e2;
            double F6 = s2ry * sb2e;
            double F7 = s2ry * cb2e2;
            double F8 = s2rx * cb2e;
            xt::xtensor_fixed<double, xt::xshape<2>> res;
            res(0) = -( M01*F1 + M00*F3 ) * ( -M10*F4 - M11*F2 - s1yc + N10*F8 + N11*F6 + s2yc ) +
                      ( M11*F1 + M10*F3 ) * ( -M00*F4 - M01*F2 - s1xc + N00*F8 + N01*F6 + s2xc );
            res(1) =  ( M01*F1 + M00*F3 ) * ( N01*F5 + N00*F7 ) + ( M11*F1 + M10*F3 ) * ( N11*F5 + N10*F7 ) +
                      std::sqrt( std::pow(M01*F1 + M00*F3, 2) + std::pow(M11*F1 + M10*F3, 2) ) * std::sqrt( std::pow(N01*F5 + N00*F7, 2) + std::pow(N11*F5 + N10*F7, 2) );
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
            double M00 = args(5);
            double M01 = args(6);
            double M10 = args(7);
            double M11 = args(8);
            double s2xc = args(9);
            double s2yc = args(10);
            double s2rx = args(11);
            double s2ry = args(12);
            double s2e = args(13);
            double N00 = args(14);
            double N01 = args(15);
            double N10 = args(16);
            double N11 = args(17);
            double cb1 = std::cos(b1);
            double cb1e = sign(cb1) * std::pow(std::abs(cb1), s1e);
            double cb1e2 = sign(cb1) * std::pow(std::abs(cb1), 2 - s1e);
            double sb1 = std::sin(b1);
            double sb1e = sign(sb1) * std::pow(std::abs(sb1),s1e);
            double sb1e2 = sign(sb1) * std::pow(std::abs(sb1), 2 - s1e);
            double F1 = s1rx * sb1e2;
            double F2 = s1ry * sb1e;
            double F3 = s1ry * cb1e2;
            double F4 = s1rx * cb1e;
            double cb2 = std::cos(b2);
            double cb2e = sign(cb2) * std::pow(std::abs(cb2),s2e);
            double cb2e2 = sign(cb2) * std::pow(std::abs(cb2),2 - s2e);
            double sb2 = std::sin(b2);
            double sb2e = sign(sb2) * std::pow(std::abs(sb2), s2e);
            double sb2e2 = sign(sb2) * std::pow(std::abs(sb2),2 - s2e);
            double F5 = s2rx * sb2e2;
            double F6 = s2ry * sb2e;
            double F7 = s2ry * cb2e2;
            double F8 = s2rx * cb2e;
            double F9 = s1e * s1rx * sb1 * std::pow(std::abs(cb1), s1e - 1);
            double F10 = s1e * s1ry * cb1 * std::pow(std::abs(sb1), s1e - 1);
            double F11 = s1ry * (2 - s1e) * sb1 * std::pow(std::abs(cb1), 1 - s1e);
            double F12 = s1rx * (2 - s1e) * cb1 * std::pow(std::abs(sb1), 1 - s1e);
            double F13 = s2e * s2rx * sb2 * std::pow(std::abs(cb2), s2e - 1);
            double F14 = s2e * s2ry * cb2 * std::pow(std::abs(sb2), s2e - 1);
            double F15 = s2ry * (2 - s2e) * sb2 * std::pow(std::abs(cb2), 1 - s2e);
            double F16 = s2rx * (2 - s2e) * cb2 * std::pow(std::abs(sb2), 1 - s2e);
            xt::xtensor_fixed<double, xt::xshape<2,2>> res;
            res(0,0) =  ( -M01*F1 - M00*F3 ) * ( M10*F9 - M11*F10 ) +
                        (  M11*F1 + M10*F3 ) * ( M00*F9 - M01*F10 ) +
                        ( -M01*F12 + M00*F11 ) * ( -M10*F4 - M11*F2 - s1yc + N10*F8 + N11*F6 + s2yc ) +
                        (  M11*F12 - M10*F11 ) * ( -M00*F4 - M01*F2 - s1xc + N00*F8 + N01*F6 + s2xc );
            res(0,1) = ( -M01*F1 - M00*F3 ) * ( -N10*F13 + N11*F14 ) +
                       (  M11*F1 + M10*F3 ) * ( -N00*F13 + N01*F14 );
            res(1,0) =  ( ( M01*F1 + M00*F3 ) * ( M01*F12 - M00*F11 ) +
                          ( M11*F1 + M10*F3) * ( M11*F12 - M10*F11 )
                        ) *
                        std::sqrt( std::pow(N01*F5 + N00*F7, 2) + std::pow(N11*F5 + N10*F7, 2) ) /
                        std::sqrt( std::pow(M01*F1 + M00*F3, 2) + std::pow(M11*F1 + M10*F3, 2) ) +
                        ( N01*F5 + N00*F7 ) * ( M01*F12 - M00*F11 ) +
                        ( N11*F5 + N10*F7 ) * ( M11*F12 - M10*F11 );
            res(1,1) = ( (N01*F5 + N00*F7) * (N01*F16 - N00*F15 ) +
                         (N11*F5 + N10*F7) * (N11*F16 - N10*F15 )
                       ) *
                       std::sqrt( std::pow(M01*F1 + M00*F3, 2) + std::pow( M11*F1 + M10*F3, 2) ) /
                       std::sqrt( std::pow(N01*F5 + N00*F7, 2) + std::pow( N11*F5 + N10*F7, 2) ) +
                       ( M01*F1 + M00*F3 ) * ( N01*F16 - N00*F15 ) + ( M11*F1 + M10*F3 ) * ( N11*F16 - N10*F15 );
            return res;
        };

        int num = 4;
        auto binit1_xy = xt::unique(xt::adapt(s1.binit_xy(num)));
        // std::cout << "binit1_xy = " << binit1_xy << std::endl;
        auto binit2_xy = xt::unique(xt::adapt(s2.binit_xy(num)));
        // std::cout << "binit2_xy = " << binit2_xy << std::endl;

        xt::xtensor<double, 2> distances = xt::zeros<double>({binit1_xy.size(), binit2_xy.size()});
        for (std::size_t i = 0; i < binit1_xy.size(); i++) {
            for (std::size_t j = 0; j < binit2_xy.size(); j++) {
                distances(i,j) = xt::linalg::norm(xt::flatten(s1.point(binit1_xy(i)))-xt::flatten(s2.point(binit2_xy(j))),2) +
                  2*(1+xt::linalg::vdot(s1.normal(binit1_xy(i)),s2.normal(binit2_xy(j))));
            }
        }
        // std::cout << "distances =" << distances << std::endl;
        auto dmin = xt::amin(distances);
        // std::cout << "distance min =" << dmin << std::endl;
        auto indmin = xt::from_indices(xt::where(xt::equal(distances, dmin(0)))); //xt::argmin(distances);
        // std::cout << "indmin =" << indmin << std::endl;
        // std::cout << "indmin(0,0) =" << xt::row(indmin,0)(0) << std::endl;
        // std::cout << "indmin(1,0) =" << xt::row(indmin,1)(0) << std::endl;

        // std::cout << "pt s1 = " << s1.point(binit1_xy(xt::row(indmin,0)(0))) << std::endl;
        // std::cout << "pt s2 = " << s2.point(binit2_xy(xt::row(indmin,1)(0))) << std::endl;
        xt::xtensor_fixed<double, xt::xshape<2>> u0 = {binit1_xy(xt::row(indmin,0)(0)), binit2_xy(xt::row(indmin,1)(0))};
        // std::cout << "u0 =" << u0 << std::endl;
        auto [ xx, info ] = hybrd1(u0,newton_F,newton_GradF,args);
        xt::xtensor_fixed<double, xt::xshape<2>> u;
        for (int j = 0; j < 2; j++ ) {
          u(j) = xx[j];
        }
        // auto [ u, info ] = newton_method(u0,newton_F,newton_GradF,args,2000,1.0e-10,1.0e-7);
        if (info==-1){
          std::cout << "\ns1 : " << std::endl;
          s1.print();
          std::cout << "s2 : " << std::endl;
          s2.print();
          std::cout << "s1.pos = " << s1.pos() << " s1.rotation = " << xt::flatten(s1.rotation()) << std::endl;
          std::cout << "s2.pos = " << s2.pos() << " s2.rotation = " << xt::flatten(s2.rotation()) << std::endl;
          exit(0);
        }
        neigh.pi = s1.point(u(0));
        neigh.pj = s2.point(u(1));
        neigh.nij = s2.normal(u(1));
        // neigh.dij = xt::linalg::dot(neigh.pi - neigh.pj, neigh.nij)[0];
        auto sign = xt::sign(xt::eval(xt::linalg::dot(xt::flatten(neigh.pi) - xt::flatten(neigh.pj), xt::flatten(neigh.nij))));
        neigh.dij = sign(0)*xt::linalg::norm(xt::flatten(neigh.pi) - xt::flatten(neigh.pj), 2);
        // std::cout << "pi = " << neigh.pi << " pj = " << neigh.pj << std::endl;
        // std::cout << "nij = " << neigh.nij << " dij = " << neigh.dij << std::endl;

        return neigh;
    }

    // SUPERELLIPSOID 3D - SUPERELLIPSOID 3D
    template<bool owner>
    auto closest_points(const superellipsoid<3, owner>& s1, const superellipsoid<3, owner>& s2)
    {
        // std::cout << "closest_points : SUPERELLIPSOID - SUPERELLIPSOID" << std::endl;
        neighbor<3> neigh;
        // xt::xtensor_fixed<double, xt::xshape<2*(2*dim+dim-1+dim*dim)>> args = xt::hstack(xt::xtuple(
        xt::xtensor_fixed<double, xt::xshape<34>> args = xt::hstack(xt::xtuple(
            xt::view(s1.pos(), 0), s1.radius(), s1.squareness(), xt::flatten(s1.rotation()),
          xt::view(s2.pos(), 0), s2.radius(), s2.squareness(), xt::flatten(s2.rotation())
        ));
        // s1.print();
        // s2.print();
        // std::cout << "s1.pos = " << s1.pos() << " s1.rotation = " << xt::flatten(s1.rotation()) << std::endl;
        // std::cout << "s2.pos = " << s2.pos() << " s2.rotation = " << xt::flatten(s2.rotation()) << std::endl;
        // std::cout << "args = " << args << std::endl;
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
            double M00 = args(8);
            double M01 = args(9);
            double M02 = args(10);
            double M10 = args(11);
            double M11 = args(12);
            double M12 = args(13);
            double M20 = args(14);
            double M21 = args(15);
            double M22 = args(16);
            double s2xc = args(17);
            double s2yc = args(18);
            double s2zc = args(19);
            double s2rx = args(20);
            double s2ry = args(21);
            double s2rz = args(22);
            double s2e = args(23);
            double s2n = args(24);
            double N00 = args(25);
            double N01 = args(26);
            double N02 = args(27);
            double N10 = args(28);
            double N11 = args(29);
            double N12 = args(30);
            double N20 = args(31);
            double N21 = args(32);
            double N22 = args(33);

            double ca1 = std::cos(a1);
            double ca1n = sign(ca1) * std::pow(std::abs(ca1), s1n);
            double ca1n2 = sign(ca1) * std::pow(std::abs(ca1), 2 - s1n);

            double sa1 = std::sin(a1);
            double sa1n = sign(sa1) * std::pow(std::abs(sa1), s1n);
            double sa1n2 = sign(sa1) * std::pow(std::abs(sa1), 2 - s1n);

            double cb1 = std::cos(b1);
            double cb1e = sign(cb1) * std::pow(std::abs(cb1), s1e);
            double cb1e2 = sign(cb1) * std::pow(std::abs(cb1), 2 - s1e);

            double sb1 = std::sin(b1);
            double sb1e = sign(sb1) * std::pow(std::abs(sb1), s1e);
            double sb1e2 = sign(sb1) * std::pow(std::abs(sb1), 2 - s1e);

            double A1 = s1ry * s1rz * ca1n2 * cb1e2;
            double A2 = s1rx * s1rz * sb1e2 * ca1n2;
            double A3 = s1rx * s1ry * sa1n2;
            double A4 = s1rx * ca1n  * cb1e;
            double A5 = s1ry * sb1e  * ca1n;
            double A6 = s1rz * sa1n;

            double ca2 = std::cos(a2);
            double ca2n = sign(ca2) * std::pow(std::abs(ca2), s2n);
            double ca2n2 = sign(ca2) * std::pow(std::abs(ca2), 2 - s2n);

            double sa2 = std::sin(a2);
            double sa2n = sign(sa2) * std::pow(std::abs(sa2), s2n);
            double sa2n2 = sign(sa2) * std::pow(std::abs(sa2), 2 - s2n);

            double cb2 = std::cos(b2);
            double cb2e = sign(cb2) * std::pow(std::abs(cb2), s2e);
            double cb2e2 = sign(cb2) * std::pow(std::abs(cb2), 2 - s2e);

            double sb2 = std::sin(b2);
            double sb2e = sign(sb2) * std::pow(std::abs(sb2), s2e);
            double sb2e2 = sign(sb2) * std::pow(std::abs(sb2), 2 - s2e);

            double B1 = s2rx * ca2n *cb2e;
            double B2 = s2ry * sb2e *ca2n;
            double B3 = s2rz * sa2n;
            double B4 = s2ry * s2rz * ca2n2 * cb2e2;
            double B5 = s2rx * s2rz * sb2e2 * ca2n2;
            double B6 = s2rx * s2ry * sa2n2;

            xt::xtensor_fixed<double, xt::xshape<4>> res;
            res(0) = - (M10*A1 + M11*A2 + M12*A3) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) +
                       (M20*A1 + M21*A2 + M22*A3) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc);
            res(1) =   (M00*A1 + M01*A2 + M02*A3) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) -
                       (M20*A1 + M21*A2 + M22*A3) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);
            res(2) = - (M00*A1 + M01*A2 + M02*A3) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc) +
                       (M10*A1 + M11*A2 + M12*A3) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);
            res(3) = (M00*A1 + M01*A2 + M02*A3) * (N00*B4 + N01*B5 + N02*B6) +
                     (M10*A1 + M11*A2 + M12*A3) * (N10*B4 + N11*B5 + N12*B6) +
                     (M20*A1 + M21*A2 + M22*A3) * (N20*B4 + N21*B5 + N22*B6) +
                     std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) ) *
                     std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) );
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
            double M00 = args(8);
            double M01 = args(9);
            double M02 = args(10);
            double M10 = args(11);
            double M11 = args(12);
            double M12 = args(13);
            double M20 = args(14);
            double M21 = args(15);
            double M22 = args(16);
            double s2xc = args(17);
            double s2yc = args(18);
            double s2zc = args(19);
            double s2rx = args(20);
            double s2ry = args(21);
            double s2rz = args(22);
            double s2e = args(23);
            double s2n = args(24);
            double N00 = args(25);
            double N01 = args(26);
            double N02 = args(27);
            double N10 = args(28);
            double N11 = args(29);
            double N12 = args(30);
            double N20 = args(31);
            double N21 = args(32);
            double N22 = args(33);

            double ca1 = std::cos(a1);
            double ca1n = sign(ca1) * std::pow(std::abs(ca1), s1n);
            double ca1n1 = s1n * std::pow(std::abs(ca1), s1n - 1);
            double ca1n2 = sign(ca1) * std::pow(std::abs(ca1), 2 - s1n);
            double ca1n3 = (2 - s1n) * std::pow(std::abs(ca1), 1 - s1n);

            double sa1 = std::sin(a1);
            double sa1n = sign(sa1) * std::pow(std::abs(sa1), s1n);
            double sa1n1 = s1n * std::pow(std::abs(sa1), s1n - 1);
            double sa1n2 = sign(sa1) * std::pow(std::abs(sa1), 2 - s1n);
            double sa1n3 = (2 - s1n) * std::pow(std::abs(sa1), 1 - s1n);

            double cb1 = std::cos(b1);
            double cb1e = sign(cb1) * std::pow(std::abs(cb1), s1e);
            double cb1e1 = s1e * std::pow(std::abs(cb1), s1e - 1);
            double cb1e2 = sign(cb1) * std::pow(std::abs(cb1), 2 - s1e);
            double cb1e3 = (2 - s1e) * std::pow(std::abs(cb1), 1 - s1e);

            double sb1 = std::sin(b1);
            double sb1e = sign(sb1) * std::pow(std::abs(sb1), s1e);
            double sb1e1 = s1e * std::pow(std::abs(sb1), s1e - 1);
            double sb1e2 = sign(sb1) * std::pow(std::abs(sb1), 2 - s1e);
            double sb1e3 = (2 - s1e) * std::pow(std::abs(sb1), 1 - s1e);

            double A1 = s1ry * s1rz * ca1n2 * cb1e2;
            double A2 = s1rx * s1rz * sb1e2 * ca1n2;
            double A3 = s1rx * s1ry * sa1n2;
            double A4 = s1rx * ca1n  * cb1e;
            double A5 = s1ry * sb1e  * ca1n;
            double A6 = s1rz * sa1n;
            double A7 = s1rx * sa1 * ca1n1 * cb1e;
            double A8 = s1ry * sa1 * sb1e  * ca1n1;
            double A9 = s1rz * ca1 * sa1n1;
            double A10 = s1ry * s1rz * sa1 * ca1n3 * cb1e2;
            double A11 = s1rx * s1rz * sa1 * sb1e2 * ca1n3;
            double A12 = s1rx * s1ry * ca1 * sa1n3;
            double A13 = s1rx * sb1 * ca1n  * cb1e1;
            double A14 = s1ry * cb1 * sb1e1 * ca1n;
            double A15 = s1ry * s1rz  * sb1 * ca1n2 * cb1e3;
            double A16 = s1rx * s1rz  * cb1 * sb1e3 * ca1n2;

            double ca2 = std::cos(a2);
            double ca2n = sign(ca2) * std::pow(std::abs(ca2), s2n);
            double ca2n1 = s2n * std::pow(std::abs(ca2), s2n - 1);
            double ca2n2 = sign(ca2) * std::pow(std::abs(ca2), 2 - s2n);
            double ca2n3 = (2 - s2n) * std::pow(std::abs(ca2), 1 - s2n);

            double sa2 = std::sin(a2);
            double sa2n = sign(sa2) * std::pow(std::abs(sa2), s2n);
            double sa2n1 = s2n * std::pow(std::abs(sa2), s2n - 1);
            double sa2n2 = sign(sa2) * std::pow(std::abs(sa2), 2 - s2n);
            double sa2n3 = (2 - s2n) * std::pow(std::abs(sa2), 1 - s2n);

            double cb2 = std::cos(b2);
            double cb2e = sign(cb2) * std::pow(std::abs(cb2), s2e);
            double cb2e1 = s2e * std::pow(std::abs(cb2), s2e - 1);
            double cb2e2 = sign(cb2) * std::pow(std::abs(cb2), 2 - s2e);
            double cb2e3 = (2 - s2e) * std::pow(std::abs(cb2), 1 - s2e);

            double sb2 = std::sin(b2);
            double sb2e = sign(sb2) * std::pow(std::abs(sb2), s2e);
            double sb2e1 = s2e * std::pow(std::abs(sb2), s2e - 1);
            double sb2e2 = sign(sb2) * std::pow(std::abs(sb2), 2 - s2e);
            double sb2e3 = (2 - s2e) * std::pow(std::abs(sb2), 1 - s2e);

            double B1 = s2rx * ca2n * cb2e;
            double B2 = s2ry * sb2e * ca2n;
            double B3 = s2rz * sa2n;
            double B4 = s2ry * s2rz * ca2n2 * cb2e2;
            double B5 = s2rx * s2rz * sb2e2 * ca2n2;
            double B6 = s2rx * s2ry * sa2n2;
            double B7 = s2rx * sa2 * ca2n1 * cb2e;
            double B8 = s2ry * sa2 * sb2e * ca2n1;
            double B9 = s2rz * ca2 * sa2n1;
            double B10 = s2rx * sb2 * ca2n * cb2e1;
            double B11 = s2ry * cb2 * sb2e1 * ca2n;
            double B12 = s2ry * s2rz * sa2 * ca2n3 * cb2e2;
            double B13 = s2rx * s2rz * sa2 * sb2e2 * ca2n3;
            double B14 = s2rx * s2ry * ca2 * sa2n3;
            double B15 = s2ry * s2rz * sb2 * ca2n2 * cb2e3;
            double B16 = s2rx * s2rz * cb2 * sb2e3 * ca2n2;
            xt::xtensor_fixed<double, xt::xshape<4,4>> res;
            res(0,0) =   (-M10*A1 - M11*A2 - M12*A3) * (M20*A7 + M21*A8 - M22*A9) +
                         ( M20*A1 + M21*A2 + M22*A3) * (M10*A7 + M11*A8 - M12*A9) +
                         ( M10*A10 + M11*A11 - M12*A12) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) +
                         (-M20*A10 - M21*A11 + M22*A12) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc);

            res(0,1) =   (-M10*A1 - M11*A2 - M12*A3) * (M20*A13 - M21*A14) +
                         ( M20*A1 + M21*A2 + M22*A3) * (M10*A13 - M11*A14) +
                         ( M10*A15 - M11*A16) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) +
                         (-M20*A15 + M21*A16)*(-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc);

            res(0,2) =   (-M10*A1 - M11*A2 - M12*A3) * (-N20*B7 - N21*B8 + N22*B9) +
                         ( M20*A1 + M21*A2 + M22*A3) * (-N10*B7 - N11*B8 + N12*B9);

            res(0,3) =   (-M10*A1 - M11*A2 - M12*A3) * (-N20*B10 + N21*B11) +
                         ( M20*A1 + M21*A2 + M22*A3) * (-N10*B10 + N11*B11);

            res(1,0) =   ( M00*A1 + M01*A2 + M02*A3) * (M20*A7 + M21*A8 - M22*A9) +
                         (-M20*A1 - M21*A2 - M22*A3) * (M00*A7 + M01*A8 - M02*A9) +
                         (-M00*A10 - M01*A11 + M02*A12) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) +
                         ( M20*A10 + M21*A11 - M22*A12) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);

            res(1,1) =   ( M00*A1 + M01*A2 + M02*A3) * (M20*A13 - M21*A14) +
                         (-M20*A1 - M21*A2 - M22*A3) * (M00*A13 - M01*A14) +
                         (-M00*A15 + M01*A16) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc) +
                         ( M20*A15 - M21*A16) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);

            res(1,2) =   ( M00*A1 + M01*A2 + M02*A3) * (-N20*B7 - N21*B8 + N22*B9) +
                         (-M20*A1 - M21*A2 - M22*A3) * (-N00*B7 - N01*B8 + N02*B9);

            res(1,3) =   ( M00*A1 + M01*A2 + M02*A3) * (-N20*B10 + N21*B11) +
                         (-M20*A1 - M21*A2 - M22*A3) * (-N00*B10 + N01*B11);

            res(2,0) =   (-M00*A1 - M01*A2 - M02*A3) * (M10*A7 + M11*A8 - M12*A9) +
                         ( M10*A1 + M11*A2 + M12*A3) * (M00*A7 + M01*A8 - M02*A9) +
                         ( M00*A10 + M01*A11 - M02*A12) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc) +
                         (-M10*A10 - M11*A11 + M12*A12) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);

            res(2,1) =   (-M00*A1 - M01*A2 - M02*A3) * (M10*A13 - M11*A14) +
                         ( M10*A1 + M11*A2 + M12*A3) * (M00*A13 - M01*A14) +
                         ( M00*A15 - M01*A16) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc) +
                         (-M10*A15 + M11*A16) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);

            res(2,2) =   (-M00*A1 - M01*A2 - M02*A3) * (-N10*B7 - N11*B8 + N12*B9) +
                         (M10*A1 + M11*A2 + M12*A3) * (-N00*B7 - N01*B8 + N02*B9);

            res(2,3) =   (-M00*A1 - M01*A2 - M02*A3) * (-N10*B10 + N11*B11) +
                         (M10*A1 + M11*A2 + M12*A3) * (-N00*B10 + N01*B11);

            res(3,0) = ( ( M00*A1 + M01*A2 + M02*A3) * (-M00*A10 - M01*A11 + M02*A12) +
                        ( M10*A1 + M11*A2 + M12*A3) * (-M10*A10 - M11*A11 + M12*A12) +
                        ( M20*A1 + M21*A2 + M22*A3) * (-M20*A10 - M21*A11 + M22*A12)
                      ) *
                       std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) ) /
                       std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) ) +
                       ( N00*B4 + N01*B5 + N02*B6) * ( -M00*A10 - M01*A11 + M02*A12) +
                       ( N10*B4 + N11*B5 + N12*B6) * (-M10*A10 - M11*A11 + M12*A12) +
                       ( N20*B4 + N21*B5 + N22*B6) * (-M20*A10 - M21*A11 + M22*A12);

            res(3,1) = ( ( M00*A1 + M01*A2 + M02*A3) * (-M00*A15 + M01*A16) +
                         ( M10*A1 + M11*A2 + M12*A3) * (-M10*A15 + M11*A16) +
                         ( M20*A1 + M21*A2 + M22*A3) * (-M20*A15 + M21*A16)
                       ) *
                       std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) ) /
                       std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) ) +
                       ( N00*B4 + N01*B5 + N02*B6) * (-M00*A15 + M01*A16 ) +
                       ( N10*B4 + N11*B5 + N12*B6) * (-M10*A15 + M11*A16) +
                       ( N20*B4 + N21*B5 + N22*B6) * (-M20*A15 + M21*A16);

            res(3,2) = (  ( N00*B4 + N01*B5 + N02*B6) * (-N00*B12 - N01*B13 + N02*B14) +
                          ( N10*B4 + N11*B5 + N12*B6) * (-N10*B12 - N11*B13 + N12*B14) +
                          ( N20*B4 + N21*B5 + N22*B6) * (-N20*B12 - N21*B13 + N22*B14)
                        ) *
                        std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) ) /
                        std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) ) +
                        (M00*A1 + M01*A2 + M02*A3) * (-N00*B12 - N01*B13 + N02*B14) +
                        (M10*A1 + M11*A2 + M12*A3) * (-N10*B12 - N11*B13 + N12*B14) +
                        (M20*A1 + M21*A2 + M22*A3) * (-N20*B12 - N21*B13 + N22*B14);

            res(3,3) = (  (N00*B4 + N01*B5 + N02*B6) * (-N00*B15 + N01*B16) +
                          (N10*B4 + N11*B5 + N12*B6) * (-N10*B15 + N11*B16) +
                          (N20*B4 + N21*B5 + N22*B6) * (-N20*B15 + N21*B16)
                       ) *
                       std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) ) /
                       std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) ) +
                       (M00*A1 + M01*A2 + M02*A3) * (-N00*B15 + N01*B16) +
                       (M10*A1 + M11*A2 + M12*A3) * (-N10*B15 + N11*B16) +
                       (M20*A1 + M21*A2 + M22*A3) * (-N20*B15 + N21*B16);
            return res;
        };
        const int num = 6;
        auto binit1 = xt::unique(xt::adapt(s1.binit_xy(num)));
        auto ainit1 = xt::unique(xt::adapt(s1.ainit_yz(num)));

        // auto binit1_xy = xt::unique(xt::adapt(s1.binit_xy(num)));
        // std::cout << " binit1_xy = " << binit1_xy << std::endl;
        // auto ainit1_yz = xt::unique(xt::adapt(s1.ainit_yz(num)));
        // std::cout << " ainit1_yz = " << ainit1_yz << std::endl;
        // auto ainit1_xz = xt::unique(xt::adapt(s1.ainit_xz(num)));
        // std::cout << " ainit1_xz = " << ainit1_xz << std::endl;
        // // exit(0);
        // // std::cout << " binit1_xy.size() = " << binit1_xy.size() << " ainit1_yz.size() = " << ainit1_yz.size() << " ainit1_xz.size() = " << ainit1_xz.size() << std::endl;
        // auto binit1 = xt::concatenate(xt::xtuple(binit1_xy,binit1_xy));
        // auto ainit1 = xt::concatenate(xt::xtuple(ainit1_yz,ainit1_xz));
        // std::cout << " ainit1.size() = " << ainit1.size() << " binit1.size() = " << binit1.size() << std::endl;
        auto [ agrid1, bgrid1 ] = xt::meshgrid(ainit1, binit1);
        auto fl_agrid1 = xt::flatten(agrid1);
        auto fl_bgrid1 = xt::flatten(bgrid1);
        // std::cout << "agrid1 = " << agrid1 << "bgrid1 = " << bgrid1 << std::endl;
        // std::cout << "flatten agrid1 =" << fl_agrid1 << std::endl;
        // std::cout << "flatten bgrid1 =" << fl_bgrid1 << std::endl;

        auto binit2 = xt::unique(xt::adapt(s2.binit_xy(num)));
        auto ainit2 = xt::unique(xt::adapt(s2.ainit_yz(num)));
        // auto binit2_xy = xt::unique(xt::adapt(s2.binit_xy(num)));
        // std::cout << " binit2_xy = " << binit2_xy << std::endl;
        // auto ainit2_yz = xt::unique(xt::adapt(s2.ainit_yz(num)));
        // std::cout << " ainit2_yz = " << ainit2_yz << std::endl;
        // auto ainit2_xz = xt::unique(xt::adapt(s2.ainit_xz(num)));
        // std::cout << " ainit2_xz = " << ainit2_xz << std::endl;
        // // std::cout << " binit2_xy.size() = " << binit2_xy.size() << " ainit2_yz.size() = " << ainit2_yz.size() << " ainit2_xz.size() = " << ainit2_xz.size() << std::endl;
        // auto binit2 = xt::concatenate(xt::xtuple(binit2_xy,binit2_xy));
        // auto ainit2 = xt::concatenate(xt::xtuple(ainit2_yz,ainit2_xz));
        // std::cout << " ainit2.size() = " << ainit2.size() << " binit2.size() = " << binit2.size() << std::endl;
        auto [ agrid2, bgrid2 ] = xt::meshgrid(ainit2, binit2);
        // std::cout << "agrid2 = " << agrid2 << "bgrid2 = " << bgrid2 << std::endl;

        auto fl_agrid2 = xt::flatten(agrid2);
        auto fl_bgrid2 = xt::flatten(bgrid2);
        // std::cout << "flatten agrid2 =" << fl_agrid2 << std::endl;
        // std::cout << "flatten bgrid2 =" << fl_bgrid2 << std::endl;
        //
        // std::cout << " fl_agrid1.size() = " << fl_agrid1.size() << " fl_agrid2.size() = " << fl_agrid2.size() << std::endl;
        xt::xtensor<double, 2> distances = xt::zeros<double>({fl_agrid1.size(), fl_agrid2.size()});
        for (std::size_t i = 0; i < fl_agrid1.size(); i++) {
            for (std::size_t j = 0; j < fl_agrid2.size(); j++) {
                distances(i,j) = xt::linalg::norm(xt::flatten(s1.point(fl_agrid1(i),fl_bgrid1(i)))-xt::flatten(s2.point(fl_agrid2(j),fl_bgrid2(j))),2) +
                2*(1+xt::linalg::vdot(s1.normal(fl_agrid1(i),fl_bgrid1(i)),s2.normal(fl_agrid2(j),fl_bgrid2(j))));
            }
        }
        // std::cout << "distances =" << distances << std::endl;
        auto dmin = xt::amin(distances);
        // std::cout << "distance min =" << dmin << std::endl;
        auto indmin = xt::from_indices(xt::where(xt::equal(distances, dmin(0)))); //xt::argmin(distances);
        // std::cout << "indmin =" << indmin << std::endl;
        // std::cout << "indmin(0,0) =" << xt::row(indmin,0)(0) << std::endl;
        // std::cout << "indmin(1,0) =" << xt::row(indmin,1)(0) << std::endl;

        // for (int l=0; xt::row(indmin,0).size(); ++l){
        //     std::cout << "l =" << l << " pt s1 = " << s1.point(fl_agrid1(xt::row(indmin,0)(l)),fl_bgrid1(xt::row(indmin,0)(l))) << std::endl;
        //     std::cout << "l =" << l << " pt s2 = " << s2.point(fl_agrid1(xt::row(indmin,1)(l)),fl_bgrid1(xt::row(indmin,1)(l))) << std::endl;
        // }
        // std::cout << "pt s1 = " << s1.point(fl_agrid1(xt::row(indmin,0)(0)),fl_bgrid1(xt::row(indmin,0)(0))) << std::endl;
        // std::cout << "pt s2 = " << s2.point(fl_agrid2(xt::row(indmin,1)(0)),fl_bgrid2(xt::row(indmin,1)(0))) << std::endl;
        // xt::xtensor_fixed<double, xt::xshape<4>> u0 = {fl_agrid1(iindmin(0,0)),fl_bgrid1(indmin(0,1)),
        //                                                fl_agrid2(indmin(1,0)),fl_bgrid2(indmin(1,1))};
        xt::xtensor_fixed<double, xt::xshape<4>> u0 = {fl_agrid1(xt::row(indmin,0)(0)),fl_bgrid1(xt::row(indmin,0)(0)),
                                                       fl_agrid2(xt::row(indmin,1)(0)),fl_bgrid2(xt::row(indmin,1)(0))};
        // std::cout << "u0 =" << u0 << std::endl;

        // exit(0);
        // auto ainit = xt::linspace<double>(-pi/2, pi/2, num);
        // auto binit = xt::linspace<double>(-pi, pi, num);
        // xt::xtensor_fixed<double, xt::xshape<num,num>> dinit;
        // for (std::size_t i = 0; i < binit.size(); i++) {
        //   for (std::size_t j = 0; j < binit.size(); j++) {
        //     dinit(i,j) = xt::linalg::norm(s1.point(ainit(i),binit(i))-s2.point(ainit(j),binit(j)),2);
        //   }
        // }
        // // std::cout << "initialization : b = " << binit << " distances = " << dinit << std::endl;
        // auto dmin = xt::amin(dinit);
        // // std::cout << "initialization : dmin = " << dmin << std::endl;
        // auto indmin = xt::from_indices(xt::where(xt::equal(dinit, dmin)));
        // // std::cout << "initialization : indmin = " << indmin << std::endl;
        // // std::cout << "initialization : imin = " << indmin(0,0) << " jmin = " << indmin(1,0) << std::endl;
        // xt::xtensor_fixed<double, xt::xshape<4>> u0 = { ainit(indmin(0,0)), binit(indmin(0,0)), ainit(indmin(1,0)), binit(indmin(1,0)) };
        // std::cout << "u0 = "<< u0 << std::endl;
        // std::cout << "newton_GradF(u0,args) = " << newton_GradF(u0,args) << " newton_F(u0,args) = " << newton_F(u0,args) << std::endl;
        auto [ xx, info ] = hybrd1(u0,newton_F,newton_GradF,args);
        xt::xtensor_fixed<double, xt::xshape<4>> u;
        for (int j = 0; j < 4; j++ ) {
          u(j) = xx[j];
        }
        // std::cout << " u hybr = " << u << std::endl;
        // std::cout << " info hybr = " << info << std::endl;
        // auto [ ubis, infobis ] = newton_method(u0,newton_F,newton_GradF,args,4000,1.0e-8,1.0e-7); //itermax,  ftol,  xtol
        // std::cout << " u newton = " << ubis << std::endl;
        // std::cout << " info newton = " << infobis << std::endl;

        // if (info==-1){
          // std::cout << "\ns1 : " << std::endl;
          // s1.print();
          // std::cout << "s2 : " << std::endl;
          // s2.print();
          // std::cout << "s1.pos = " << s1.pos() << " s1.rotation = " << xt::flatten(s1.rotation()) << std::endl;
          // std::cout << "s2.pos = " << s2.pos() << " s2.rotation = " << xt::flatten(s2.rotation()) << std::endl;
          // exit(0);
        // }
        if (info==-1){
          exit(0);
        }
        neigh.pi = s1.point(u(0),u(1));
        neigh.pj = s2.point(u(2),u(3));
        neigh.nij = s2.normal(u(2),u(3));
        auto sign = xt::sign(xt::eval(xt::linalg::dot(xt::flatten(neigh.pi) - xt::flatten(neigh.pj), xt::flatten(neigh.nij))));
        neigh.dij = sign(0)*xt::linalg::norm(xt::flatten(neigh.pi) - xt::flatten(neigh.pj), 2);
        // std::cout << "pi = " << neigh.pi << " pj = " << neigh.pj << std::endl;
        // std::cout << "nij = " << neigh.nij << " dij = " << neigh.dij << std::endl;
        // std::cout << "dij = " << neigh.dij << std::endl;
        return neigh;
    }

    // PLAN - PLAN
    template<std::size_t dim, bool owner>
    auto closest_points(const plan<dim, owner>, const plan<dim, owner>)
    {
        return neighbor<dim>();
    }

    // GLOBULE - GLOBULE
    template<std::size_t dim, bool owner>
    auto closest_points(const globule<dim, owner> gi, const globule<dim, owner> gj)
    {
        std::vector<neighbor<dim>> neigh(gi.size()*gj.size());
        for (std::size_t i = 0; i < gi.size(); ++i)
        {
            for (std::size_t j = 0; j < gj.size(); ++j)
            {
                // auto si_pos = xt::view(gi.pos(i), 0);
                // auto sj_pos = xt::view(gj.pos(j), 0);
                auto si_pos = gi.pos(i);
                auto sj_pos = gj.pos(j);
                auto si_to_sj = (sj_pos - si_pos)/xt::linalg::norm(sj_pos - si_pos);

                neigh[gj.size()*i + j].pi = si_pos + gi.radius()*si_to_sj;
                neigh[gj.size()*i + j].pj = sj_pos - gj.radius()*si_to_sj;
                neigh[gj.size()*i + j].nij = (neigh[gj.size()*i + j].pj - sj_pos)/xt::linalg::norm(neigh[gj.size()*i + j].pj - sj_pos);
                neigh[gj.size()*i + j].dij = xt::linalg::dot(neigh[gj.size()*i + j].pi - neigh[gj.size()*i + j].pj, neigh[gj.size()*i + j].nij)[0];
            }
        }
        return neigh;
    }

    // SPHERE - GLOBULE
    template<std::size_t dim, bool owner>
    auto closest_points(const sphere<dim, owner>, const globule<dim, owner>)
    {
        return neighbor<dim>();
    }

    // GLOBULE - SPHERE
    template<std::size_t dim, bool owner>
    auto closest_points(const globule<dim, owner>, const sphere<dim, owner>)
    {
        return neighbor<dim>();
    }

    // SPHERE - PLAN
    template<std::size_t dim, bool owner>
    auto closest_points(const sphere<dim, owner>& s, const plan<dim, owner>& p)
    {
        // std::cout << "closest_points : SPHERE - PLAN" << std::endl;
        auto s_pos = s.pos(0);
        auto p_pos = p.pos(0);

        auto normal = p.normal();

        // plan2sphs.n
        auto plan_to_sphere = xt::eval(xt::linalg::dot(s_pos - p_pos, normal));
        auto sign = xt::sign(plan_to_sphere);

        neighbor<dim> neigh;
        neigh.pi = s_pos - sign*s.radius()*normal;
        neigh.pj = s_pos - plan_to_sphere*normal;
        neigh.nij = sign*normal;
        neigh.dij = xt::linalg::dot(neigh.pi - neigh.pj, neigh.nij)[0];
        return neigh;
    }

    // PLAN - SPHERE
    template<std::size_t dim, bool owner>
    auto closest_points(const plan<dim, owner>& p, const sphere<dim, owner>& s)
    {
        auto neigh = closest_points(s, p);
        neigh.nij *= -1.;
        std::swap(neigh.pi, neigh.pj);
        return neigh;
    }

    // SUPERELLIPSOID 3D - SPHERE 3D
    template<bool owner>
    auto closest_points(const sphere<3, owner>& s2, const superellipsoid<3, owner>& s1)
    {
        std::cout << "closest_points : SPHERE 3D - SUPERELLIPSOID 3D" << std::endl;
        double pi = 4*std::atan(1);
        neighbor<3> neigh;
        // 2*dim (pos) + 1+3 (r) + 2 (e et n) + 2*dim*dim (rot) = 30
        xt::xtensor<double, 1> xts2r = { s2.radius() };
        xt::xtensor_fixed<double, xt::xshape<30>> args = xt::hstack(xt::xtuple(
            xt::view(s1.pos(), 0), s1.radius(), s1.squareness(), xt::flatten(s1.rotation()),
            xt::view(s2.pos(), 0), xts2r, xt::flatten(s2.rotation())
        ));
        // std::cout << "args = " << args << std::endl;
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
            double M00 = args(8);
            double M01 = args(9);
            double M02 = args(10);
            double M10 = args(11);
            double M11 = args(12);
            double M12 = args(13);
            double M20 = args(14);
            double M21 = args(15);
            double M22 = args(16);
            double s2xc = args(17);
            double s2yc = args(18);
            double s2zc = args(19);
            double s2r = args(20);
            double N00 = args(21);
            double N01 = args(22);
            double N02 = args(23);
            double N10 = args(24);
            double N11 = args(25);
            double N12 = args(26);
            double N20 = args(27);
            double N21 = args(28);
            double N22 = args(29);

            double ca1 = std::cos(a1);
            double ca1n = sign(ca1) * std::pow(std::abs(ca1), s1n);
            double ca1n2 = std::pow(std::abs(ca1), 2 - s1n);

            double sa1 = std::sin(a1);
            double sa1n = sign(sa1) * std::pow(std::abs(sa1), s1n);
            double sa1n2 = sign(sa1) * std::pow(std::abs(sa1), 2 - s1n);

            double cb1 = std::cos(b1);
            double cb1e = sign(cb1) * std::pow(std::abs(cb1), s1e);
            double cb1e2 = sign(cb1) * std::pow(std::abs(cb1), 2 - s1e);

            double sb1 = std::sin(b1);
            double sb1e = sign(sb1) * std::pow(std::abs(sb1), s1e);
            double sb1e2 = sign(sb1) * std::pow(std::abs(sb1), 2 - s1e);

            double A1 = s1ry * s1rz * ca1n2 * cb1e2;
            double A2 = s1rx * s1rz * sb1e2 * ca1n2;
            double A3 = s1rx * s1ry * sa1n2 * sign(ca1);
            double A4 = s1rx * ca1n  * cb1e;
            double A5 = s1ry * sb1e  * ca1n;
            double A6 = s1rz * sa1n;

            double ca2 = std::cos(a2);

            double ca2n2 = std::abs(ca2);

            double sa2 = std::sin(a2);

            double cb2 = std::cos(b2);

            double sb2 = std::sin(b2);

            double B1 = s2r * ca2 * cb2;
            double B2 = s2r * sb2 * ca2;
            double B3 = s2r * sa2;
            double B4 = s2r * s2r * ca2n2 * cb2;
            double B5 = s2r * s2r * sb2 * ca2n2;
            double B6 = s2r * s2r * sa2 * sign(ca2);

            xt::xtensor_fixed<double, xt::xshape<4>> res;
            res(0) = - (M10*A1 + M11*A2 + M12*A3) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc)
                     + (M20*A1 + M21*A2 + M22*A3) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc);
            res(1) =   (M00*A1 + M01*A2 + M02*A3) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc)
                     - (M20*A1 + M21*A2 + M22*A3) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);
            res(2) = - (M00*A1 + M01*A2 + M02*A3) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc)
                     + (M10*A1 + M11*A2 + M12*A3) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);
            res(3) =   (M00*A1 + M01*A2 + M02*A3) * (N00*B4 + N01*B5 + N02*B6)
                     + (M10*A1 + M11*A2 + M12*A3) * (N10*B4 + N11*B5 + N12*B6)
                     + (M20*A1 + M21*A2 + M22*A3) * (N20*B4 + N21*B5 + N22*B6)
                     + std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) )
                     * std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) );
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
          double M00 = args(8);
          double M01 = args(9);
          double M02 = args(10);
          double M10 = args(11);
          double M11 = args(12);
          double M12 = args(13);
          double M20 = args(14);
          double M21 = args(15);
          double M22 = args(16);
          double s2xc = args(17);
          double s2yc = args(18);
          double s2zc = args(19);
          double s2r = args(20);
          double N00 = args(21);
          double N01 = args(22);
          double N02 = args(23);
          double N10 = args(24);
          double N11 = args(25);
          double N12 = args(26);
          double N20 = args(27);
          double N21 = args(28);
          double N22 = args(29);

          double ca1 = std::cos(a1);
          double ca1n = sign(ca1) * std::pow(std::abs(ca1), s1n);
          double ca1n1 = s1n * std::pow(std::abs(ca1), s1n - 1);
          double ca1n2 = std::pow(std::abs(ca1), 2 - s1n);
          double ca1n3 = (2 - s1n) * sign(ca1) * std::pow(std::abs(ca1), 1 - s1n);

          double sa1 = std::sin(a1);
          double sa1n = sign(sa1) * std::pow(std::abs(sa1), s1n);
          double sa1n1 = s1n * std::pow(std::abs(sa1), s1n - 1);
          double sa1n2 = sign(sa1) * std::pow(std::abs(sa1), 2 - s1n);
          double sa1n3 = (2 - s1n) * std::pow(std::abs(sa1), 1 - s1n);

          double cb1 = std::cos(b1);
          double cb1e = sign(cb1) * std::pow(std::abs(cb1), s1e);
          double cb1e1 = s1e * std::pow(std::abs(cb1), s1e - 1);
          double cb1e2 = sign(cb1) * std::pow(std::abs(cb1), 2 - s1e);
          double cb1e3 = (2 - s1e) * std::pow(std::abs(cb1), 1 - s1e);

          double sb1 = std::sin(b1);
          double sb1e = sign(sb1) * std::pow(std::abs(sb1), s1e);
          double sb1e1 = s1e * std::pow(std::abs(sb1), s1e - 1);
          double sb1e2 = sign(sb1) * std::pow(std::abs(sb1), 2 - s1e);
          double sb1e3 = (2 - s1e) * std::pow(std::abs(sb1), 1 - s1e);

          double A1 = s1ry * s1rz * ca1n2 * cb1e2;
          double A2 = s1rx * s1rz * sb1e2 * ca1n2;
          double A3 = s1rx * s1ry * sa1n2 * sign(ca1);
          double A4 = s1rx * ca1n  * cb1e;
          double A5 = s1ry * sb1e  * ca1n;
          double A6 = s1rz * sa1n;
          double A7 = s1rx * sa1 * ca1n1 * cb1e;
          double A8 = s1ry * sa1 * sb1e  * ca1n1;
          double A9 = s1rz * ca1 * sa1n1;
          double A10 = s1ry * s1rz * sa1 * ca1n3 * cb1e2;
          double A11 = s1rx * s1rz * sa1 * sb1e2 * ca1n3;
          double A12 = s1rx * s1ry * std::abs(ca1) * sa1n3;
          double A13 = s1rx * sb1 * ca1n  * cb1e1;
          double A14 = s1ry * cb1 * sb1e1 * ca1n;
          double A15 = s1ry * s1rz  * sb1 * ca1n2 * cb1e3;
          double A16 = s1rx * s1rz  * cb1 * sb1e3 * ca1n2;

          double ca2 = std::cos(a2);
          double ca2n2 = std::abs(ca2);
          double ca2n3 = sign(ca2);

          double sa2 = std::sin(a2);

          double cb2 = std::cos(b2);

          double sb2 = std::sin(b2);

          double B1 = s2r * ca2 * cb2;
          double B2 = s2r * sb2 * ca2;
          double B3 = s2r * sa2;
          double B4 = s2r * s2r * ca2n2 * cb2;
          double B5 = s2r * s2r * sb2 * ca2n2;
          double B6 = s2r * s2r * sa2 * sign(ca2);
          double B7 = s2r * sa2 * cb2;
          double B8 = s2r * sa2 * sb2;
          double B9 = s2r * ca2;
          double B10 = s2r * sb2 * ca2;
          double B11 = s2r * cb2 * ca2;
          double B12 = s2r * s2r * sa2 * ca2n3 * cb2;
          double B13 = s2r * s2r * sa2 * sb2 * ca2n3;
          double B14 = s2r * s2r * std::abs(ca2);
          double B15 = s2r * s2r * sb2 * ca2n2;
          double B16 = s2r * s2r * cb2 * ca2n2;

          xt::xtensor_fixed<double, xt::xshape<4,4>> res;
          res(0,0) =   (-M10*A1 - M11*A2 - M12*A3) * (M20*A7 + M21*A8 - M22*A9)
                     + ( M20*A1 + M21*A2 + M22*A3) * (M10*A7 + M11*A8 - M12*A9)
                     + ( M10*A10 + M11*A11 - M12*A12) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc)
                     + (-M20*A10 - M21*A11 + M22*A12) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc);

          res(0,1) =   (-M10*A1 - M11*A2 - M12*A3) * (M20*A13 - M21*A14)
                     + ( M20*A1 + M21*A2 + M22*A3) * (M10*A13 - M11*A14)
                     + ( M10*A15 - M11*A16) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc)
                     + (-M20*A15 + M21*A16)*(-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc);

          res(0,2) =   (-M10*A1 - M11*A2 - M12*A3) * (-N20*B7 - N21*B8 + N22*B9)
                     + ( M20*A1 + M21*A2 + M22*A3) * (-N10*B7 - N11*B8 + N12*B9);

          res(0,3) =   (-M10*A1 - M11*A2 - M12*A3) * (-N20*B10 + N21*B11)
                     + ( M20*A1 + M21*A2 + M22*A3) * (-N10*B10 + N11*B11);

          res(1,0) =   ( M00*A1 + M01*A2 + M02*A3) * (M20*A7 + M21*A8 - M22*A9)
                     + (-M20*A1 - M21*A2 - M22*A3) * (M00*A7 + M01*A8 - M02*A9)
                     + (-M00*A10 - M01*A11 + M02*A12) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc)
                     + ( M20*A10 + M21*A11 - M22*A12) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);

          res(1,1) =   ( M00*A1 + M01*A2 + M02*A3) * (M20*A13 - M21*A14)
                     + (-M20*A1 - M21*A2 - M22*A3) * (M00*A13 - M01*A14)
                     + (-M00*A15 + M01*A16) * (-M20*A4 - M21*A5 - M22*A6 + N20*B1 + N21*B2 + N22*B3 - s1zc + s2zc)
                     + ( M20*A15 - M21*A16) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);

          res(1,2) =   ( M00*A1 + M01*A2 + M02*A3) * (-N20*B7 - N21*B8 + N22*B9)
                     + (-M20*A1 - M21*A2 - M22*A3) * (-N00*B7 - N01*B8 + N02*B9);

          res(1,3) =   ( M00*A1 + M01*A2 + M02*A3) * (-N20*B10 + N21*B11)
                     + (-M20*A1 - M21*A2 - M22*A3) * (-N00*B10 + N01*B11);

          res(2,0) =   (-M00*A1 - M01*A2 - M02*A3) * (M10*A7 + M11*A8 - M12*A9)
                     + ( M10*A1 + M11*A2 + M12*A3) * (M00*A7 + M01*A8 - M02*A9)
                     + ( M00*A10 + M01*A11 - M02*A12) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc)
                     + (-M10*A10 - M11*A11 + M12*A12) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);

          res(2,1) =   (-M00*A1 - M01*A2 - M02*A3) * (M10*A13 - M11*A14)
                     + ( M10*A1 + M11*A2 + M12*A3) * (M00*A13 - M01*A14)
                     + ( M00*A15 - M01*A16) * (-M10*A4 - M11*A5 - M12*A6 + N10*B1 + N11*B2 + N12*B3 - s1yc + s2yc)
                     + (-M10*A15 + M11*A16) * (-M00*A4 - M01*A5 - M02*A6 + N00*B1 + N01*B2 + N02*B3 - s1xc + s2xc);

          res(2,2) =   (-M00*A1 - M01*A2 - M02*A3) * (-N10*B7 - N11*B8 + N12*B9)
                     + (M10*A1 + M11*A2 + M12*A3) * (-N00*B7 - N01*B8 + N02*B9);

          res(2,3) =   (-M00*A1 - M01*A2 - M02*A3) * (-N10*B10 + N11*B11)
                     + (M10*A1 + M11*A2 + M12*A3) * (-N00*B10 + N01*B11);

          res(3,0) = (   ( M00*A1 + M01*A2 + M02*A3) * (-M00*A10 - M01*A11 + M02*A12)
                       + ( M10*A1 + M11*A2 + M12*A3) * (-M10*A10 - M11*A11 + M12*A12)
                       + ( M20*A1 + M21*A2 + M22*A3) * (-M20*A10 - M21*A11 + M22*A12)
                     ) \
                     * std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) )
                     / std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) )
                     + ( N00*B4 + N01*B5 + N02*B6) * ( -M00*A10 - M01*A11 + M02*A12)
                     + ( N10*B4 + N11*B5 + N12*B6) * (-M10*A10 - M11*A11 + M12*A12)
                     + ( N20*B4 + N21*B5 + N22*B6) * (-M20*A10 - M21*A11 + M22*A12);

          res(3,1) = (   ( M00*A1 + M01*A2 + M02*A3) * (-M00*A15 + M01*A16)
                       + ( M10*A1 + M11*A2 + M12*A3) * (-M10*A15 + M11*A16)
                       + ( M20*A1 + M21*A2 + M22*A3) * (-M20*A15 + M21*A16)
                     )
                     * std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) )
                     / std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) )
                     + ( N00*B4 + N01*B5 + N02*B6) * (-M00*A15 + M01*A16 )
                     + ( N10*B4 + N11*B5 + N12*B6) * (-M10*A15 + M11*A16)
                     + ( N20*B4 + N21*B5 + N22*B6) * (-M20*A15 + M21*A16);

          res(3,2) =  (   ( N00*B4 + N01*B5 + N02*B6) * (-N00*B12 - N01*B13 + N02*B14)
                        + ( N10*B4 + N11*B5 + N12*B6) * (-N10*B12 - N11*B13 + N12*B14)
                        + ( N20*B4 + N21*B5 + N22*B6) * (-N20*B12 - N21*B13 + N22*B14)
                      )
                      * std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) )
                      / std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) )
                      + (M00*A1 + M01*A2 + M02*A3) * (-N00*B12 - N01*B13 + N02*B14)
                      + (M10*A1 + M11*A2 + M12*A3) * (-N10*B12 - N11*B13 + N12*B14)
                      + (M20*A1 + M21*A2 + M22*A3) * (-N20*B12 - N21*B13 + N22*B14);

          res(3,3) = (    (N00*B4 + N01*B5 + N02*B6) * (-N00*B15 + N01*B16)
                        + (N10*B4 + N11*B5 + N12*B6) * (-N10*B15 + N11*B16)
                        + (N20*B4 + N21*B5 + N22*B6) * (-N20*B15 + N21*B16)
                     )
                     * std::sqrt( std::pow(M00*A1 + M01*A2 + M02*A3, 2) + std::pow(M10*A1 + M11*A2 + M12*A3, 2) + std::pow(M20*A1 + M21*A2 + M22*A3, 2) )
                     / std::sqrt( std::pow(N00*B4 + N01*B5 + N02*B6, 2) + std::pow(N10*B4 + N11*B5 + N12*B6, 2) + std::pow(N20*B4 + N21*B5 + N22*B6, 2) )
                     + (M00*A1 + M01*A2 + M02*A3) * (-N00*B15 + N01*B16)
                     + (M10*A1 + M11*A2 + M12*A3) * (-N10*B15 + N11*B16)
                     + (M20*A1 + M21*A2 + M22*A3) * (-N20*B15 + N21*B16);
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
        auto [ u, info ] = newton_method(u0,newton_F,newton_GradF,args,200,1.0e-10,1.0e-7);
        neigh.pi = s1.point(u(0),u(1));
        neigh.pj = s2.point(u(2),u(3));
        neigh.nij = s2.normal(u(2),u(3));
        neigh.dij = xt::linalg::dot(neigh.pi - neigh.pj, neigh.nij)[0];
        return neigh;
    }

    // SPHERE 3D - SUPERELLIPSOID 3D
    template<bool owner>
    auto closest_points(const superellipsoid<3, owner>& s2, const sphere<3, owner>& s1)
    {
        auto neigh = closest_points(s1, s2);
        neigh.nij *= -1.;
        std::swap(neigh.pi, neigh.pj);
        return neigh;
    }

    // SUPERELLIPSOID 2D - SPHERE 2D
    template<bool owner>
    auto closest_points(const sphere<2, owner> s2, const superellipsoid<2, owner> s1)
    {
      std::cout << "closest_points : SUPERELLIPSOID 2D - SPHERE 2D" << std::endl;
      double pi = 4*std::atan(1);
      neighbor<2> neigh;
      // 2*dim (pos) + 1+2 (r) + 1 (e) + 2*dim*dim (rot) = 30
      xt::xtensor<double, 1> xts2r = { s2.radius() };
      xt::xtensor_fixed<double, xt::xshape<16>> args = xt::hstack(xt::xtuple(
          xt::view(s1.pos(), 0), s1.radius(), s1.squareness(), xt::flatten(s1.rotation()),
          xt::view(s2.pos(), 0), xts2r, xt::flatten(s2.rotation())
      ));
      // std::cout << "args = " << args << std::endl;
      auto newton_F = [](auto u, auto args)
      {
        double b1 = u(0);
        double b2 = u(1);
        double s1xc = args(0);
        double s1yc = args(1);
        double s1rx = args(2);
        double s1ry = args(3);
        double s1e = args(4);
        double M00 = args(5);
        double M01 = args(6);
        double M10 = args(7);
        double M11 = args(8);
        double s2xc = args(9);
        double s2yc = args(10);
        double s2r = args(11);
        double N00 = args(12);
        double N01 = args(13);
        double N10 = args(14);
        double N11 = args(15);

        double D1 = s1rx * std::pow(std::abs(std::sin(b1)), 2 - s1e) * sign(std::sin(b1));
        double D2 = s1ry * std::pow(std::abs(std::sin(b1)), s1e) * sign(std::sin(b1));
        double D3 = s1ry * std::pow(std::abs(std::cos(b1)), 2 - s1e) * sign(std::cos(b1));
        double D4 = s1rx * std::pow(std::abs(std::cos(b1)), s1e) * sign(std::cos(b1));
        double D5 = s2r * std::sin(b2);
        double D6 = s2r * std::cos(b2);

        xt::xtensor_fixed<double, xt::xshape<2>> res;
        res(0) = -( M00*D3 + M01*D1 ) *
                  (-M10*D4 - M11*D2 + N10*D6 + N11*D5 - s1yc + s2yc ) +
                  ( M10*D3 + M11*D1 ) *
                  (-M00*D4 - M01*D2 + N00*D6 + N01*D5 - s1xc + s2xc );
        res(1) =  ( N00*D6 + N01*D5 ) *
                  ( M00*D3 + M01*D1 ) +
                  ( N10*D6 + N11*D5 ) *
                  ( M10*D3 + M11*D1 ) +
                  std::sqrt( std::pow(N00*D6 + N01*D5, 2) + std::pow(N10*D6 + N11*D5, 2) ) *
                  std::sqrt( std::pow(M00*D3 + M01*D1, 2) + std::pow(M10*D3 + M11*D1, 2) );
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
        double M00 = args(5);
        double M01 = args(6);
        double M10 = args(7);
        double M11 = args(8);
        double s2xc = args(9);
        double s2yc = args(10);
        double s2r = args(11);
        double N00 = args(12);
        double N01 = args(13);
        double N10 = args(14);
        double N11 = args(15);

        double D1 = s1rx * std::pow(std::abs(std::sin(b1)), 2 - s1e) * sign(std::sin(b1));
        double D2 = s1ry * std::pow(std::abs(std::sin(b1)), s1e) * sign(std::sin(b1));
        double D3 = s1ry * std::pow(std::abs(std::cos(b1)), 2 - s1e) * sign(std::cos(b1));
        double D4 = s1rx * std::pow(std::abs(std::cos(b1)), s1e) * sign(std::cos(b1));
        double D5 = s2r * std::sin(b2);
        double D6 = s2r * std::cos(b2);
        double D7 = s1e * s1rx * std::sin(b1) * std::pow(std::abs(std::cos(b1)), s1e - 1);
        double D8 = s1e * s1ry * std::cos(b1) * std::pow(std::abs(std::sin(b1)), s1e - 1);
        double D9 = s1ry * (2 - s1e) * std::sin(b1) * std::pow(std::abs(std::cos(b1)), 1 - s1e);
        double D10 = s1rx * (2 - s1e) * std::cos(b1) * std::pow(std::abs(std::sin(b1)), 1 - s1e);

        xt::xtensor_fixed<double, xt::xshape<2,2>> res;
        res(0,0) = (-M00*D3 - M01*D1 ) * ( M10*D7 - M11*D8 ) +
                   ( M10*D3 + M11*D1 ) * ( M00*D7 - M01*D8 ) +
                   ( M00*D9 - M01*D10 ) * (-M10*D4 - M11*D2 + N10*D6 + N11*D5 - s1yc + s2yc ) +
                   (-M10*D9 + M11*D10 ) * (-M00*D4 - M01*D2 + N00*D6 + N01*D5 - s1xc + s2xc);

        res(0,1) = (-N00*D5 + N01*D6 ) * ( M10*D3 + M11*D1 ) +
                   (-N10*D5 + N11*D6 ) * (-M00*D3 - M01*D1 );

        res(1,0) = ( ( M00*D3 + M01*D1 ) * (-M00*D9 + M01*D10 ) +
                     ( M10*D3 + M11*D1 ) * (-M10*D9 + M11*D10 )
                   )
                  * std::sqrt( std::pow( N00*D6 + N01*D5, 2 ) + std::pow( N10*D6 + N11*D5, 2 ) )
                  / std::sqrt( std::pow( M00*D3 + M01*D1, 2 ) + std::pow( M10*D3 + M11*D1, 2 ) )
                  + ( N00*D6 + N01*D5 ) * (-M00*D9 + M01*D10 ) +
                    ( N10*D6 + N11*D5 ) * (-M10*D9 + M11*D10 );

        res(1,1) = ( (-N00*D5 + N01*D6 ) * ( N00*D6 + N01*D5 ) +
                     (-N10*D5 + N11*D6 ) * ( N10*D6 + N11*D5 ) )
                   * std::sqrt( std::pow( M00*D3 + M01*D1, 2 ) + std::pow( M10*D3 + M11*D1, 2 ) )
                   / std::sqrt( std::pow( N00*D6 + N01*D5, 2 ) + std::pow( N10*D6 + N11*D5, 2 ) )
                   + (-N00*D5 + N01*D6 ) * ( M00*D3 + M01*D1 ) +
                     (-N10*D5 + N11*D6 ) * ( M10*D3 + M11*D1 );
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
      auto [ u, info ] = newton_method(u0,newton_F,newton_GradF,args,200,1.0e-10,1.0e-7);
      neigh.pi = s1.point(u(0));
      neigh.pj = s2.point(u(1));
      neigh.nij = s2.normal(u(1));
      neigh.dij = xt::linalg::dot(neigh.pi - neigh.pj, neigh.nij)[0];
      return neigh;
    }

    // SUPERELLIPSOID 2D - SPHERE 2D
    template<bool owner>
    auto closest_points(const superellipsoid<2, owner> superellipsoid, const sphere<2, owner> sphere)
    {
        auto neigh = closest_points(sphere, superellipsoid);
        neigh.nij *= -1.;
        std::swap(neigh.pi, neigh.pj);
        return neigh;
    }

    // SUPERELLIPSOID 3D - PLAN 3D
    template<bool owner>
    auto closest_points(const superellipsoid<3, owner> s1, const plan<3, owner> p2)
    {
      std::cout << "closest_points : SUPERELLIPSOID 3D - PLAN 3D" << std::endl;
      double pi = 4*std::atan(1);
      neighbor<3> neigh;
      // Pour déterminer de quel cote on est
      auto  xt_sign_p2s = xt::linalg::dot( xt::view(s1.pos(), 0)-xt::view(p2.pos(), 0), xt::flatten(p2.normal()) );
      // xt::xtensor<double, 1> xt_sign_p2s = { xt::linalg::dot( xt::view(s1.pos(), 0)-xt::view(p2.pos(), 0), xt::flatten(p2.normal()) ) };
      // 2*dim (pos) + 1+3 (r) + 2 (e et n) + 2*dim*dim (rot) = 30
      xt::xtensor_fixed<double, xt::xshape<30>> args = xt::hstack(xt::xtuple(
          xt::view(s1.pos(), 0), s1.radius(), s1.squareness(), xt::flatten(s1.rotation()),
          xt::view(p2.pos(), 0), xt::flatten(p2.rotation()), xt_sign_p2s
      ));
      // std::cout << "args = " << args << std::endl;
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
        double M00 = args(8);
        double M01 = args(9);
        double M02 = args(10);
        double M10 = args(11);
        double M11 = args(12);
        double M12 = args(13);
        double M20 = args(14);
        double M21 = args(15);
        double M22 = args(16);
        double p2xc = args(17);
        double p2yc = args(18);
        double p2zc = args(19);
        double N00 = args(20);
        double N01 = args(21);
        double N02 = args(22);
        double N10 = args(23);
        double N11 = args(24);
        double N12 = args(25);
        double N20 = args(26);
        double N21 = args(27);
        double N22 = args(28);
        double sign_p2s = args(29);

        double ca1 = std::cos(a1);
        double ca1n = sign(ca1) * std::pow(std::abs(ca1), s1n);
        double ca1n2 = std::pow(std::abs(ca1), 2 - s1n);

        double sa1 = std::sin(a1);
        double sa1n = std::pow(std::abs(sa1), s1n) * sign(sa1);
        double sa1n2 = std::pow(std::abs(sa1), 2 - s1n) * sign(sa1);

        double cb1 = std::cos(b1);
        double cb1e = sign(cb1)*std::pow(std::abs(cb1), s1e);
        double cb1e2 = std::pow(std::abs(cb1), 2 - s1e) * sign(cb1);

        double sb1 = std::sin(b1);
        double sb1e = std::pow(std::abs(sb1), s1e) * sign(sb1);
        double sb1e2 = std::pow(std::abs(sb1), 2 - s1e) * sign(sb1);

        double A1 = s1ry * s1rz * ca1n2 * cb1e2;
        double A2 = s1rx * s1rz * sb1e2 * ca1n2;
        double A3 = s1rx * s1ry * sa1n2 * sign(ca1);
        double A4 = s1rx * ca1n  * cb1e;
        double A5 = s1ry * sb1e  * ca1n;
        double A6 = s1rz * sa1n;

        xt::xtensor_fixed<double, xt::xshape<4>> res;

        res(0) = - (  M10*A1 + M11*A2 + M12*A3 )
                 * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc )
                 + (  M20*A1 + M21*A2 + M22*A3 )
                 * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc );

        res(1) =   (  M00*A1 + M01*A2 + M02*A3 )
                 * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc )
                 - (  M20*A1 + M21*A2 + M22*A3 )
                 * ( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc );

        res(2) = - (  M00*A1 + M01*A2 + M02*A3 )
                 * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc )
                 + (  M10*A1 + M11*A2 + M12*A3 )
                 * ( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc );

        res(3) =   N00*sign_p2s*( M00*A1 + M01*A2 + M02*A3 )
                 + N10*sign_p2s*( M10*A1 + M11*A2 + M12*A3 )
                 + N20*sign_p2s*( M20*A1 + M21*A2 + M22*A3 )
                 + std::sqrt( std::pow(N00, 2) + std::pow(N10, 2) + std::pow(N20, 2) )
                 * std::sqrt(  std::pow( M00*A1 + M01*A2 + M02*A3, 2)
                             + std::pow( M10*A1 + M11*A2 + M12*A3, 2)
                             + std::pow( M20*A1 + M21*A2 + M22*A3, 2) );
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
        double M00 = args(8);
        double M01 = args(9);
        double M02 = args(10);
        double M10 = args(11);
        double M11 = args(12);
        double M12 = args(13);
        double M20 = args(14);
        double M21 = args(15);
        double M22 = args(16);
        double p2xc = args(17);
        double p2yc = args(18);
        double p2zc = args(19);
        double N00 = args(20);
        double N01 = args(21);
        double N02 = args(22);
        double N10 = args(23);
        double N11 = args(24);
        double N12 = args(25);
        double N20 = args(26);
        double N21 = args(27);
        double N22 = args(28);
        double sign_p2s = args(29);

        double ca1 = std::cos(a1);
        double ca1n = sign(ca1) * std::pow(std::abs(ca1), s1n);
        double ca1n1 = s1n * std::pow(std::abs(ca1), s1n - 1);
        double ca1n2 = std::pow(std::abs(ca1), 2 - s1n);
        double ca1n3 = (2 - s1n) * sign(ca1) * std::pow(std::abs(ca1), 1 - s1n);

        double sa1 = std::sin(a1);
        double sa1n = std::pow(std::abs(sa1), s1n) * sign(sa1);
        double sa1n1 = s1n * std::pow(std::abs(sa1), s1n - 1);
        double sa1n2 = std::pow(std::abs(sa1), 2 - s1n) * sign(sa1);
        double sa1n3 = (2 - s1n) * std::pow(std::abs(sa1), 1 - s1n);

        double cb1 = std::cos(b1);
        double cb1e = sign(cb1) * std::pow(std::abs(cb1), s1e);
        double cb1e1 = s1e * std::pow(std::abs(cb1), s1e - 1);
        double cb1e2 = std::pow(std::abs(cb1), 2 - s1e) * sign(cb1);
        double cb1e3 = (2 - s1e) * std::pow(std::abs(cb1), 1 - s1e);

        double sb1 = std::sin(b1);
        double sb1e = std::pow(std::abs(sb1), s1e) * sign(sb1);
        double sb1e1 = s1e * std::pow(std::abs(sb1), s1e - 1);
        double sb1e2 = std::pow(std::abs(sb1), 2 - s1e) * sign(sb1);
        double sb1e3 = (2 - s1e) * std::pow(std::abs(sb1), 1 - s1e);

        double A1 = s1ry * s1rz * ca1n2 * cb1e2;
        double A2 = s1rx * s1rz * sb1e2 * ca1n2;
        double A3 = s1rx * s1ry * sa1n2 * sign(ca1);
        double A4 = s1rx * ca1n  * cb1e;
        double A5 = s1ry * sb1e  * ca1n;
        double A6 = s1rz * sa1n;
        double A7 = s1rx * sa1 * ca1n1 * cb1e;
        double A8 = s1ry * sa1 * sb1e  * ca1n1;
        double A9 = s1rz * ca1 * sa1n1;
        double A10 = s1ry * s1rz * sa1 * ca1n3 * cb1e2;
        double A11 = s1rx * s1rz * sa1 * sb1e2 * ca1n3;
        double A12 = s1rx * s1ry * std::abs(ca1) * sa1n3;
        double A13 = s1rx * sb1 * ca1n  * cb1e1;
        double A14 = s1ry * cb1 * sb1e1 * ca1n;
        double A15 = s1ry * s1rz  * sb1 * ca1n2 * cb1e3;
        double A16 = s1rx * s1rz  * cb1 * sb1e3 * ca1n2;

        xt::xtensor_fixed<double, xt::xshape<4,4>> res;

        res(0,0) = ( -M10*A1 - M11*A2 - M12*A3 ) * (  M20*A7 + M21*A8 - M22*A9 )
                  +(  M20*A1 + M21*A2 + M22*A3 ) * (  M10*A7 + M11*A8 - M12*A9 )
                  +(  M10*A10 + M11*A11 - M12*A12 ) * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc )
                  +( -M20*A10 - M21*A11 + M22*A12 ) * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc );

        res(0,1) = ( -M10*A1 - M11*A2 - M12*A3 ) * ( M20*A13 - M21*A14 )
                  +(  M20*A1 + M21*A2 + M22*A3 ) * ( M10*A13 - M11*A14 )
                  +(  M10*A15 - M11*A16 ) * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc )
                  +( -M20*A15 + M21*A16 ) * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc );

        res(0,2) = N11*( M20*A1 + M21*A2 + M22*A3 ) + N21*( -M10*A1 - M11*A2 - M12*A3 );

        res(0,3) = N12*( M20*A1 + M21*A2 + M22*A3 ) + N22*( -M10*A1 - M11*A2 - M12*A3 );

        res(1,0) = (  M00*A1 + M01*A2 + M02*A3 ) * ( M20*A7 + M21*A8 - M22*A9 )
                  +( -M20*A1 - M21*A2 - M22*A3 ) * ( M00*A7 + M01*A8 - M02*A9 )
                  +( -M00*A10 - M01*A11 + M02*A12 ) * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc )
                 + (  M20*A10 + M21*A11 - M22*A12 ) * ( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc );

        res(1,1) = (  M00*A1 + M01*A2 + M02*A3 ) * ( M20*A13 - M21*A14 )
                  +( -M20*A1 - M21*A2 - M22*A3 ) * ( M00*A13 - M01*A14 )
                  +( -M00*A15 + M01*A16 ) * ( -M20*A4 - M21*A5 - M22*A6 + N21*a2 + N22*b2 + p2zc - s1zc )
                  +(  M20*A15 - M21*A16 ) * ( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc );

        res(1,2) = N01*( -M20*A1 - M21*A2 - M22*A3 ) + N21*(  M00*A1 + M01*A2 + M02*A3 );

        res(1,3) = N02*(  -M20*A1 - M21*A2 - M22*A3 ) + N22*(   M00*A1 + M01*A2 + M02*A3 );

        res(2,0) = ( -M00*A1 - M01*A2 - M02*A3 ) * ( M10*A7 + M11*A8 - M12*A9 )
                  +(  M10*A1 + M11*A2 + M12*A3 ) * ( M00*A7 + M01*A8 - M02*A9 )
                  +(  M00*A10 + M01*A11 - M02*A12 ) * ( -M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc )
                  +(  -M10*A10 - M11*A11 + M12*A12) *( -M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc );

        res(2,1) = (-M00*A1 - M01*A2 - M02*A3 ) * ( M10*A13 - M11*A14 )
                 + ( M10*A1 + M11*A2 + M12*A3 ) * ( M00*A13 - M01*A14 )
                 + ( M00*A15 - M01*A16 ) * (-M10*A4 - M11*A5 - M12*A6 + N11*a2 + N12*b2 + p2yc - s1yc )
                  +(-M10*A15 + M11*A16 ) * (-M00*A4 - M01*A5 - M02*A6 + N01*a2 + N02*b2 + p2xc - s1xc );

        res(2,2) =  N01*( M10*A1 + M11*A2 + M12*A3 ) + N11*(-M00*A1 - M01*A2 - M02*A3 );

        res(2,3) = N02*(   M10*A1 + M11*A2 + M12*A3 ) + N12*(  -M00*A1 - M01*A2 - M02*A3 );

        res(3,0) = N00*sign_p2s*(-M00*A10 - M01*A11 + M02*A12 )
                 + N10*sign_p2s*(-M10*A10 - M11*A11 + M12*A12 )
                 + N20*sign_p2s*(-M20*A10 - M21*A11 + M22*A12 )
                 + std::sqrt( std::pow(N00, 2) + std::pow(N10, 2) + std::pow(N20, 2) )
                 * ( ( M00*A1 + M01*A2 + M02*A3 ) * (-M00*A10 - M01*A11 + M02*A12 )
                   + ( M10*A1 + M11*A2 + M12*A3 ) * (-M10*A10 - M11*A11 + M12*A12 )
                   + ( M20*A1 + M21*A2 + M22*A3 ) * (-M20*A10 - M21*A11 + M22*A12 ) )
                 / std::sqrt( std::pow( M00*A1 + M01*A2 + M02*A3, 2)
                            + std::pow( M10*A1 + M11*A2 + M12*A3, 2)
                            + std::pow( M20*A1 + M21*A2 + M22*A3, 2) );

        res(3,1) = N00*sign_p2s*(-M00*A15 + M01*A16 )
                 + N10*sign_p2s*(-M10*A15 + M11*A16 )
                 + N20*sign_p2s*(-M20*A15 + M21*A16 )
                 + std::sqrt( std::pow(N00, 2) + std::pow(N10, 2) +std::pow(N20, 2) )
                 * ( ( M00*A1 + M01*A2 + M02*A3 ) * (-M00*A15 + M01*A16 )
                   + ( M10*A1 + M11*A2 + M12*A3 ) * (-M10*A15 + M11*A16 )
                   + ( M20*A1 + M21*A2 + M22*A3 ) * (-M20*A15 + M21*A16 ) )
                 / std::sqrt( std::pow( M00*A1 + M01*A2 + M02*A3, 2)
                          + std::pow( M10*A1 + M11*A2 + M12*A3, 2)
                          + std::pow( M20*A1 + M21*A2 + M22*A3, 2) );

        res(3,2) = 0;

        res(3,3) = 0;

        return res;
      };
      const int num = 10;
      auto ainit = xt::linspace<double>(-pi/2, pi/2, num);
      auto binit = xt::linspace<double>(-pi, pi, num);
      xt::xtensor_fixed<double, xt::xshape<num,num>> dinit;
      for (std::size_t i = 0; i < binit.size(); i++) {
          for (std::size_t j = 0; j < binit.size(); j++) {
              dinit(i,j) = xt::linalg::norm(s1.point(ainit(i),binit(i))-p2.point(ainit(j),binit(j)),2);
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
      auto [ u, info ] = newton_method(u0,newton_F,newton_GradF,args,200,1.0e-10,1.0e-7);
      neigh.pi = s1.point(u(0),u(1));
      neigh.pj = p2.point(u(2),u(3));
      neigh.nij = p2.normal();
      neigh.dij = xt::linalg::dot(neigh.pi - neigh.pj, neigh.nij)[0];
      return neigh;
    }

    // PLAN 3D - SUPERELLIPSOID 3D
    template<bool owner>
    auto closest_points(const plan<3, owner> p2, const superellipsoid<3, owner> s1)
    {
        auto neigh = closest_points(s1, p2);
        neigh.nij *= -1.;
        std::swap(neigh.pi, neigh.pj);
        return neigh;
    }

    // SUPERELLIPSOID 2D - DROITE 2D
    template<bool owner>
    auto closest_points(const superellipsoid<2, owner> s1, const plan<2, owner> d2)
    {
      // std::cout << "closest_points : SUPERELLIPSOID 2D - DROITE 2D" << std::endl;
      double pi = 4*std::atan(1);
      neighbor<2> neigh;
      // Pour déterminer de quel cote on est
      auto xt_sign_d2s = xt::linalg::dot( xt::view(s1.pos(), 0)-xt::view(d2.pos(), 0), xt::flatten(d2.normal()) );
      // xt::xtensor<double, 1> xt_sign_d2s = { xt::linalg::dot( xt::view(s1.pos(), 0)-xt::view(d2.pos(), 0), xt::flatten(d2.normal()) ) };
      // 2*dim (pos) + 2 (r) + 1 (e) + 2*dim*dim (rot) = 30
      xt::xtensor_fixed<double, xt::xshape<16>> args = xt::hstack(xt::xtuple(
          xt::view(s1.pos(), 0), s1.radius(), s1.squareness(), xt::flatten(s1.rotation()),
          xt::view(d2.pos(), 0), xt::flatten(d2.rotation()), xt_sign_d2s
      ));
      // std::cout << "args = " << args << std::endl;
      auto newton_F = [](auto u, auto args)
      {
        double b1 = u(0);
        double a2 = u(1);
        double s1xc = args(0);
        double s1yc = args(1);
        double s1rx = args(2);
        double s1ry = args(3);
        double s1e = args(4);
        double M00 = args(5);
        double M01 = args(6);
        double M10 = args(7);
        double M11 = args(8);
        double d2xc = args(9);
        double d2yc = args(10);
        double N00 = args(11);
        double N01 = args(12);
        double N10 = args(13);
        double N11 = args(14);
        double sign_d2s = args(15);

        double cb1 = std::cos(b1);
        double cb1e = sign(cb1) * std::pow(std::abs(cb1), s1e);
        double cb1e2 = std::pow(std::abs(cb1), 2 - s1e) * sign(cb1);

        double sb1 = std::sin(b1);
        double sb1e = std::pow(std::abs(sb1), s1e) * sign(sb1);
        double sb1e2 = std::pow(std::abs(sb1), 2 - s1e) * sign(sb1);

        double A1 = s1ry * cb1e2;
        double A2 = s1rx * sb1e2;
        double A4 = s1rx * cb1e;
        double A5 = s1ry * sb1e;

        xt::xtensor_fixed<double, xt::xshape<2>> res;
        res(0) = - ( M00*A1 + M01*A2 ) * (-M10*A4 - M11*A5 + N11*a2 + d2yc - s1yc )
                 + ( M10*A1 + M11*A2 ) * (-M00*A4 - M01*A5 + N01*a2 + d2xc - s1xc );
        res(1) = N00*sign_d2s*( M00*A1 + M01*A2 ) + N10*sign_d2s*( M10*A1 + M11*A2 )
               + std::sqrt( std::pow(N00, 2) + std::pow(N10, 2) )
               * std::sqrt( std::pow(M00*A1 + M01*A2, 2) + std::pow(M10*A1 + M11*A2, 2) );
        return res;
      };
      auto newton_GradF = [](auto u, auto args)
      {
        double b1 = u(0);
        double a2 = u(1);
        double s1xc = args(0);
        double s1yc = args(1);
        double s1rx = args(2);
        double s1ry = args(3);
        double s1e = args(4);
        double M00 = args(5);
        double M01 = args(6);
        double M10 = args(7);
        double M11 = args(8);
        double d2xc = args(9);
        double d2yc = args(10);
        double N00 = args(11);
        double N01 = args(12);
        double N10 = args(13);
        double N11 = args(14);
        double sign_d2s = args(15);

        double cb1 = std::cos(b1);
        double cb1e = sign(cb1) * std::pow(std::abs(cb1), s1e);
        double cb1e1 = s1e * std::pow(std::abs(cb1), s1e - 1);
        double cb1e2 = std::pow(std::abs(cb1), 2 - s1e) * sign(cb1);
        double cb1e3 = (2 - s1e) * std::pow(std::abs(cb1), 1 - s1e);

        double sb1 = std::sin(b1);
        double sb1e = std::pow(std::abs(sb1), s1e) * sign(sb1);
        double sb1e1 = s1e * std::pow(std::abs(sb1), s1e - 1);
        double sb1e2 = std::pow(std::abs(sb1), 2 - s1e) * sign(sb1);
        double sb1e3 = (2 - s1e) * std::pow(std::abs(sb1), 1 - s1e);

        double A1 = s1ry * cb1e2;
        double A2 = s1rx * sb1e2;
        double A4 = s1rx * cb1e;
        double A5 = s1ry * sb1e;
        double A13 = s1rx * sb1 * cb1e1;
        double A14 = s1ry * cb1 * sb1e1;
        double A15 = s1ry * sb1 * cb1e3;
        double A16 = s1rx * cb1 * sb1e3;

        xt::xtensor_fixed<double, xt::xshape<2,2>> res;

        res(0,0) = ( -M00*A1 - M01*A2 ) * (  M10*A13 - M11*A14 )
                 + (  M10*A1 + M11*A2 ) * (  M00*A13 - M01*A14 )
                 + (  M00*A15 - M01*A16 ) * ( -M10*A4 - M11*A5 + N11*a2 + d2yc - s1yc )
                 + ( -M10*A15 + M11*A16 ) * ( -M00*A4 - M01*A5 + N01*a2 + d2xc - s1xc );

        res(0,1) =  N01 * ( M10*A1 + M11*A2 ) + N11 * (-M00*A1 - M01*A2 );

        res(1,0) = N00 * sign_d2s * ( -M00*A15 + M01*A16 )
                 + N10 * sign_d2s * ( -M10*A15 + M11*A16 )
                 + std::sqrt( std::pow(N00, 2) + std::pow(N10, 2) )
                 * ( ( M00*A1 +  M01*A2 ) * ( -M00*A15 + M01*A16 ) + (M10*A1 + M11*A2 ) * ( -M10*A15 + M11*A16 ) )
                 / std::sqrt( std::pow( M00*A1 + M01*A2, 2) + std::pow( M10*A1 + M11*A2, 2) );

        res(1,1) = 0;

        return res;

      };
      const int num = 10;
      auto ainit = xt::linspace<double>(-num, num, num);
      auto binit = xt::linspace<double>(-pi, pi, num);
      xt::xtensor_fixed<double, xt::xshape<num,num>> dinit;
      for (std::size_t i = 0; i < binit.size(); i++) {
          for (std::size_t j = 0; j < binit.size(); j++) {
              dinit(i,j) = xt::linalg::norm(s1.point(binit(i))-d2.point(ainit(j)),2);
          }
      }
      // std::cout << "initialization : b = " << binit << " distances = " << dinit << std::endl;
      auto dmin = xt::amin(dinit);
      // std::cout << "initialization : dmin = " << dmin << std::endl;
      auto indmin = xt::from_indices(xt::where(xt::equal(dinit, dmin)));
      // std::cout << "initialization : indmin = " << indmin << std::endl;
      // std::cout << "initialization : imin = " << indmin(0,0) << " jmin = " << indmin(1,0) << std::endl;
      xt::xtensor_fixed<double, xt::xshape<2>> u0 = {binit(indmin(0,0)), ainit(indmin(1,0))};
      // std::cout << "newton_GradF(u0,args) = " << newton_GradF(u0,args) << " newton_F(u0,args) = " << newton_F(u0,args) << std::endl;
      auto [ u, info ] = newton_method(u0,newton_F,newton_GradF,args,200,1.0e-10,1.0e-7);
      // std::cout << "info = " << info << std::endl;
      if (info==-1){
        std::cout << "\ns1 : " << std::endl;
        s1.print();
        std::cout << "s2 : " << std::endl;
        d2.print();
        std::cout << "s1.pos = " << s1.pos() << " s1.rotation = " << xt::flatten(s1.rotation()) << std::endl;
        std::cout << "d2.pos = " << d2.pos() << " d2.rotation = " << xt::flatten(d2.rotation()) << std::endl;
      }
      neigh.pi = s1.point(u(0));
      neigh.pj = d2.point(u(1));
      neigh.nij = d2.normal();
      neigh.dij = xt::linalg::dot(neigh.pi - neigh.pj, neigh.nij)[0];
      return neigh;
    }

    // DROITE 2D - SUPERELLIPSOID 2D
    template<bool owner>
    auto closest_points(const plan<2, owner> d2, const superellipsoid<2, owner> s1)
    {
        auto neigh = closest_points(s1, d2);
        neigh.nij *= -1.;
        std::swap(neigh.pi, neigh.pj);
        return neigh;
    }

    // SUPERELLIPSOID - GLOBULE
    template<std::size_t dim, bool owner>
    auto closest_points(const superellipsoid<dim, owner>, const globule<dim, owner>)
    {
        std::cout << "closest_points : SUPERELLIPSOID - GLOBULE" << std::endl;
        return neighbor<dim>();
    }

    // GLOBULE  - SUPERELLIPSOID
    template<std::size_t dim, bool owner>
    auto closest_points(const globule<dim, owner> g, const superellipsoid<dim, owner> s)
    {
        auto neigh = closest_points(s, g);
        neigh.nij *= -1.;
        std::swap(neigh.pi, neigh.pj);
        return neigh;
    }

    // GLOBULE - PLAN
    template<std::size_t dim, bool owner>
    auto closest_points(const globule<dim, owner>, const plan<dim, owner>)
    {
        return neighbor<dim>();
    }

    // PLAN - GLOBULE
    template<std::size_t dim, bool owner>
    auto closest_points(const plan<dim, owner>, const globule<dim, owner>)
    {
        return neighbor<dim>();
    }

    template <std::size_t dim>
    struct closest_points_functor
    {
        using return_type = std::vector<neighbor<dim>>;
        // using return_type = neighbor<dim>;
         template <class T1, class T2>
        return_type run(const T1& obj1, const T2& obj2) const
        {
            return closest_points(obj1, obj2);
        }
         return_type on_error(const object<dim, false>&, const object<dim, false>&) const
        {
            return {};
        }
    };

    // template <std::size_t dim, bool owner=false>
    // using closest_points_dispatcher = double_static_dispatcher
    // <
    //     closest_points_functor<dim>,
    //     const object<dim, owner>,
    //     mpl::vector<const sphere<dim, owner>,
    //                 const superellipsoid<dim, owner>,
    //                 const globule<dim, owner>,
    //                 const plan<dim, owner>>,
    //     typename closest_points_functor<dim>::return_type,
    //     antisymmetric_dispatch
    // >;
    template <std::size_t dim, bool owner=false>
    using closest_points_dispatcher = double_static_dispatcher
    <
        closest_points_functor<dim>,
        const object<dim, owner>,
        mpl::vector<const globule<dim, owner>>,
        typename closest_points_functor<dim>::return_type,
        antisymmetric_dispatch
    >;
}

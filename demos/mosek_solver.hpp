#pragma once

#include <iostream>
#include <fstream>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include <fmt/format.h>
#include <fusion.h>
#include <nlohmann/json.hpp>

#include <scopi/container.hpp>
#include <scopi/functors.hpp>
#include <scopi/object/neighbor.hpp>
#include <scopi/quaternion.hpp>

using namespace mosek::fusion;
using namespace monty;
namespace nl = nlohmann;

using namespace xt::placeholders;

template<std::size_t dim>
void mosek_solver(scopi::scopi_container<dim>& particles, double dt, std::size_t total_it, std::size_t active_ptr)
{
    std::size_t Nactive = particles.size() - active_ptr;
    // Time Loop
    for (std::size_t nite=0; nite<total_it; ++nite)
    {
      std::cout << "\n\n------------------- Time iteration ----------------> " << nite << std::endl;
        std::vector<scopi::neighbor<dim>> contacts;
        double dmax = 1;

        //displacement of obstacles
        for (std::size_t i=0; i<active_ptr; ++i)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, particles.desired_omega()(i)});
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            scopi::type::quaternion expw;
            expw(0) = std::cos(0.5*normw*dt);
            xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*dt)/normw*w;

            for (std::size_t d=0; d<dim; ++d)
            {
                particles.pos()(i)(d) += dt*particles.vd()(i)(d);
            }
            particles.q()(i) = scopi::mult_quaternion(particles.q()(i), expw);

            std::cout << "obstacle " << i << ": " << particles.pos()(0) << " " << particles.q()(0) << std::endl;
        }

        // create list of contacts
        for(std::size_t i = 0; i < particles.size() - 1; ++i)
        {
            for(std::size_t j = i + 1; j < particles.size(); ++j)
            {
                auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                if (neigh.dij < dmax)
                {
                    contacts.emplace_back(std::move(neigh));
                    contacts.back().i = i;
                    contacts.back().j = j;
                }
            }
        }

        // output files
        nl::json json_output;

        std::ofstream file(fmt::format("./Results/scopi_objects_{:04d}.json", nite));

        json_output["objects"] = {};

        for(std::size_t i = 0; i < particles.size(); ++i)
        {
            json_output["objects"].push_back(scopi::write_objects_dispatcher<dim>::dispatch(*particles[i]));
        }

        json_output["contacts"] = {};

        for(std::size_t i=0; i<contacts.size(); ++i)
        {
            nl::json contact;

            contact["pi"] = contacts[i].pi;
            contact["pj"] = contacts[i].pj;
            contact["nij"] = contacts[i].nij;

            json_output["contacts"].push_back(contact);

        }

        file << std::setw(4) << json_output;
        file.close();

        // for (std::size_t i=0; i<Nactive; ++i)
        // {
        //     for (std::size_t d=0; d<dim; ++d)
        //     {
        //         particles.pos()(i + active_ptr)(d) += dt*particles.vd()(i + active_ptr)(d);
        //     }
        // }

        // create mass and inertia matrices
        double mass = 1.;
        double moment = .1;

        xt::xtensor<double, 1> MU = xt::zeros<double>({3*Nactive});
        xt::xtensor<double, 1> JW = xt::zeros<double>({3*Nactive});
        xt::xtensor<double, 2> sqrtM = mass*xt::eye<double>(3*Nactive);
        xt::xtensor<double, 2> sqrtJ = moment*xt::eye<double>(3*Nactive);

        for (std::size_t i=0; i<Nactive; ++i)
        {
            JW(3*i + 2) = moment*particles.desired_omega()(active_ptr + i);
            for (std::size_t d=0; d<dim; ++d)
            {
                MU(3*i+ d) = mass*particles.vd()(active_ptr + i)[d]; // TODO: add mass into particles
            }
        }

        // fill vector with distances
        xt::xtensor<double, 1> distances = xt::zeros<double>({contacts.size()});
        for(std::size_t i=0; i<contacts.size(); ++i)
        {
            distances[i] = contacts[i].dij;
        }
        std::cout << "distances " << distances << std::endl;

        //Create contstaint matrices A and B
        xt::xtensor<double, 2> Au = xt::zeros<double>({3*contacts.size(), Nactive*3});
        // xt::xtensor<double, 2> Aw = xt::zeros<double>({dim*contacts.size(), {(dim ==3)?Nactive*dim:Nactive}});
        xt::xtensor<double, 2> Aw = xt::zeros<double>({3*contacts.size(), Nactive*3});
        xt::xtensor<double, 2> B = xt::zeros<double>({contacts.size(), 3*contacts.size()});

        std::size_t ic = 0;
        for (auto &c: contacts)
        {
            auto r_i = c.pi - particles.pos()(c.i);
            auto r_j = c.pj - particles.pos()(c.j);

            xt::xtensor_fixed<double, xt::xshape<3, 3>> ri_cross, rj_cross;

            if (dim == 2)
            {
                ri_cross = {{      0,      0, r_i(1)},
                            {      0,      0, -r_i(0)},
                            {-r_i(1), r_i(0),       0}};

                rj_cross = {{      0,      0,  r_j(1)},
                            {      0,      0, -r_j(0)},
                            {-r_j(1), r_j(0),       0}};
            }
            else
            {
                ri_cross = {{      0, -r_i(2),  r_i(1)},
                            { r_i(2),       0, -r_i(0)},
                            {-r_i(1),  r_i(0),       0}};

                rj_cross = {{      0, -r_j(2),  r_j(1)},
                            { r_j(2),       0, -r_j(0)},
                            {-r_j(1),  r_j(0),       0}};
            }

            auto Ri = scopi::rotation_matrix<3>(particles.q()(c.i));
            auto Rj = scopi::rotation_matrix<3>(particles.q()(c.j));


            if (c.i >= active_ptr)
            {
                std::size_t ind_part = c.i - active_ptr;
                xt::view(Aw, xt::range(3*ic, 3*(ic+1)), xt::range(3*ind_part, 3*(ind_part+1))) = -xt::linalg::dot(ri_cross, Ri);
            }

            if (c.j >= active_ptr)
            {
                std::size_t ind_part = c.j - active_ptr;
                xt::view(Aw, xt::range(3*ic, 3*(ic+1)), xt::range(3*ind_part, 3*(ind_part+1))) =  xt::linalg::dot(rj_cross, Rj);
            }

            for (std::size_t d=0; d<3; ++d)
            {
                if (c.i >= active_ptr)
                {
                    Au(ic*3 + d, (c.i - active_ptr)*3 + d) = 1;
                }
                if (c.j >= active_ptr)
                {
                    Au(ic*3 + d, (c.j - active_ptr)*3 + d) = -1;
                }
                // Aw(ic*3 + d, c.i) = std::pow(-1, d+1) * r_i(dim - d - 1);
                // Aw(ic*3 + d, c.j) = std::pow(-1, d) * r_j(dim - d - 1);
            }

            for (std::size_t d=0; d<dim; ++d)
            {
                B(ic, ic*3 + d) = c.nij[d];
            }
            ++ic;
        }

        auto dtBAu = xt::eval(dt*xt::linalg::dot(B, Au));
        auto dtBAw = xt::eval(dt*xt::linalg::dot(B, Aw));

        // Create Mosek optimization problem
        Model::t model = new Model("contact"); auto _M = finally([&]() { model->dispose(); });
        // variables
        Variable::t u = model->variable("u", 3*Nactive);
        Variable::t w = model->variable("w", 3*Nactive);
        Variable::t su = model->variable("su", 1);
        Variable::t sw = model->variable("sw", 1);

        // functional to minimize
        auto MU_mosek = std::make_shared<ndarray<double, 1>>(MU.data(), shape_t<1>({MU.shape(0)}));
        auto JW_mosek = std::make_shared<ndarray<double, 1>>(JW.data(), shape_t<1>({JW.shape(0)}));
        model->objective("minvar", ObjectiveSense::Minimize, Expr::add(Expr::sub(su, Expr::dot(MU_mosek, u)), Expr::sub(sw, Expr::dot(JW_mosek, w))));

        // constraints
        auto sqrtM_mosek = std::make_shared<ndarray<double, 2>>(sqrtM.data(), shape_t<2>({sqrtM.shape(0), sqrtM.shape(1)}));
        auto sqrtJ_mosek = std::make_shared<ndarray<double, 2>>(sqrtJ.data(), shape_t<2>({sqrtJ.shape(0), sqrtJ.shape(1)}));

        auto D_mosek = std::make_shared<ndarray<double, 1>>(distances.data(), shape_t<1>({distances.shape(0)}));
        auto dtBAu_mosek = std::make_shared<ndarray<double, 2>>(dtBAu.data(), shape_t<2>({dtBAu.shape(0), dtBAu.shape(1)}));
        auto dtBAw_mosek = std::make_shared<ndarray<double, 2>>(dtBAw.data(), shape_t<2>({dtBAw.shape(0), dtBAw.shape(1)}));

        Constraint::t qc1 = model->constraint("qc1", Expr::add(Expr::add(D_mosek, Expr::mul(dtBAu_mosek, u)), Expr::mul(dtBAw_mosek, w)), Domain::greaterThan(0.));
        Constraint::t qc2 = model->constraint("qc2", Expr::vstack(1, su, Expr::mul(sqrtM_mosek, u)), Domain::inRotatedQCone());
        Constraint::t qc3 = model->constraint("qc3", Expr::vstack(1, sw, Expr::mul(sqrtJ_mosek, w)), Domain::inRotatedQCone());

        //solve
        model->solve();

        // move the active particles
        ndarray<double, 1> ulvl   = *(u->level());
        ndarray<double, 1> wlvl   = *(w->level());

        using position_type = typename scopi::scopi_container<dim>::position_type;
        auto uadapt = xt::adapt(reinterpret_cast<double*>(ulvl.raw()), {particles.size()-active_ptr, 3UL});
        auto wadapt = xt::adapt(reinterpret_cast<double*>(wlvl.raw()), {particles.size()-active_ptr, 3UL});
        std::cout << "uadapt = " << uadapt << std::endl;
        std::cout << "pos = " << particles.pos() << std::endl << std::endl;

        for (std::size_t i=0; i<Nactive; ++i)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, wadapt(i, 2)});
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            scopi::type::quaternion expw;
            expw(0) = std::cos(0.5*normw*dt);
            xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*dt)/normw*w;
            for (std::size_t d=0; d<dim; ++d)
            {
                particles.pos()(i + active_ptr)(d) += dt*uadapt(i, d);
            }
            // xt::view(particles.pos(), i) += dt*xt::view(uadapt, i);

            // particles.q()(i) = scopi::quaternion(theta(i));
            // std::cout << expw << " " << particles.q()(i) << std::endl;
            particles.q()(i + active_ptr) = scopi::mult_quaternion(particles.q()(i + active_ptr), expw);
            // std::cout << particles.q()(i) << std::endl << std::endl;

        }
    }
}

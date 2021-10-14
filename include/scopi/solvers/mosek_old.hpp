#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include <fmt/format.h>
#include <fusion.h>
#include <nlohmann/json.hpp>

#include "../container.hpp"
#include "../objects/methods/closest_points.hpp"
#include "../objects/methods/write_objects.hpp"
#include "../objects/neighbor.hpp"
#include "../quaternion.hpp"

#include <nanoflann.hpp>

using namespace mosek::fusion;
using namespace monty;
namespace nl = nlohmann;

using namespace xt::placeholders;


namespace scopi
{
  ///////////////////////
  // KdTree definition //
  ///////////////////////
  template<std::size_t dim>
  class KdTree
  {
    public:
      KdTree(scopi::scopi_container<dim> &p, std::size_t actptr) : _p{p}, _actptr{actptr} {}
      inline std::size_t kdtree_get_point_count() const
      {
        //std::cout << "KDTREE _p.size() = "<< _p.size() <<std::endl;
        return _p.size();
      }
      inline double kdtree_get_pt(std::size_t idx, const std::size_t d) const
      {
        //std::cout << "KDTREE _p["<< _actptr+idx << "][" << d << "] = " << _p.pos()(_actptr+idx)[d] << std::endl;
        return _p.pos()(_actptr+idx)(d); //_p[idx]->pos()[d];
        // return _p.pos()(idx)[d];
      }
      template<class BBOX>
      bool kdtree_get_bbox(BBOX & /* bb */) const
      {
        return false;
      }
    private:
      scopi::scopi_container<dim> &_p;
      std::size_t _actptr;
  };



template<std::size_t dim>
void mosek_solver(scopi::scopi_container<dim>& particles, double dt, std::size_t total_it, std::size_t active_ptr)
{
    std::size_t Nactive = particles.size() - active_ptr;
    // Time Loop
    for (std::size_t nite=0; nite<total_it; ++nite)
    {
      std::cout << "\n\n------------------- Time iteration ----------------> " << nite << std::endl;
        std::vector<scopi::neighbor<dim>> contacts;
        double dmax = 2;

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
        std::cout << "----> create list of contacts " << nite << std::endl;

        // utilisation de kdtree pour ne rechercher les contacts que pour les particules proches
        tic();
        using my_kd_tree_t = typename nanoflann::KDTreeSingleIndexAdaptor<
          nanoflann::L2_Simple_Adaptor<double, KdTree<dim>>, KdTree<dim>, dim >;
        KdTree<dim> kd(particles,active_ptr);
        my_kd_tree_t index(
          dim, kd,
          nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)
        );
        index.buildIndex();
        auto duration = toc();
        std::cout << "----> CPUTIME : build kdtree index = " << duration << std::endl;

        tic();
        #pragma omp parallel for num_threads(8)

        for(std::size_t i = 0; i < particles.size() - 1; ++i)
        {

            // for(std::size_t j = i + 1; j < particles.size(); ++j)
            // {
            //     auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
            //     if (neigh.dij < dmax)
            //     {
            //         contacts.emplace_back(std::move(neigh));
            //         contacts.back().i = i;
            //         contacts.back().j = j;
            //     }
            // }

            double query_pt[dim];
            for (std::size_t d=0; d<dim; ++d)
            {
                query_pt[d] = particles.pos()(active_ptr + i)(d);
                // query_pt[d] = particles.pos()(i)(d);
            }
            //std::cout << "i = " << i << " query_pt = " << query_pt[0] << " " << query_pt[1] << std::endl;

            std::vector<std::pair<size_t, double>> indices_dists;
            double radius = 4;

            nanoflann::RadiusResultSet<double, std::size_t> resultSet(
                radius, indices_dists);

              std::vector<std::pair<unsigned long, double>> ret_matches;

              const std::size_t nMatches = index.radiusSearch(query_pt, radius, ret_matches,
                  nanoflann::SearchParams());

              //std::cout << i << " nMatches = " << nMatches << std::endl;

              for (std::size_t ic = 0; ic < nMatches; ++ic) {

                std::size_t j = ret_matches[ic].first;
                //double dist = ret_matches[ic].second;
                if (i != j) {
                  auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                  if (neigh.dij < dmax) {
                      neigh.i = i;
                      neigh.j = j;
                      #pragma omp critical
                      contacts.emplace_back(std::move(neigh));
                      // contacts.back().i = i;
                      // contacts.back().j = j;
                  }
                }

              }

        }
        duration = toc();
        std::cout << "----> CPUTIME : compute " << contacts.size() << " contacts = " << duration << std::endl;

        tic();
        std::sort(contacts.begin(), contacts.end(), [](auto& a, auto& b )
        {
            return a.i < b.i && a.j < b.j;
        });
        duration = toc();
        std::cout << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration << std::endl;
        //exit(0);



        // output files
        std::cout << "----> json output files " << nite << std::endl;

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
        std::cout << "----> create mass and inertia matrices " << nite << std::endl;
        tic();
        double mass = 1.;
        double moment = .1;

        xt::xtensor<double, 1> MU = xt::zeros<double>({3*Nactive});
        xt::xtensor<double, 1> JW = xt::zeros<double>({3*Nactive});
        xt::xtensor<double, 2> sqrtM = std::sqrt(mass)*xt::eye<double>(3*Nactive);
        xt::xtensor<double, 2> sqrtJ = std::sqrt(moment)*xt::eye<double>(3*Nactive);

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
        // std::cout << "distances " << distances << std::endl;

        //Create contstaint matrices A and B
        // xt::xtensor<double, 2> Au = xt::zeros<double>({3*contacts.size(), Nactive*3});
        // // xt::xtensor<double, 2> Aw = xt::zeros<double>({dim*contacts.size(), {(dim ==3)?Nactive*dim:Nactive}});
        // xt::xtensor<double, 2> Aw = xt::zeros<double>({3*contacts.size(), Nactive*3});
        // xt::xtensor<double, 2> B = xt::zeros<double>({contacts.size(), 3*contacts.size()});

        // Preallocate
        std::vector<int> Au_rows;
        std::vector<int> Au_cols;
        std::vector<double> Au_values;

        Au_rows.reserve(3*contacts.size()*2);
        Au_cols.reserve(3*contacts.size()*2);
        Au_values.reserve(3*contacts.size()*2);

        std::vector<int> Aw_rows;
        std::vector<int> Aw_cols;
        std::vector<double> Aw_values;

        Aw_rows.reserve(3*contacts.size()*6);
        Aw_cols.reserve(3*contacts.size()*6);
        Aw_values.reserve(3*contacts.size()*6);

        // std::vector<int> B_rows;
        // std::vector<int> B_cols;
        // std::vector<double> B_values;

        // B_rows.reserve(contacts.size()*3);
        // B_cols.reserve(contacts.size()*3);
        // B_values.reserve(contacts.size()*3);

        std::size_t ic = 0;
        for (auto &c: contacts)
        {

            auto r_i = c.pi - particles.pos()(c.i);
            auto r_j = c.pj - particles.pos()(c.j);
            // auto r_i = xt::eval(c.pi - particles.pos()(c.i));
            // auto r_j = xt::eval(c.pj - particles.pos()(c.j));

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
                auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                for (std::size_t ip=0; ip<3; ++ip)
                {
                    Aw_rows.push_back(ic);
                    Aw_cols.push_back(3*ind_part + ip);
                    Aw_values.push_back(-dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
                // xt::view(Aw, xt::range(3*ic, 3*(ic+1)), xt::range(3*ind_part, 3*(ind_part+1))) = -xt::linalg::dot(ri_cross, Ri);
            }

            if (c.j >= active_ptr)
            {
                std::size_t ind_part = c.j - active_ptr;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip=0; ip<3; ++ip)
                {
                    Aw_rows.push_back(ic);
                    Aw_cols.push_back(3*ind_part + ip);
                    Aw_values.push_back(dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
                // xt::view(Aw, xt::range(3*ic, 3*(ic+1)), xt::range(3*ind_part, 3*(ind_part+1))) =  xt::linalg::dot(rj_cross, Rj);
            }

            for (std::size_t d=0; d<3; ++d)
            {
                if (c.i >= active_ptr)
                {
                    Au_rows.push_back(ic);
                    Au_cols.push_back((c.i - active_ptr)*3 + d);
                    Au_values.push_back(dt*c.nij[d]);
                    // Au(ic*3 + d, (c.i - active_ptr)*3 + d) = 1;
                }
                if (c.j >= active_ptr)
                {
                    Au_rows.push_back(ic);
                    Au_cols.push_back((c.j - active_ptr)*3 + d);
                    Au_values.push_back(-dt*c.nij[d]);
                    // Au(ic*3 + d, (c.j - active_ptr)*3 + d) = -1;
                }
                // Aw(ic*3 + d, c.i) = std::pow(-1, d+1) * r_i(dim - d - 1);
                // Aw(ic*3 + d, c.j) = std::pow(-1, d) * r_j(dim - d - 1);
            }

            // for (std::size_t d=0; d<dim; ++d)
            // {
            //     B_rows.push_back(ic);
            //     B_cols.push_back(ic*3 + d);
            //     B_values.push_back(c.nij[d]);
            //     // B(ic, ic*3 + d) = c.nij[d];
            // }
            ++ic;
        }

        // auto Au = Matrix::sparse(Au_rows.size(), Au_cols.size(),
        auto Au = Matrix::sparse(contacts.size(), 3*Nactive,
                                 std::make_shared<ndarray<int, 1>>(Au_rows.data(), shape_t<1>({Au_rows.size()})),
                                 std::make_shared<ndarray<int, 1>>(Au_cols.data(), shape_t<1>({Au_cols.size()})),
                                 std::make_shared<ndarray<double, 1>>(Au_values.data(), shape_t<1>({Au_values.size()})));
        // auto Aw = Matrix::sparse(Aw_rows.size(), Aw_cols.size(),
        auto Aw = Matrix::sparse(contacts.size(), 3*Nactive,
                                 std::make_shared<ndarray<int, 1>>(Aw_rows.data(), shape_t<1>({Aw_rows.size()})),
                                 std::make_shared<ndarray<int, 1>>(Aw_cols.data(), shape_t<1>({Aw_cols.size()})),
                                 std::make_shared<ndarray<double, 1>>(Aw_values.data(), shape_t<1>({Aw_values.size()})));
        // auto B = Matrix::sparse(B_rows.size(), B_cols.size(),
        //                          std::make_shared<ndarray<int, 1>>(B_rows.data(), shape_t<1>({B_rows.size()})),
        //                          std::make_shared<ndarray<int, 1>>(B_rows.data(), shape_t<1>({B_cols.size()})),
        //                          std::make_shared<ndarray<double, 1>>(B_values.data(), shape_t<1>({B_values.size()})));

        // auto dtBAu = xt::eval(dt*xt::linalg::dot(B, Au));
        // auto dtBAw = xt::eval(dt*xt::linalg::dot(B, Aw));

        // auto dtBAu = Expr::mul(dt, Au);
        auto duration4 = toc();
        std::cout << "----> CPUTIME : matrices = " << duration4 << std::endl;


        // Create Mosek optimization problem
        std::cout << "----> Create Mosek optimization problem " << nite << std::endl;
        tic();
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
        // auto dtBAu_mosek = std::make_shared<ndarray<double, 2>>(dtBAu.data(), shape_t<2>({dtBAu.shape(0), dtBAu.shape(1)}));
        // auto dtBAw_mosek = std::make_shared<ndarray<double, 2>>(dtBAw.data(), shape_t<2>({dtBAw.shape(0), dtBAw.shape(1)}));

        Constraint::t qc1 = model->constraint("qc1", Expr::add(Expr::add(D_mosek, Expr::mul(Au, u)), Expr::mul(Aw, w)), Domain::greaterThan(0.));
        Constraint::t qc2 = model->constraint("qc2", Expr::vstack(1, su, Expr::mul(sqrtM_mosek, u)), Domain::inRotatedQCone());
        Constraint::t qc3 = model->constraint("qc3", Expr::vstack(1, sw, Expr::mul(sqrtJ_mosek, w)), Domain::inRotatedQCone());

        //solve
        model->solve();

        auto duration5 = toc();
        std::cout << "----> CPUTIME : mosek = " << duration5 << std::endl;

        // move the active particles
        ndarray<double, 1> ulvl   = *(u->level());
        ndarray<double, 1> wlvl   = *(w->level());

        auto uadapt = xt::adapt(reinterpret_cast<double*>(ulvl.raw()), {particles.size()-active_ptr, 3UL});
        auto wadapt = xt::adapt(reinterpret_cast<double*>(wlvl.raw()), {particles.size()-active_ptr, 3UL});
        // std::cout << "uadapt = " << uadapt << std::endl;
        // std::cout << "wadapt = " << wadapt << std::endl;
        // std::cout << "pos = " << particles.pos() << std::endl << std::endl;

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
            normalize(particles.q()(i + active_ptr));
            // std::cout << "position" << particles.pos()(i) << std::endl << std::endl;
            // std::cout << "quaternion " << particles.q()(i) << std::endl << std::endl;

        }
    }
}
}

#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "../container.hpp"
#include "../objects/methods/closest_points.hpp"
#include "../objects/methods/write_objects.hpp"
#include "../objects/neighbor.hpp"
#include "../quaternion.hpp"
#include "MosekSolver.hpp"
#include "ScsSolver.hpp"

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

    template<std::size_t dim, typename SolverType>
        class ScopiSolver
        {
            public:
                ScopiSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t active_ptr);
                void solve(std::size_t total_it);

            private:
                void displacementObstacles();

                std::vector<scopi::neighbor<dim>> computeContacts();
                void sortContacts(std::vector<scopi::neighbor<dim>>& contacts);
                void writeOutputFiles(std::vector<scopi::neighbor<dim>>& contacts, std::size_t nite);
                void moveActiveParticles();

                scopi::scopi_container<dim>& _particles;
                double _dt;
                std::size_t _active_ptr;
                std::size_t _Nactive;
                SolverType _solver;

        };

    template<std::size_t dim, typename SolverType>
        ScopiSolver<dim, SolverType>::ScopiSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t active_ptr) : 
            _particles(particles),
            _dt(dt),
            _active_ptr(active_ptr),
            _Nactive(_particles.size() - _active_ptr),
            _solver(_particles, _dt, _Nactive, _active_ptr)
    {
    }

    template<std::size_t dim, typename SolverType>
        void ScopiSolver<dim, SolverType>::solve(std::size_t total_it)
        {
            // Time Loop
            for (std::size_t nite=0; nite<total_it; ++nite)
            {
                std::cout << "\n\n------------------- Time iteration ----------------> " << nite << std::endl;

                //displacement of obstacles
                displacementObstacles();

                // create list of contacts
                std::cout << "----> create list of contacts " << nite << std::endl;
                auto contacts = computeContacts();

                tic();
                sortContacts(contacts);
                auto duration = toc();
                std::cout << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration << std::endl;

                // output files
                std::cout << "----> json output files " << nite << std::endl;
                writeOutputFiles(contacts, nite);

                // for (std::size_t i=0; i<_Nactive; ++i)
                // {
                //     for (std::size_t d=0; d<dim; ++d)
                //     {
                //         particles.pos()(i + active_ptr)(d) += dt*particles.vd()(i + active_ptr)(d);
                //     }
                // }
                //

                // create mass and inertia matrices
                tic();
                _solver.allocateMemory(contacts.size());
                _solver.createMatrixConstraint(contacts);
                _solver.createMatrixMass();
                _solver.createVectorC();
                _solver.createVectorDistances(contacts);
                auto duration4 = toc();
                std::cout << "----> CPUTIME : matrices = " << duration4 << std::endl;

                // Solve optimization problem
                std::cout << "----> Create optimization problem " << nite << std::endl;
                tic();
                auto nbIter = _solver.solveOptimizationProbelm(contacts);
                auto duration5 = toc();
                std::cout << "----> CPUTIME : solve = " << duration5 << std::endl;
                std::cout << "iterations : " << nbIter << std::endl;

                // move the active particles
                moveActiveParticles();

                // free the memory for the next solve
                _solver.freeMemory();
            }
        }

    template<std::size_t dim, typename SolverType>
        void ScopiSolver<dim, SolverType>::displacementObstacles()
        {
            for (std::size_t i=0; i<_active_ptr; ++i)
            {
                xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, _particles.desired_omega()(i)});
                double normw = xt::linalg::norm(w);
                if (normw == 0)
                {
                    normw = 1;
                }
                scopi::type::quaternion expw;
                expw(0) = std::cos(0.5*normw*_dt);
                xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*_dt)/normw*w;

                for (std::size_t d=0; d<dim; ++d)
                {
                    _particles.pos()(i)(d) += _dt*_particles.vd()(i)(d);
                }
                _particles.q()(i) = scopi::mult_quaternion(_particles.q()(i), expw);

                std::cout << "obstacle " << i << ": " << _particles.pos()(0) << " " << _particles.q()(0) << std::endl;
            }
        }

    template<std::size_t dim, typename SolverType>
        std::vector<scopi::neighbor<dim>> ScopiSolver<dim, SolverType>::computeContacts()
        {
            double dmax = 2;
            std::vector<scopi::neighbor<dim>> contacts;

            tic();
            // utilisation de kdtree pour ne rechercher les contacts que pour les particules proches
            using my_kd_tree_t = typename nanoflann::KDTreeSingleIndexAdaptor<
                nanoflann::L2_Simple_Adaptor<double, KdTree<dim>>, KdTree<dim>, dim >;
            KdTree<dim> kd(_particles,_active_ptr);
            my_kd_tree_t index(
                    dim, kd,
                    nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)
                    );
            index.buildIndex();
            auto duration = toc();
            std::cout << "----> CPUTIME : build kdtree index = " << duration << std::endl;

            tic();
#pragma omp parallel for num_threads(8)

            for(std::size_t i = 0; i < _particles.size() - 1; ++i)
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
                    query_pt[d] = _particles.pos()(_active_ptr + i)(d);
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
                        auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*_particles[i], *_particles[j]);
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
            return contacts;
        }

    template<std::size_t dim, typename SolverType>
        void ScopiSolver<dim, SolverType>::sortContacts(std::vector<scopi::neighbor<dim>>& contacts)
        {
            std::sort(contacts.begin(), contacts.end(), [](auto& a, auto& b )
                    {
                    return a.i < b.i && a.j < b.j;
                    });
            //exit(0);
        }

    template<std::size_t dim, typename SolverType>
        void ScopiSolver<dim, SolverType>::writeOutputFiles(std::vector<scopi::neighbor<dim>>& contacts, std::size_t nite)
        {
            nl::json json_output;

            std::ofstream file(fmt::format("./Results/scopi_objects_{:04d}.json", nite));

            json_output["objects"] = {};

            for(std::size_t i = 0; i < _particles.size(); ++i)
            {
                json_output["objects"].push_back(scopi::write_objects_dispatcher<dim>::dispatch(*_particles[i]));
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
        }

    template<std::size_t dim, typename SolverType>
        void ScopiSolver<dim, SolverType>::moveActiveParticles()
        {

            auto uadapt = _solver.getUadapt();
            auto wadapt = _solver.getWadapt();

            for (std::size_t i=0; i<_Nactive; ++i)
            {
                xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, wadapt(i, 2)});
                double normw = xt::linalg::norm(w);
                if (normw == 0)
                {
                    normw = 1;
                }
                scopi::type::quaternion expw;
                expw(0) = std::cos(0.5*normw*_dt);
                xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*_dt)/normw*w;
                for (std::size_t d=0; d<dim; ++d)
                {
                    _particles.pos()(i + _active_ptr)(d) += _dt*uadapt(i, d);
                }
                // xt::view(particles.pos(), i) += dt*xt::view(uadapt, i);

                // particles.q()(i) = scopi::quaternion(theta(i));
                // std::cout << expw << " " << particles.q()(i) << std::endl;
                _particles.q()(i + _active_ptr) = scopi::mult_quaternion(_particles.q()(i + _active_ptr), expw);
                normalize(_particles.q()(i + _active_ptr));
                // std::cout << "position" << particles.pos()(i) << std::endl << std::endl;
                // std::cout << "quaternion " << particles.q()(i) << std::endl << std::endl;

            }
        }
}


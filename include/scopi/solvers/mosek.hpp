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

#include "OptimMosek.hpp"
#include "OptimScs.hpp"
#include "OptimUzawa.hpp"

#include <scopi/contact/contact_kdtree.hpp>
// #include <scopi/contact/contact_brute_force.hpp>

#include "../vap/vap_fixed.hpp"

#include <nanoflann.hpp>

using namespace mosek::fusion;
using namespace monty;
namespace nl = nlohmann;

using namespace xt::placeholders;

namespace scopi
{

    template<std::size_t dim>
        class ScopiSolver
        {
            public:
                ScopiSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t active_ptr);
                void solve(std::size_t total_it);

            private:
                void displacementObstacles();

                std::vector<scopi::neighbor<dim>> computeContacts();
                void writeOutputFiles(std::vector<scopi::neighbor<dim>>& contacts, std::size_t nite);
                void moveActiveParticles();
                scopi::scopi_container<dim>& _particles;
                double _dt;
                std::size_t _active_ptr;
                std::size_t _Nactive;
                OptimUzawa<dim> _solver;
                vap_fixed _vap;

        };

    template<std::size_t dim>
        ScopiSolver<dim>::ScopiSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t active_ptr) : 
            _particles(particles),
            _dt(dt),
            _active_ptr(active_ptr),
            _Nactive(_particles.size() - _active_ptr),
            _solver(_particles, _dt, _Nactive, _active_ptr),
            _vap(_Nactive, _active_ptr, _dt)
    {
    }

    template<std::size_t dim>
        void ScopiSolver<dim>::solve(std::size_t total_it)
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
                _vap.aPrioriVelocity(_particles);

                // solver optimization problem
                _solver.run(contacts, nite);

                // move the active particles
                moveActiveParticles();

                _vap.updateVelocity(_particles);

                // free the memory for the next solve
                _solver.freeMemory();
            }
        }

    template<std::size_t dim>
        void ScopiSolver<dim>::displacementObstacles()
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

    template<std::size_t dim>
        std::vector<scopi::neighbor<dim>> ScopiSolver<dim>::computeContacts()
        {
            // // scopi::contact_brute_force cont(2);
            scopi::contact_kdtree cont(2, 10);
            auto contacts = cont.run(_particles, _active_ptr);
            std::cout << "----> MOSEK : contacts.size() = " << contacts.size() << std::endl;
            return contacts;
        }

    template<std::size_t dim>
        void ScopiSolver<dim>::writeOutputFiles(std::vector<scopi::neighbor<dim>>& contacts, std::size_t nite)
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

    template<std::size_t dim>
        void ScopiSolver<dim>::moveActiveParticles()
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


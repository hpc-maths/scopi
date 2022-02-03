#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "container.hpp"
#include "objects/methods/closest_points.hpp"
#include "objects/methods/write_objects.hpp"
#include "objects/neighbor.hpp"
#include "quaternion.hpp"

#include "solvers/OptimUzawaMatrixFreeOmp.hpp"
#include "contact/contact_kdtree.hpp"
#include "vap/vap_fixed.hpp"

namespace nl = nlohmann;

using namespace xt::placeholders;

namespace scopi
{

    template<std::size_t dim,
             template<std::size_t> class optim_solver_t = OptimUzawaMatrixFreeOmp,
             class contact_t = contact_kdtree,
             class vap_t = vap_fixed
             >
    class ScopiSolver
    {
    public:
        ScopiSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t active_ptr);
        void solve(std::size_t total_it);
        std::string get_optim_solver_name() const;

    private:
        void displacement_obstacles();

        std::vector<scopi::neighbor<dim>> compute_contacts();
        void write_output_files(std::vector<scopi::neighbor<dim>>& contacts, std::size_t nite);
        void move_active_particles();
        scopi::scopi_container<dim>& _particles;
        double m_dt;
        std::size_t m_active_ptr;
        std::size_t m_Nactive;
        optim_solver_t<dim> m_solver;
        vap_t m_vap;

    };

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::ScopiSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t active_ptr)
    : _particles(particles)
    , m_dt(dt)
    , m_active_ptr(active_ptr)
    , m_Nactive(_particles.size() - m_active_ptr)
    , m_solver(_particles, m_dt, m_Nactive, m_active_ptr)
    , m_vap(m_Nactive, m_active_ptr, m_dt)
    {}

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    std::string ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::get_optim_solver_name() const
    {
        return m_solver.getName();
    }

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::solve(std::size_t total_it)
    {
        // Time Loop
        for (std::size_t nite = 0; nite < total_it; ++nite)
        {
            // std::cout << "\n\n------------------- Time iteration ----------------> " << nite << std::endl;

            displacement_obstacles();

            // create list of contacts
            // std::cout << "----> create list of contacts " << nite << std::endl;
            auto contacts = compute_contacts();

            // output files
            // std::cout << "----> json output files " << nite << std::endl;
            write_output_files(contacts, nite);

            // for (std::size_t i=0; i<m_Nactive; ++i)
            // {
            //     for (std::size_t d=0; d<dim; ++d)
            //     {
            //         particles.pos()(i + active_ptr)(d) += dt*particles.vd()(i + active_ptr)(d);
            //     }
            // }
            //
            m_vap.aPrioriVelocity(_particles);

            // solver optimization problem
            m_solver.run(contacts, nite);

            // move the active particles
            move_active_particles();

            m_vap.updateVelocity(_particles, m_solver.getUadapt(), m_solver.getWadapt());

            // free the memory for the next solve
            m_solver.freeMemory();
        }
    }

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::displacement_obstacles()
    {
        for (std::size_t i = 0; i < m_active_ptr; ++i)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, _particles.desired_omega()(i)});
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            scopi::type::quaternion expw;
            expw(0) = std::cos(0.5*normw*m_dt);
            xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*m_dt)/normw*w;

            for (std::size_t d = 0; d < dim; ++d)
            {
                _particles.pos()(i)(d) += m_dt*_particles.vd()(i)(d);
            }
            _particles.q()(i) = scopi::mult_quaternion(_particles.q()(i), expw);

            // std::cout << "obstacle " << i << ": " << _particles.pos()(0) << " " << _particles.q()(0) << std::endl;
        }
    }

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    std::vector<scopi::neighbor<dim>> ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::compute_contacts()
    {
        // // scopi::contact_brute_force cont(2);
        contact_t cont(2., 10.);
        auto contacts = cont.run(_particles, m_active_ptr);
        // std::cout << "----> MOSEK : contacts.size() = " << contacts.size() << std::endl;
        return contacts;
    }

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::write_output_files(std::vector<scopi::neighbor<dim>>& contacts, std::size_t nite)
    {
        nl::json json_output;

        std::ofstream file(fmt::format("./Results/scopi_objects_{:04d}.json", nite));

        json_output["objects"] = {};

        for(std::size_t i = 0; i < _particles.size(); ++i)
        {
            json_output["objects"].push_back(scopi::write_objects_dispatcher<dim>::dispatch(*_particles[i]));
        }

        json_output["contacts"] = {};

        for(std::size_t i = 0; i < contacts.size(); ++i)
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

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::move_active_particles()
    {

        auto uadapt = m_solver.getUadapt();
        auto wadapt = m_solver.getWadapt();

        for (std::size_t i = 0; i < m_Nactive; ++i)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, wadapt(i, 2)});
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            scopi::type::quaternion expw;
            expw(0) = std::cos(0.5*normw*m_dt);
            xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*m_dt)/normw*w;
            for (std::size_t d = 0; d < dim; ++d)
            {
                _particles.pos()(i + m_active_ptr)(d) += m_dt*uadapt(i, d);
            }
            // xt::view(particles.pos(), i) += dt*xt::view(uadapt, i);

            // particles.q()(i) = scopi::quaternion(theta(i));
            // std::cout << expw << " " << particles.q()(i) << std::endl;
            _particles.q()(i + m_active_ptr) = scopi::mult_quaternion(_particles.q()(i + m_active_ptr), expw);
            normalize(_particles.q()(i + m_active_ptr));
            // std::cout << "position" << particles.pos()(i) << std::endl << std::endl;
            // std::cout << "quaternion " << particles.q()(i) << std::endl << std::endl;

        }
    }
}


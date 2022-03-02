#pragma once

#include <algorithm>
#include <iostream>
#include <fstream>
#include <vector>

#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

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
             class optim_solver_t = OptimUzawaMatrixFreeOmp,
             class contact_t = contact_kdtree,
             class vap_t = vap_fixed
             >
    class ScopiSolver
    {
    public:
        ScopiSolver(scopi_container<dim>& particles, double dt);
        void solve(std::size_t total_it);
        void set_coeff_friction(double mu);

    private:
        void displacement_obstacles();

        std::vector<neighbor<dim>> compute_contacts();
        void write_output_files(std::vector<neighbor<dim>>& contacts, std::size_t nite);
        void move_active_particles();
        scopi_container<dim>& m_particles;
        double m_dt;
        optim_solver_t m_solver;
        vap_t m_vap;
    };

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::ScopiSolver(scopi_container<dim>& particles, double dt)
    : m_particles(particles)
    , m_dt(dt)
    , m_solver(m_particles.nb_active(), m_dt)
    , m_vap(m_particles.nb_active(), m_particles.nb_inactive(), m_dt)
    {}

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::solve(std::size_t total_it)
    {
        // Time Loop
        for (std::size_t nite = 0; nite < total_it; ++nite)
        {
            PLOG_INFO << "\n\n------------------- Time iteration ----------------> " << nite;

            displacement_obstacles();

            PLOG_INFO << "----> create list of contacts " << nite;
            auto contacts = compute_contacts();

            PLOG_INFO << "----> json output files " << nite << std::endl;
            write_output_files(contacts, nite);

            m_vap.set_a_priori_velocity(m_particles);
            m_solver.run(m_particles, contacts, nite);
            move_active_particles();
            m_vap.update_velocity(m_particles, m_solver.get_uadapt(), m_solver.get_wadapt());
        }
    }

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::displacement_obstacles()
    {
        for (std::size_t i = 0; i < m_particles.nb_inactive(); ++i)
        {

            auto  w = get_omega(m_particles.desired_omega()(i));
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            type::quaternion_t expw;
            expw(0) = std::cos(0.5*normw*m_dt);
            xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*m_dt)/normw*w;

            for (std::size_t d = 0; d < dim; ++d)
            {
                m_particles.pos()(i)(d) += m_dt*m_particles.vd()(i)(d);
            }
            m_particles.q()(i) = mult_quaternion(m_particles.q()(i), expw);
            normalize(m_particles.q()(i));
            // std::cout << "obstacle " << i << ": " << m_particles.pos()(0) << " " << m_particles.q()(0) << std::endl;
        }
    }

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    std::vector<neighbor<dim>> ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::compute_contacts()
    {
        // // contact_brute_force cont(2);
        // contact_t cont(2., 10.);
        contact_t cont(2.);
        auto contacts = cont.run(m_particles, m_particles.nb_inactive());
        PLOG_INFO << "contacts.size() = " << contacts.size() << std::endl;
        return contacts;
    }

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::write_output_files(std::vector<neighbor<dim>>& contacts, std::size_t nite)
    {
        nl::json json_output;

        std::ofstream file(fmt::format("./Results/scopi_objects_{:04d}.json", nite));

        json_output["objects"] = {};

        for (std::size_t i = 0; i < m_particles.size(); ++i)
        {
            json_output["objects"].push_back(write_objects_dispatcher<dim>::dispatch(*m_particles[i]));
        }

        json_output["contacts"] = {};

        for (std::size_t i = 0; i < contacts.size(); ++i)
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

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::move_active_particles()
    {
        std::size_t active_offset = m_particles.nb_inactive();
        auto uadapt = m_solver.get_uadapt();
        auto wadapt = m_solver.get_wadapt();

        for (std::size_t i = 0; i < m_particles.nb_active(); ++i)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, wadapt(i, 2)});
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            type::quaternion_t expw;
            expw(0) = std::cos(0.5*normw*m_dt);
            xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*m_dt)/normw*w;
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_particles.pos()(i + active_offset)(d) += m_dt*uadapt(i, d);
            }

            m_particles.q()(i + active_offset) = mult_quaternion(m_particles.q()(i + active_offset), expw);
            normalize(m_particles.q()(i + active_offset));
        }
    }

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::set_coeff_friction(double mu)
    {
        m_solver.set_coeff_friction(mu);
    }
}


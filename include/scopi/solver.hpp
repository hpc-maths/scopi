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
             template<std::size_t> class optim_solver_t = OptimUzawaMatrixFreeOmp,
             class contact_t = contact_kdtree,
             class vap_t = vap_fixed
             >
    class ScopiSolver
    {
    public:
        ScopiSolver(scopi_container<dim>& particles, double dt, std::size_t active_ptr);
        void solve(std::size_t total_it);

    private:
        void displacement_obstacles();

        std::vector<neighbor<dim>> compute_contacts();
        void write_output_files(std::vector<neighbor<dim>>& contacts, std::size_t nite);
        void move_active_particles();
        scopi_container<dim>& m_particles;
        double m_dt;
        std::size_t m_active_ptr;
        std::size_t m_Nactive;
        optim_solver_t<dim> m_solver;
        vap_t m_vap;

    };

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::ScopiSolver(scopi_container<dim>& particles, double dt, std::size_t active_ptr)
    : m_particles(particles)
    , m_dt(dt)
    , m_active_ptr(active_ptr)
    , m_Nactive(m_particles.size() - m_active_ptr)
    , m_solver(m_particles, m_dt, m_Nactive, m_active_ptr)
    , m_vap(m_Nactive, m_active_ptr, m_dt)
    {}

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
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

            // for (std::size_t i=0; i<m_Nactive; ++i)
            // {
            //     for (std::size_t d=0; d<dim; ++d)
            //     {
            //         particles.pos()(i + active_ptr)(d) += dt*particles.vd()(i + active_ptr)(d);
            //     }
            // }
            //
            m_vap.set_a_priori_velocity(m_particles);

            // solver optimization problem
            m_solver.run(contacts, nite);

            // move the active particles
            move_active_particles();

            m_vap.update_velocity(m_particles, m_solver.get_uadapt(), m_solver.get_wadapt());
        }
    }

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::displacement_obstacles()
    {
        for (std::size_t i = 0; i < m_active_ptr; ++i)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, m_particles.desired_omega()(i)});
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            type::quaternion expw;
            expw(0) = std::cos(0.5*normw*m_dt);
            xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*m_dt)/normw*w;

            for (std::size_t d = 0; d < dim; ++d)
            {
                m_particles.pos()(i)(d) += m_dt*m_particles.vd()(i)(d);
            }
            m_particles.q()(i) = mult_quaternion(m_particles.q()(i), expw);

            // std::cout << "obstacle " << i << ": " << m_particles.pos()(0) << " " << m_particles.q()(0) << std::endl;
        }
    }

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    std::vector<neighbor<dim>> ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::compute_contacts()
    {
        // // contact_brute_force cont(2);
        // contact_t cont(2., 10.);
        contact_t cont(2.);
        auto contacts = cont.run(m_particles, m_active_ptr);
        PLOG_INFO << "contacts.size() = " << contacts.size() << std::endl;
        return contacts;
    }

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
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

    template<std::size_t dim, template<std::size_t> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::move_active_particles()
    {

        auto uadapt = m_solver.get_uadapt();
        auto wadapt = m_solver.get_wadapt();

        for (std::size_t i = 0; i < m_Nactive; ++i)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, wadapt(i, 2)});
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            type::quaternion expw;
            expw(0) = std::cos(0.5*normw*m_dt);
            xt::view(expw, xt::range(1, _)) = std::sin(0.5*normw*m_dt)/normw*w;
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_particles.pos()(i + m_active_ptr)(d) += m_dt*uadapt(i, d);
            }
            // xt::view(particles.pos(), i) += dt*xt::view(uadapt, i);

            // particles.q()(i) = quaternion(theta(i));
            // std::cout << expw << " " << particles.q()(i) << std::endl;
            m_particles.q()(i + m_active_ptr) = mult_quaternion(m_particles.q()(i + m_active_ptr), expw);
            normalize(m_particles.q()(i + m_active_ptr));
            // std::cout << "position" << particles.pos()(i) << std::endl << std::endl;
            // std::cout << "quaternion " << particles.q()(i) << std::endl << std::endl;

        }
    }
}


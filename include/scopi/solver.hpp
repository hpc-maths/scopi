#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
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
#include "solvers/OptimMosek.hpp"
#include "problems/DryWithoutFriction.hpp"
#include "problems/ViscousWithFriction.hpp"
#include "contact/contact_kdtree.hpp"
#include "vap/vap_fixed.hpp"
#include "vap/vap_projection.hpp"
#include "params/OptimParams.hpp"
#include "params/ProblemParams.hpp"

namespace nl = nlohmann;

using namespace xt::placeholders;

namespace scopi
{
    template<std::size_t dim,
             class optim_solver_t = OptimUzawaMatrixFreeOmp<DryWithoutFriction>,
             class contact_t = contact_kdtree,
             class vap_t = vap_fixed
             >
    class ScopiSolver : public optim_solver_t
    {
    public:
        using solver_type = optim_solver_t;
    private:
        using problem_t = typename optim_solver_t::problem_type;
    public:
        ScopiSolver(scopi_container<dim>& particles,
                    double dt,
                    const OptimParams<optim_solver_t>& optim_params = OptimParams<optim_solver_t>());
        void solve(std::size_t total_it, std::size_t initial_iter = 0);

    private:
        void displacement_obstacles();

        std::vector<neighbor<dim>> compute_contacts();
        std::vector<neighbor<dim>> compute_contacts_worms();
        void write_output_files(std::vector<neighbor<dim>>& contacts, std::size_t nite);
        void move_active_particles();
        scopi_container<dim>& m_particles;
        double m_dt;
        problem_t m_problem;
        vap_t m_vap;
    };

    /*
    template<std::size_t dim,
             class optim_solver_t,
             class contact_t,
             class vap_t
             >
    class ScopiSolver<dim, optim_solver_t, contact_t, vap_t, 2>: public ScopiSolverBase<dim, optim_solver_t, contact_t, vap_t>
    {
    public:
        using problem_t = typename optim_solver_t::problem_type;
        using solver_type = optim_solver_t;
        ScopiSolver(scopi_container<dim>& particles,
                    double dt,
                    const OptimParams<optim_solver_t>& optim_params = OptimParams<optim_solver_t>());
        void solve(std::size_t total_it, std::size_t initial_iter = 0);
    private:
        vap_projection m_vap_projection;
    };
    */

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::ScopiSolver(scopi_container<dim>& particles, double dt, const OptimParams<optim_solver_t>& optim_params)
    : optim_solver_t(particles.nb_active(), dt, particles, optim_params)
    , m_particles(particles)
    , m_dt(dt)
    , m_problem(m_particles.nb_active(), m_dt, optim_params.m_problem_params)
    , m_vap(m_particles.nb_active(), m_particles.nb_inactive(), m_dt)
    {}

    /*
    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    ScopiSolver<dim, optim_solver_t, contact_t, vap_t, 2>::ScopiSolver(scopi_container<dim>& particles, double dt, const OptimParams<optim_solver_t>& optim_params)
    : ScopiSolverBase<dim, optim_solver_t, contact_t, vap_t>(particles, dt, optim_params)
    , m_vap_projection(this->m_particles.nb_active(), this->m_particles.nb_inactive(), this->m_dt)
    {}
    */

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::solve(std::size_t total_it, std::size_t initial_iter)
    {
        // Time Loop
        for (std::size_t nite = initial_iter; nite < total_it; ++nite)
        {
            PLOG_INFO << "\n\n------------------- Time iteration ----------------> " << nite;

            displacement_obstacles();
            auto contacts = compute_contacts();
            auto contacts_worms = compute_contacts_worms();
            write_output_files(contacts, nite);
            m_vap.set_a_priori_velocity(m_particles);
            m_problem.extra_setps_before_solve(contacts);
            this->run(m_particles, contacts, contacts_worms, m_problem, nite);
            m_problem.extra_setps_after_solve(contacts, this->get_lagrange_multiplier(contacts, contacts_worms, m_problem));
            move_active_particles();
            m_vap.update_velocity(m_particles, this->get_uadapt(), this->get_wadapt());
        }
    }

    /*
    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::solve(std::size_t total_it, std::size_t initial_iter)
    {
        // Time Loop
        for (std::size_t nite = initial_iter; nite < total_it; ++nite)
        {
            PLOG_INFO << "\n\n------------------- Time iteration ----------------> " << nite;

            this->displacement_obstacles();
            auto contacts = this->compute_contacts();

            tic();
            this->m_problem.set_gamma(contacts);
            duration = toc();
            PLOG_INFO << "----> CPUTIME : set gamma = " << duration;

            this->write_output_files(contacts, nite);
            this->m_vap.set_a_priori_velocity(this->m_particles);

            this->m_problem.setup_first_resolution();
            this->m_solver.run(this->m_particles, contacts, this->m_problem, nite);

            tic();
            this->m_problem.correct_lambda(contacts, this->m_solver.get_lagrange_multiplier(contacts, this->m_particles, this->m_problem), this->m_particles, this->m_solver.get_uadapt());
            duration = toc();
            PLOG_INFO << "----> CPUTIME : update gamma = " << duration;

            m_vap_projection.set_u_w(this->m_solver.get_uadapt(), this->m_solver.get_wadapt());
            m_vap_projection.set_a_priori_velocity(this->m_particles);
            this->m_problem.setup_projection();
            this->m_solver.run(this->m_particles, contacts, this->m_problem,  nite);

            tic();
            this->m_problem.update_gamma(contacts, this->m_solver.get_lagrange_multiplier(contacts, this->m_particles, this->m_problem));
            duration = toc();
            PLOG_INFO << "----> CPUTIME : update gamma = " << duration;

            this->move_active_particles();
            this->m_vap.update_velocity(this->m_particles, this->m_solver.get_uadapt(), this->m_solver.get_wadapt());
        }
    }
    */

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::displacement_obstacles()
    {
        tic();
        for (std::size_t i = 0; i < m_particles.nb_inactive(); ++i)
        {

            auto  w = get_omega(m_particles.desired_omega()(i));
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            type::quaternion_t expw;
            auto expw_adapt = xt::adapt(expw);
            expw_adapt(0) = std::cos(0.5*normw*m_dt);
            xt::view(expw_adapt, xt::range(1, _)) = std::sin(0.5*normw*m_dt)/normw*w;

            for (std::size_t d = 0; d < dim; ++d)
            {
                m_particles.pos()(i)(d) += m_dt*m_particles.vd()(i)(d);
            }
            m_particles.q()(i) = mult_quaternion(m_particles.q()(i), expw);
            normalize(m_particles.q()(i));
            // std::cout << "obstacle " << i << ": " << m_particles.pos()(0) << " " << m_particles.q()(0) << std::endl;
        }
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : obstacles = " << duration;
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
    std::vector<neighbor<dim>> ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::compute_contacts_worms()
    {
        std::vector<neighbor<dim>> contacts;
        #pragma omp parallel for
        for (std::size_t i = 0; i < m_particles.size(); ++i)
        {
            for (std::size_t j = 0; j < m_particles[i]->size()-1; ++j)
            {
                auto neigh = closest_points_dispatcher<dim>::dispatch(*select_object_dispatcher<dim>::dispatch(*m_particles[i], index(j  )),
                                                                      *select_object_dispatcher<dim>::dispatch(*m_particles[i], index(j+1)));
                neigh.i = m_particles.offset(i) + j;
                neigh.j = m_particles.offset(i) + j + 1;
                #pragma omp critical
                contacts.emplace_back(std::move(neigh));
            }
        }
        return contacts;
    }

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::write_output_files(std::vector<neighbor<dim>>& contacts, std::size_t nite)
    {
        tic();
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

        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : write output files = " << duration;
    }

    template<std::size_t dim, class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::move_active_particles()
    {
        tic();
        std::size_t active_offset = m_particles.nb_inactive();
        auto uadapt = this->get_uadapt();
        auto wadapt = this->get_wadapt();

        for (std::size_t i = 0; i < m_particles.nb_active(); ++i)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> w({0, 0, wadapt(i, 2)});
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            type::quaternion_t expw;
            auto expw_adapt = xt::adapt(expw);
            expw_adapt(0) = std::cos(0.5*normw*m_dt);
            xt::view(expw_adapt, xt::range(1, _)) = std::sin(0.5*normw*m_dt)/normw*w;
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_particles.pos()(i + active_offset)(d) += m_dt*uadapt(i, d);
            }

            m_particles.q()(i + active_offset) = mult_quaternion(m_particles.q()(i + active_offset), expw);
            normalize(m_particles.q()(i + active_offset));
        }

        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : move active particles = " << duration;
    }
}


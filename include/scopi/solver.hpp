#pragma once

#include <algorithm>
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
             class problem_t = DryWithoutFriction,
             template <class> class optim_solver_t = OptimUzawaMatrixFreeOmp,
             class contact_t = contact_kdtree,
             class vap_t = vap_fixed
             >
    class ScopiSolverBase
    {
    protected:
        ScopiSolverBase(scopi_container<dim>& particles, double dt, OptimParams<optim_solver_t>& optim_params, ProblemParams<problem_t>& problem_params);
    public:
        void solve(std::size_t total_it, std::size_t initial_iter = 0);

    protected:
        void displacement_obstacles();

        std::vector<neighbor<dim>> compute_contacts();
        void write_output_files(std::vector<neighbor<dim>>& contacts, std::size_t nite);
        void move_active_particles();
        scopi_container<dim>& m_particles;
        double m_dt;
        optim_solver_t<problem_t> m_solver;
        problem_t m_problem;
        vap_t m_vap;
    };

    template<std::size_t dim,
             class problem_t = DryWithoutFriction,
             template <class> class optim_solver_t = OptimUzawaMatrixFreeOmp,
             class contact_t = contact_kdtree,
             class vap_t = vap_fixed
             >
    class ScopiSolver: public ScopiSolverBase<dim, problem_t, optim_solver_t, contact_t, vap_t>
    {
    public:
        ScopiSolver(scopi_container<dim>& particles, double dt, OptimParams<optim_solver_t>& optim_params, ProblemParams<problem_t>& problem_params);
    };

    template<std::size_t dim,
             template <class> class optim_solver_t,
             class contact_t,
             class vap_t
             >
    class ScopiSolver<dim, ViscousWithoutFriction<dim>, optim_solver_t, contact_t, vap_t>: public ScopiSolverBase<dim, ViscousWithoutFriction<dim>, optim_solver_t, contact_t, vap_t>
    {
    public:
        ScopiSolver(scopi_container<dim>& particles, double dt, OptimParams<optim_solver_t>& optim_params, ProblemParams<ViscousWithoutFriction<dim>>& problem_params);
        void solve(std::size_t total_it, std::size_t initial_iter = 0);
    };

    template<std::size_t dim,
             class contact_t,
             class vap_t
             >
    class ScopiSolver<dim, ViscousWithFriction<dim>, OptimMosek, contact_t, vap_t>: public ScopiSolverBase<dim, ViscousWithFriction<dim>, OptimMosek, contact_t, vap_t>
    {
    public:
        ScopiSolver(scopi_container<dim>& particles, double dt, OptimParams<OptimMosek>& optim_params, ProblemParams<ViscousWithFriction<dim>>& problem_params);
        void solve(std::size_t total_it, std::size_t initial_iter = 0);
    private:
        vap_projection m_vap_projection;
    };


    template<std::size_t dim, class problem_t, template <class> class optim_solver_t, class contact_t, class vap_t>
    ScopiSolverBase<dim, problem_t, optim_solver_t, contact_t, vap_t>::ScopiSolverBase(scopi_container<dim>& particles, double dt, OptimParams<optim_solver_t>& optim_params, ProblemParams<problem_t>& problem_params)
    : m_particles(particles)
    , m_dt(dt)
    , m_solver(m_particles.nb_active(), m_dt, particles, optim_params)
    , m_problem(m_particles.nb_active(), m_dt, problem_params)
    , m_vap(m_particles.nb_active(), m_particles.nb_inactive(), m_dt)
    {}

    template<std::size_t dim, class problem_t, template <class> class optim_solver_t, class contact_t, class vap_t>
    ScopiSolver<dim, problem_t, optim_solver_t, contact_t, vap_t>::ScopiSolver(scopi_container<dim>& particles, double dt, OptimParams<optim_solver_t>& optim_params, ProblemParams<problem_t>& problem_params)
    : ScopiSolverBase<dim, problem_t, optim_solver_t, contact_t, vap_t>(particles, dt, optim_params, problem_params)
    {}

    template<std::size_t dim, template <class> class optim_solver_t, class contact_t, class vap_t>
    ScopiSolver<dim, ViscousWithoutFriction<dim>, optim_solver_t, contact_t, vap_t>::ScopiSolver(scopi_container<dim>& particles, double dt, OptimParams<optim_solver_t>& optim_params, ProblemParams<ViscousWithoutFriction<dim>>& problem_params)
    : ScopiSolverBase<dim, ViscousWithoutFriction<dim>, optim_solver_t, contact_t, vap_t>(particles, dt, optim_params, problem_params)
    {}

    template<std::size_t dim, class contact_t, class vap_t>
    ScopiSolver<dim, ViscousWithFriction<dim>, OptimMosek, contact_t, vap_t>::ScopiSolver(scopi_container<dim>& particles, double dt, OptimParams<OptimMosek>& optim_params, ProblemParams<ViscousWithFriction<dim>>& problem_params)
    : ScopiSolverBase<dim, ViscousWithFriction<dim>, OptimMosek, contact_t, vap_t>(particles, dt, optim_params, problem_params)
    , m_vap_projection(this->m_particles.nb_active(), this->m_particles.nb_inactive(), this->m_dt)
    {}

    template<std::size_t dim, class problem_t, template <class> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolverBase<dim, problem_t, optim_solver_t, contact_t, vap_t>::solve(std::size_t total_it, std::size_t initial_iter)
    {
        // Time Loop
        for (std::size_t nite = initial_iter; nite < total_it; ++nite)
        {
            PLOG_INFO << "\n\n------------------- Time iteration ----------------> " << nite;

            tic();
            displacement_obstacles();
            auto duration = toc();
            PLOG_INFO << "----> CPUTIME : obstacles = " << duration;

            auto contacts = compute_contacts();

            tic();
            write_output_files(contacts, nite);
            duration = toc();
            PLOG_INFO << "----> CPUTIME : write output files = " << duration;

            tic();
            m_vap.set_a_priori_velocity(m_particles);
            duration = toc();
            PLOG_INFO << "----> CPUTIME : set vap = " << duration;

            m_solver.run(m_particles, contacts, m_problem, nite);

            tic();
            move_active_particles();
            duration = toc();
            PLOG_INFO << "----> CPUTIME : move active particles = " << duration;

            tic();
            m_vap.update_velocity(m_particles, m_solver.get_uadapt(), m_solver.get_wadapt());
            duration = toc();
            PLOG_INFO << "----> CPUTIME : update vap = " << duration;
        }
    }

    template<std::size_t dim, template <class> class optim_solver_t, class contact_t, class vap_t>
    void ScopiSolver<dim, ViscousWithoutFriction<dim>, optim_solver_t, contact_t, vap_t>::solve(std::size_t total_it, std::size_t initial_iter)
    {
        // Time Loop
        for (std::size_t nite = initial_iter; nite < total_it; ++nite)
        {
            PLOG_INFO << "\n\n------------------- Time iteration ----------------> " << nite;

            tic();
            this->displacement_obstacles();
            auto duration = toc();
            PLOG_INFO << "----> CPUTIME : obstacles = " << duration;

            auto contacts = this->compute_contacts();

            tic();
            this->m_problem.set_gamma(contacts);
            duration = toc();
            PLOG_INFO << "----> CPUTIME : set gamma = " << duration;

            tic();
            this->write_output_files(contacts, nite);
            duration = toc();
            PLOG_INFO << "----> CPUTIME : write output files = " << duration;

            tic();
            this->m_vap.set_a_priori_velocity(this->m_particles);
            duration = toc();
            PLOG_INFO << "----> CPUTIME : set vap = " << duration;

            this->m_solver.run(this->m_particles, contacts, this->m_problem, nite);

            tic();
            this->move_active_particles();
            duration = toc();
            PLOG_INFO << "----> CPUTIME : move active particles = " << duration;

            tic();
            this->m_problem.update_gamma(contacts, this->m_solver.get_lagrange_multiplier(contacts, this->m_problem));
            duration = toc();
            PLOG_INFO << "----> CPUTIME : update gamma = " << duration;

            tic();
            this->m_vap.update_velocity(this->m_particles, this->m_solver.get_uadapt(), this->m_solver.get_wadapt());
            duration = toc();
            PLOG_INFO << "----> CPUTIME : update vap = " << duration;
        }
    }


    template<std::size_t dim, class contact_t, class vap_t>
    void ScopiSolver<dim, ViscousWithFriction<dim>, OptimMosek, contact_t, vap_t>::solve(std::size_t total_it, std::size_t initial_iter)
    {
        // Time Loop
        for (std::size_t nite = initial_iter; nite < total_it; ++nite)
        {
            PLOG_INFO << "\n\n------------------- Time iteration ----------------> " << nite;

            tic();
            this->displacement_obstacles();
            auto duration = toc();
            PLOG_INFO << "----> CPUTIME : obstacles = " << duration;

            auto contacts = this->compute_contacts();

            tic();
            this->m_problem.set_gamma(contacts);
            duration = toc();
            PLOG_INFO << "----> CPUTIME : set gamma = " << duration;

            tic();
            this->write_output_files(contacts, nite);
            duration = toc();
            PLOG_INFO << "----> CPUTIME : write output files = " << duration;

            tic();
            this->m_vap.set_a_priori_velocity(this->m_particles);
            duration = toc();
            PLOG_INFO << "----> CPUTIME : set vap = " << duration;

            this->m_problem.setup_first_resolution();
            this->m_solver.run(this->m_particles, contacts, this->m_problem, nite);

            tic();
            this->m_problem.correct_lambda(contacts, this->m_solver.get_lagrange_multiplier(contacts, this->m_problem), this->m_particles, this->m_solver.get_uadapt());
            duration = toc();
            PLOG_INFO << "----> CPUTIME : update gamma = " << duration;

            m_vap_projection.set_u_w(this->m_solver.get_uadapt(), this->m_solver.get_wadapt());
            m_vap_projection.set_a_priori_velocity(this->m_particles);
            this->m_problem.setup_projection();
            this->m_solver.run(this->m_particles, contacts, this->m_problem,  nite);

            tic();
            this->m_problem.update_gamma(contacts, this->m_solver.get_lagrange_multiplier(contacts, this->m_problem));
            duration = toc();
            PLOG_INFO << "----> CPUTIME : update gamma = " << duration;

            tic();
            this->move_active_particles();
            duration = toc();
            PLOG_INFO << "----> CPUTIME : move active particles = " << duration;

            tic();
            this->m_vap.update_velocity(this->m_particles, this->m_solver.get_uadapt(), this->m_solver.get_wadapt());
            duration = toc();
            PLOG_INFO << "----> CPUTIME : update vap = " << duration;
        }
    }

    template<std::size_t dim, class problem_t, template <class> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolverBase<dim, problem_t, optim_solver_t, contact_t, vap_t>::displacement_obstacles()
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
    }

    template<std::size_t dim, class problem_t, template <class> class optim_solver_t,class contact_t, class vap_t>
    std::vector<neighbor<dim>> ScopiSolverBase<dim, problem_t, optim_solver_t, contact_t, vap_t>::compute_contacts()
    {
        // // contact_brute_force cont(2);
        // contact_t cont(2., 10.);
        contact_t cont(2.);
        auto contacts = cont.run(m_particles, m_particles.nb_inactive());
        PLOG_INFO << "contacts.size() = " << contacts.size() << std::endl;
        return contacts;
    }

    template<std::size_t dim, class problem_t, template <class> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolverBase<dim, problem_t, optim_solver_t, contact_t, vap_t>::write_output_files(std::vector<neighbor<dim>>& contacts, std::size_t nite)
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

    template<std::size_t dim, class problem_t, template <class> class optim_solver_t,class contact_t, class vap_t>
    void ScopiSolverBase<dim, problem_t, optim_solver_t, contact_t, vap_t>::move_active_particles()
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
    }
}


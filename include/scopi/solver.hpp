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
#include "problems/DryWithoutFriction.hpp"
#include "contact/contact_kdtree.hpp"
#include "vap/vap_fixed.hpp"
#include "params.hpp"

namespace nl = nlohmann;

using namespace xt::placeholders;

namespace scopi
{
    /**
     * @brief Store the rotation velocities, solution of the optimization problem.
     *
     * In 2D, the rotation velocity is a scalar, whereas in 3D it is a vector.
     * Therefore, the way to update it in the container is different.
     *
     * @tparam dim Dimension (2 or 3).
     * @param particles [out] Container whose field \c omega is updated.
     * @param i [in] Index of the particle to update.
     * @param wadapt [in] \f$N \times 3\f$ array that containes the new velocity, where \f$N\f$ is the total number of particles.
     */
    template<std::size_t dim>
    void update_velocity_omega(scopi_container<dim>& particles, std::size_t i, const xt::xtensor<double, 2>& wadapt);

    /**
     * @brief Entry point of SCoPI.
     *
     * @tparam dim dimension (2 or 3)
     * @tparam optim_solver_t Optimization solver (Mosek, Uzawa, ...)
     * @tparam contact_t Algorithm to search closest contacts (k-d tree, brute force, ...)
     * @tparam vap_t A priori velocity, problem dependant
     *
     * Solve the contact problem: at each time step
     *      - Move obstacles (particles with an imposed velocity);
     *      - Compute the list of contacts;
     *      - Write output files (json format) for visualization;
     *      - Set a priori velocity: describe how the particles would move if they weren't interacting with the other ones;
     *      - Compute the effective velocity as the solution of an optimization problem under constraint \f$D > 0\f$;
     *      - Use these velocities to move the particles;
     *      - Store the computed velocities.
     *
     * The optimization solver \c optim_solver_t describes which algorithm is used to solve the optimization problem.
     * It is itself templated by a \c problem_t that describes which model is used.
     *
     */
    template<std::size_t dim,
             class optim_solver_t = OptimUzawaMatrixFreeOmp<DryWithoutFriction>,
             class contact_t = contact_kdtree,
             class vap_t = vap_fixed
             >
    class ScopiSolver : public optim_solver_t
                      , public vap_t
                      , public contact_t
    {
    private:
        /**
         * @brief Alias for problem type.
         */
        using problem_t = typename optim_solver_t::problem_type;
    public:
        /**
         * @brief Alias for the type of parameters class.
         */
        using params_t = Params<optim_solver_t, contact_t, vap_t>;

        /**
         * @brief Constructor.
         *
         * @param particles "Array" of particles.
         * @param dt Time step. It is fixed during the simulation.
         * @param params Parameters for the different steps of the algorithm.
         */
        ScopiSolver(scopi_container<dim>& particles,
                    double dt,
                    const Params<optim_solver_t, contact_t, vap_t>& params = Params<optim_solver_t, contact_t, vap_t>());

        /**
         * @brief Run the simulation.
         *
         * @param total_it [in] Total number of iterations to perform.
         * @param initial_iter [in] Initial index of iteration. Used for restart or to change external parameters.
         */
        void run(std::size_t total_it, std::size_t initial_iter = 0);

    private:
        /**
         * @brief Move obstacles (particles with an imposed velocity).
         */
        void displacement_obstacles();

        /**
         * @brief Compute the list of contacts.
         *
         * @return Vector containing all the contacts in a cut-off radius.
         */
        std::vector<neighbor<dim>> compute_contacts();

        /**
         * @brief Compute the list of contacts to impose \f$D < 0\f$.
         *
         * If some particles are of type worms, compute a second list of contacts to also impose \f$D < 0\f$ between the spheres that form the worm.
         *
         * @return Vector containing all the contacts involved in a worm.
         */
        std::vector<neighbor<dim>> compute_contacts_worms();

        /**
         * @brief Write output files (json format) for visualization.
         *
         * @param contacts [in] List of contacts (only \f$D > 0\f$).
         * @param nite [in] Current index of iteration in time.
         */
        void write_output_files(const std::vector<neighbor<dim>>& contacts, std::size_t nite);

        /**
         * @brief Use the velocities solution of the optimization problem to move the particles;
         */
        void move_active_particles();

        /**
         * @brief Store the computed velocities.
         */
        void update_velocity();

        /**
         * @brief Parameters specific to the main algorithm.
         */
        ScopiParams m_params;

        /**
         * @brief Array of particles.
         */
        scopi_container<dim>& m_particles;

        /**
         * @brief Time step, fixed during the simulation.
         */
        double m_dt;
    };

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::ScopiSolver(scopi_container<dim>& particles,
                                                                    double dt,
                                                                    const Params<optim_solver_t, contact_t, vap_t>& params)
    : optim_solver_t(particles.nb_active(), dt, particles, params.optim_params, params.problem_params)
    , vap_t(particles.nb_active(), particles.nb_inactive(), particles.size(), dt, params.vap_params)
    , contact_t(params.contacts_params)
    , m_params(params.scopi_params)
    , m_particles(particles)
    , m_dt(dt)
    {}

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::run(std::size_t total_it, std::size_t initial_iter)
    {
        // Time Loop
        for (std::size_t nite = initial_iter; nite < total_it; ++nite)
        {
            PLOG_INFO << "\n\n------------------- Time iteration ----------------> " << nite;

            displacement_obstacles();
            auto contacts = compute_contacts();
            auto contacts_worms = compute_contacts_worms();
            if (nite % m_params.output_frequency == 0 && m_params.output_frequency != std::size_t(-1))
            {
                write_output_files(contacts, nite);
            }
            this->set_a_priori_velocity(m_particles, contacts, contacts_worms);
            this->extra_steps_before_solve(contacts);
            while (this->should_solve_optimization_problem())
            {
                optim_solver_t::run(m_particles, contacts, contacts_worms, nite);
                this->extra_steps_after_solve(contacts, this->get_lagrange_multiplier(contacts, contacts_worms), this->get_constraint(contacts));
            }
            move_active_particles();
            update_velocity();
        }
    }

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
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

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    std::vector<neighbor<dim>> ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::compute_contacts()
    {
        auto contacts = contact_t::run(m_particles, m_particles.nb_inactive());
        PLOG_INFO << "contacts.size() = " << contacts.size() << std::endl;
        return contacts;
    }

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
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

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::write_output_files(const std::vector<neighbor<dim>>& contacts, std::size_t nite)
    {
        tic();
        nl::json json_output;

        std::ofstream file(fmt::format("{}{:04d}.json", m_params.filename, nite));

        json_output["objects"] = {};

        for (std::size_t i = 0; i < m_particles.size(); ++i)
        {
            json_output["objects"].push_back(write_objects_dispatcher<dim>::dispatch(*m_particles[i]));
        }

        if (m_params.write_velocity)
        {
            for (std::size_t i = 0; i < m_particles.size(); ++i)
            {
                json_output["objects"][i]["velocity"] = m_particles.v()(i);
                json_output["objects"][i]["rotationvelocity"] = m_particles.omega()(i);
            }
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

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::move_active_particles()
    {
        tic();
        std::size_t active_offset = m_particles.nb_inactive();
        auto uadapt = this->get_uadapt();
        auto wadapt = this->get_wadapt();

        #pragma omp parallel for
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

            for (auto& p: m_particles.pos())
            {
                if (p[0] > 1.)
                {
                    p[0] -= 1.;
                }
                else if (p[0] < 0.)
                {
                    p[0] += 1.;
                }
            }
        }

        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : move active particles = " << duration;
    }

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::update_velocity()
    {
        tic();
        std::size_t active_offset = m_particles.nb_inactive();
        auto uadapt = this->get_uadapt();
        auto wadapt = this->get_wadapt();

        #pragma omp parallel for
        for (std::size_t i = 0; i < m_particles.nb_active(); ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_particles.v()(i + active_offset)(d) = uadapt(i, d);
            }
            update_velocity_omega(m_particles, i, wadapt);
        }
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : update velocity = " << duration;
    }
}


#pragma once

#include <algorithm>
#include <cstddef>
#include <functional>
#include <iostream>
#include <fstream>
#include <vector>

#include <CLI/CLI.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "container.hpp"
#include "objects/methods/add_contact.hpp"
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
         * @param box used to apply periodic boundary conditions.
         * @param particles "Array" of particles.
         * @param dt Time step. It is fixed during the simulation.
         * @param params Parameters for the different steps of the algorithm.
         */
        ScopiSolver(const BoxDomain<dim>& box,
                    scopi_container<dim>& particles,
                    double dt,
                    const Params<optim_solver_t, contact_t, vap_t>& params = Params<optim_solver_t, contact_t, vap_t>());

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

        void init_options(CLI::App& app);

        /**
         * @brief Run the simulation.
         *
         * @param total_it [in] Total number of iterations to perform.
         * @param initial_iter [in] Initial index of iteration. Used for restart or to change external parameters.
         */
        void run(std::size_t total_it, std::size_t initial_iter = 0);

        /**
         * @brief Return the parameters of the solver.
         */
        params_t& get_params();

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

        BoxDomain<dim> m_box;

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
    ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::ScopiSolver(const BoxDomain<dim>& box,
                                                                    scopi_container<dim>& particles,
                                                                    double dt,
                                                                    const Params<optim_solver_t, contact_t, vap_t>& params)
    : optim_solver_t(particles.nb_active(), dt, particles, params.optim_params, params.problem_params)
    , vap_t(particles.nb_active(), particles.nb_inactive(), particles.size(), dt, params.vap_params)
    , contact_t(params.contacts_params)
    , m_params(params.scopi_params)
    , m_box(box)
    , m_particles(particles)
    , m_dt(dt)
    {}

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::ScopiSolver(scopi_container<dim>& particles,
                                                                    double dt,
                                                                    const Params<optim_solver_t, contact_t, vap_t>& params)
    : optim_solver_t(particles.nb_active(), dt, particles, params.optim_params, params.problem_params)
    , vap_t(particles.nb_active(), particles.nb_inactive(), particles.size(), dt, params.vap_params)
    , contact_t(params.contacts_params)
    , m_params(params.scopi_params)
    , m_box()
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
            if (nite % m_params.output_frequency == 0 && m_params.output_frequency != std::size_t(-1))
            {
                write_output_files(contacts, nite);
            }
            this->set_a_priori_velocity(m_particles, contacts);
            this->extra_steps_before_solve(contacts);
            while (this->should_solve_optimization_problem())
            {
                optim_solver_t::run(m_particles, contacts, nite);
                this->extra_steps_after_solve(contacts, this->get_lagrange_multiplier(contacts), this->get_constraint(contacts));
            }
            move_active_particles();
            update_velocity();
        }
    }

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::init_options(CLI::App& app)
    {
        m_params.init_options(app);
        contact_t::init_options(app);
        optim_solver_t::init_options(app);
    }

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    auto ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::get_params() -> params_t&
    {
        return m_params;
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
        }
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : obstacles = " << duration;
    }

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    std::vector<neighbor<dim>> ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::compute_contacts()
    {
        auto contacts = contact_t::run(m_box, m_particles, m_particles.nb_inactive());
        for (std::size_t i = 0; i < m_particles.size(); ++i)
        {
            add_contact_from_object_dispatcher<dim>::dispatch(*m_particles[i], m_particles.offset(i), contacts);
        }
        PLOG_INFO << "contacts.size() = " << contacts.size() << std::endl;
        return contacts;
    }

    template<std::size_t dim, class optim_solver_t, class contact_t, class vap_t>
    void ScopiSolver<dim, optim_solver_t, contact_t, vap_t>::write_output_files(const std::vector<neighbor<dim>>& contacts, std::size_t nite)
    {
        tic();

        if (!std::filesystem::exists(m_params.path))
        {
            std::filesystem::create_directories(m_params.path);
        }

        nl::json json_output;

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

            contact["i"] = contacts[i].i;
            contact["j"] = contacts[i].j;
            contact["pi"] = contacts[i].pi;
            contact["pj"] = contacts[i].pj;
            contact["nij"] = contacts[i].nij;

            json_output["contacts"].push_back(contact);

        }

        if (m_params.binary_output)
        {
            std::ofstream file(fmt::format("{}_{:04d}.bson", (m_params.path / m_params.filename).string(), nite), std::ios::out | std::ios::binary);
            const std::vector<std::uint8_t> vbson = nl::json::to_bson(json_output);
            file.write(reinterpret_cast<const char*>(vbson.data()), vbson.size() * sizeof(uint8_t));
            file.close();
        }
        else
        {
            std::ofstream file(fmt::format("{}_{:04d}.json", (m_params.path / m_params.filename).string(), nite));
            file << std::setw(4) << json_output;
            file.close();
        }

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
        }

        for (std::size_t io=0; io < m_particles.size(); ++io)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (m_box.is_periodic(d))
                {
                    std::size_t plus = 0;
                    std::size_t minus = 0;
                    for (std::size_t offset = m_particles.offset(io); offset < m_particles.offset(io+1); ++offset)
                    {
                        auto& p = m_particles.pos()[offset];
                        if (p[d] > m_box.upper_bound(d))
                        {
                            plus++;
                        }
                        if (p[d] < m_box.lower_bound(d))
                        {
                            minus++;
                        }
                    }

                    auto object_size = m_particles.offset(io+1) - m_particles.offset(io);
                    if (plus == object_size)
                    {
                        for (std::size_t offset = m_particles.offset(io); offset < m_particles.offset(io+1); ++offset)
                        {
                            auto& p = m_particles.pos()[offset];
                            p[d] -= m_box.upper_bound(d) - m_box.lower_bound(d);
                        }
                    }
                    if (minus == object_size)
                    {
                        for (std::size_t offset = m_particles.offset(io); offset < m_particles.offset(io+1); ++offset)
                        {
                            auto& p = m_particles.pos()[offset];
                            p[d] += m_box.upper_bound(d) - m_box.lower_bound(d);
                        }
                    }
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


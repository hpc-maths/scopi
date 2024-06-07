#pragma once

#include <algorithm>
#include <cstddef>
#include <fstream>
#include <functional>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <CLI/CLI.hpp>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

#include "container.hpp"
#include "objects/methods/add_contact.hpp"
#include "objects/methods/closest_points.hpp"
#include "objects/methods/write_objects.hpp"
#include "objects/neighbor.hpp"
#include "quaternion.hpp"

#include "contact/contact_kdtree.hpp"
#include "contact/property.hpp"
#include "params.hpp"
#include "solvers/OptimGradient.hpp"
#include "solvers/apgd.hpp"
#include "vap/vap_fixed.hpp"

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
     * @param wadapt [in] \f$N \times 3\f$ array that contains the new velocity, where \f$N\f$ is the total number of particles.
     */
    template <std::size_t dim, class xt_container>
    inline std::enable_if_t<dim == 2, void> update_velocity_omega(scopi_container<dim>& particles, std::size_t i, const xt_container& wadapt)
    {
        particles.omega()(i + particles.nb_inactive()) = wadapt(i, 2);
    }

    template <std::size_t dim, class xt_container>
    inline std::enable_if_t<dim == 3, void> update_velocity_omega(scopi_container<dim>& particles, std::size_t i, const xt_container& wadapt)
    {
        for (std::size_t d = 0; d < 3; ++d)
        {
            particles.omega()(i + particles.nb_inactive())(d) = wadapt(i, d);
        }
    }

    // template<std::size_t dim, class xt_container>
    // void update_velocity_omega(scopi_container<dim>& particles, std::size_t i, const xt_container& wadapt);

    /**
     * @brief Entry point of SCoPI.
     *
     * @tparam dim dimension (2 or 3)
     * @tparam optim_solver_t Optimization solver (Mosek, Uzawa, ...)
     * @tparam contact_method_t Algorithm to search closest contacts (k-d tree, brute force, ...)
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
    template <std::size_t dim,
              class problem_type                         = NoFriction,
              class optim_solver_type                    = OptimGradient<apgd>,
              template <class> class contact_method_type = contact_kdtree,
              class vap_type                             = vap_fixed>
    class ScopiSolver
    {
      public:

        using problem_t        = problem_type;
        using optim_solver_t   = optim_solver_type;
        using contact_method_t = contact_method_type<problem_t>;
        using vap_t            = vap_type;
        using params_t         = Params<ScopiSolver<dim, problem_t, optim_solver_t, contact_method_type, vap_t>>;

        using contact_container_t  = std::vector<neighbor<dim, problem_t>>;
        using particle_container_t = scopi_container<dim>;

        /**
         * @brief Constructor.
         *
         * @param box used to apply periodic boundary conditions.
         * @param particles "Array" of particles.
         * @param dt Time step. It is fixed during the simulation.
         * @param params Parameters for the different steps of the algorithm.
         */
        explicit ScopiSolver(const BoxDomain<dim>& box, scopi_container<dim>& particles);

        /**
         * @brief Constructor.
         *
         * @param particles "Array" of particles.
         * @param dt Time step. It is fixed during the simulation.
         * @param params Parameters for the different steps of the algorithm.
         */
        explicit ScopiSolver(scopi_container<dim>& particles);

        void init_options();

        /**
         * @brief Run the simulation.
         *
         * @param total_it [in] Total number of iterations to perform.
         * @param initial_iter [in] Initial index of iteration. Used for restart or to change external parameters.
         */
        void run(double dt, std::size_t total_it, std::size_t initial_iter = 0);

        /**
         * @brief Return the current contacts of the simulation.
         *
         * @return A vector containing all the contacts.
         */
        auto& current_contacts();

        /**
         * @brief Return the parameters of the solver.
         */
        params_t get_params();

      private:

        void set_timestep(double dt);

        /**
         * @brief Move obstacles (particles with an imposed velocity).
         */
        void displacement_obstacles();

        /**
         * @brief Compute the list of contacts.
         *
         * @return Vector containing all the contacts in a cut-off radius.
         */
        contact_container_t compute_contacts();

        /**
         * @brief Write output files (json format) for visualization.
         *
         * @param contacts [in] List of contacts (only \f$D > 0\f$).
         * @param nite [in] Current index of iteration in time.
         */
        void write_output_files(const contact_container_t& contacts, std::size_t nite);

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
        particle_container_t& m_particles;

        /**
         * @brief Time step, fixed during the simulation.
         */
        double m_dt;

        optim_solver_t m_optim_solver;
        contact_method_t m_contact_method;
        vap_t m_vap;
        contact_container_t m_old_contacts;
        std::size_t m_current_save = 0;
    };

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::ScopiSolver(const BoxDomain<dim>& box,
                                                                                      scopi_container<dim>& particles)
        : m_box(box)
        , m_particles(particles)
        , m_optim_solver()
        , m_contact_method()
        , m_vap()
    {
        init_options();
    }

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::ScopiSolver(scopi_container<dim>& particles)
        : m_particles(particles)
        , m_optim_solver()
        , m_contact_method()
        , m_vap()
    {
        init_options();
    }

    struct pairhash
    {
      public:

        template <typename T, typename U>
        std::size_t operator()(const std::pair<T, U>& x) const
        {
            return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
        }
    };

    template <class Contacts>
    void transfer(const Contacts& m_old_contacts, Contacts& contacts)
    {
        std::unordered_map<std::pair<std::size_t, std::size_t>, std::size_t, pairhash> indices;
        std::size_t index = 0;
        for (const auto& c : m_old_contacts)
        {
            indices[{c.i, c.j}] = index++;
        }

        for (auto& c : contacts)
        {
            if (auto search = indices.find({c.i, c.j}); search != indices.end())
            {
                c.property = m_old_contacts[search->second].property;
            }
        }
    }

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    void ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::set_timestep(double dt)
    {
        m_dt = dt;
    }

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    void ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::run(double dt, std::size_t total_it, std::size_t initial_iter)
    {
        // Time Loop
        write_output_files(m_old_contacts, initial_iter);
        set_timestep(dt);
        m_optim_solver.set_timestep(m_dt);

        for (std::size_t nite = initial_iter; nite < total_it; ++nite)
        {
            PLOG_INFO << "\n\n------------------- Time iteration ----------------> " << nite;

            displacement_obstacles();
            auto contacts = compute_contacts();

            if (!m_old_contacts.empty())
            {
                transfer(m_old_contacts, contacts);
            }

            m_vap.set_a_priori_velocity(m_dt, m_particles, contacts);
            m_optim_solver.extra_steps_before_solve(contacts);
            while (m_optim_solver.should_solve())
            {
                m_optim_solver.run(m_particles, contacts, nite);
                m_optim_solver.extra_steps_after_solve(contacts, m_particles);
            }
            m_optim_solver.update_contact_properties(contacts);
            update_velocity();
            move_active_particles();

            if (nite % m_params.output_frequency == 0 && m_params.output_frequency != std::size_t(-1))
            {
                write_output_files(contacts, nite + 1); // m_current_save++);
            }
            std::swap(m_old_contacts, contacts);
        }
    }

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    void ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::init_options()
    {
        m_params.init_options();
        m_contact_method.init_options();
        m_optim_solver.init_options();
    }

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    auto& ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::current_contacts()
    {
        return m_old_contacts;
    }

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    auto ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::get_params() -> params_t
    {
        return params_t(m_params,
                        m_optim_solver.get_params(),
                        m_contact_method.default_contact_property(),
                        m_contact_method.get_params(),
                        m_vap.get_params());
    }

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    void ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::displacement_obstacles()
    {
        tic();
        for (std::size_t i = 0; i < m_particles.nb_inactive(); ++i)
        {
            auto w       = get_omega(m_particles.desired_omega()(i));
            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            type::quaternion_t expw;
            auto expw_adapt                       = xt::adapt(expw);
            expw_adapt(0)                         = std::cos(0.5 * normw * m_dt);
            xt::view(expw_adapt, xt::range(1, _)) = std::sin(0.5 * normw * m_dt) / normw * w;

            for (std::size_t d = 0; d < dim; ++d)
            {
                m_particles.pos()(i)(d) += m_dt * m_particles.vd()(i)(d);
            }
            m_particles.q()(i) = mult_quaternion(m_particles.q()(i), expw);
            normalize(m_particles.q()(i));
        }
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : obstacles = " << duration;
    }

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    auto ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::compute_contacts() -> contact_container_t
    {
        auto contacts = m_contact_method.run(m_box, m_particles, m_particles.nb_inactive());
        for (std::size_t i = 0; i < m_particles.size(); ++i)
        {
            add_contact_from_object_dispatcher<dim>::dispatch(*m_particles[i], m_particles.offset(i), contacts);
        }
        PLOG_INFO << "contacts.size() = " << contacts.size() << std::endl;
        return contacts;
    }

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    void ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::write_output_files(const contact_container_t& contacts,
                                                                                                  std::size_t nite)
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
            json_output["objects"].push_back(write_objects_dispatcher<dim>::dispatch(*m_particles[i], i));
        }

        if (m_params.write_velocity)
        {
            for (std::size_t i = 0; i < m_particles.size(); ++i)
            {
                json_output["objects"][i]["velocity"]         = m_particles.v()(i);
                json_output["objects"][i]["rotationvelocity"] = m_particles.omega()(i);
            }
        }

        json_output["contacts"] = {};

        for (const auto& c : contacts)
        {
            json_output["contacts"].push_back(c.to_json());
        }

        if (m_params.binary_output)
        {
            std::ofstream file(fmt::format("{}_{:04d}.bson", (m_params.path / m_params.filename).string(), nite),
                               std::ios::out | std::ios::binary);
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

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    void ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::move_active_particles()
    {
        tic();
        std::size_t active_offset = m_particles.nb_inactive();

#pragma omp parallel for
        for (std::size_t i = 0; i < m_particles.nb_active(); ++i)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> w;
            if constexpr (dim == 2)
            {
                w = {0, 0, m_particles.omega()(i + active_offset)};
            }
            else
            {
                w = m_particles.omega()(i + active_offset);
            }

            double normw = xt::linalg::norm(w);
            if (normw == 0)
            {
                normw = 1;
            }
            type::quaternion_t expw;
            auto expw_adapt                       = xt::adapt(expw);
            expw_adapt(0)                         = std::cos(0.5 * normw * m_dt);
            xt::view(expw_adapt, xt::range(1, _)) = std::sin(0.5 * normw * m_dt) / normw * w;
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_particles.pos()(i + active_offset)(d) += m_dt * m_particles.v()(i + active_offset)(d);
            }

            m_particles.q()(i + active_offset) = mult_quaternion(m_particles.q()(i + active_offset), expw);
            normalize(m_particles.q()(i + active_offset));
        }

        for (std::size_t io = 0; io < m_particles.size(); ++io)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                if (m_box.is_periodic(d))
                {
                    std::size_t plus  = 0;
                    std::size_t minus = 0;
                    for (std::size_t offset = m_particles.offset(io); offset < m_particles.offset(io + 1); ++offset)
                    {
                        const auto& p = m_particles.pos()[offset];
                        if (p[d] > m_box.upper_bound(d))
                        {
                            plus++;
                        }
                        if (p[d] < m_box.lower_bound(d))
                        {
                            minus++;
                        }
                    }

                    auto object_size = m_particles.offset(io + 1) - m_particles.offset(io);
                    if (plus == object_size)
                    {
                        for (std::size_t offset = m_particles.offset(io); offset < m_particles.offset(io + 1); ++offset)
                        {
                            auto& p = m_particles.pos()[offset];
                            p[d] -= m_box.upper_bound(d) - m_box.lower_bound(d);
                        }
                    }
                    if (minus == object_size)
                    {
                        for (std::size_t offset = m_particles.offset(io); offset < m_particles.offset(io + 1); ++offset)
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

    template <std::size_t dim, class problem_t, class optim_solver_t, template <class> class contact_method_t, class vap_t>
    void ScopiSolver<dim, problem_t, optim_solver_t, contact_method_t, vap_t>::update_velocity()
    {
        tic();
        std::size_t active_offset = m_particles.nb_inactive();
        auto uadapt               = m_optim_solver.get_uadapt();
        auto wadapt               = m_optim_solver.get_wadapt();

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

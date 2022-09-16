#pragma once

#include <cstddef>
#include <plog/Log.h>
#include <vector>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../container.hpp"
#include "../crtp.hpp"
#include "../objects/neighbor.hpp"
#include "../utils.hpp"
#include "../params.hpp"

namespace scopi{
    /**
     * @brief Commun interface for the different optimization solvers.
     *
     * @tparam Derived Optimization solver.
     * @tparam problem_t Problem to be solved.
     */
    template <class Derived, class problem_t>
    class OptimBase : protected problem_t
    {

    protected:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         * @param dt [in] Time step.
         * @param cSize [in] Size of the vector \f$ \c \f$ (depends on the problem).
         * @param c_dec [in] For some solvers (mostly OptimMosek), the vector \f$ \c \f$ contains more elements than just the a priori velocities. \c c_dec is the index of the first a priori velocity.
         * @param optim_params [in] Parameters for the optimization solver.
         * @param problem_params [in] Parameters for the problem.
         */
        OptimBase(std::size_t nparts, double dt, std::size_t cSize, std::size_t c_dec, const OptimParams<Derived>& optim_params, const ProblemParams<problem_t>& problem_params);

        /**
         * @brief Build the vectors and matrices necessary to solve the optimization problem and solve it.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         * @param nite [in] Current time step.
         */
        template<std::size_t dim>
        void run(const scopi_container<dim>& particles,
                 const std::vector<neighbor<dim>>& contacts,
                 const std::vector<neighbor<dim>>& contacts_worms,
                 const std::size_t nite);

        /**
         * @brief \f$ \u \in \mathbb{R}^{6N} \f$ contains the velocities and the rotations of the particles, the function returns the velocities solution of the optimization problem..
         *
         * \pre Call \c run before calling this function.
         *
         * @return \f$ N \times 3 \f$ array.
         */
        auto get_uadapt();
        /**
         * @brief \f$ \u \in \mathbb{R}^{6N} \f$ contains the velocities and the rotations of the particles, the function returns the rotations solution of the optimization problem..
         *
         * \pre Call \c run before calling this function.
         *
         * @return \f$ N \times 3 \f$ array.
         */
        auto get_wadapt();
        /**
         * @brief Returns \f$ \d + \B \u \f$, where \f$ \u \f$ is the solution of the optimization problem.
         *
         * \pre Call \c run before calling this function.
         * \todo Fix dimensions depending on the problem.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contatcs.
         *
         * @return 
         */
        template<std::size_t dim>
        xt::xtensor<double, 2> get_constraint(const std::vector<neighbor<dim>>& contacts);
        /**
         * @brief Returns the Lagrange multipliers (solution of the dual problem) when the optimization is solved.
         *
         * \pre Call \c run before calling this function.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         *
         * @return \f$ N_c \f$ array.
         */
        template<std::size_t dim>
        auto get_lagrange_multiplier(const std::vector<neighbor<dim>>& contacts, const std::vector<neighbor<dim>>& contacts_worms);

    protected:
        /**
         * @brief Parameters for the optimization solver.
         */
        OptimParams<Derived> m_params;
        /**
         * @brief Number of particles.
         */
        std::size_t m_nparts;
        /**
         * @brief Vector \f$ \c \f$.
         */
        xt::xtensor<double, 1> m_c;

    private:
        /**
         * @brief Build the vector \f$ \c \f$.
         *
         * \f$ \c = \P \mathbf{v}^d \f$, where \f$ \mathbf{v}^d \f$ is the a priori velocity (see ProblemBase for the notations).
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for a priori velocities, masses, and moments of inertia).
         */
        template<std::size_t dim>
        void create_vector_c(const scopi_container<dim>& particles);
        /**
         * @brief Solve the optimization problem.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for a priori velocities, masses, and moments of inertia).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         *
         * @return Number of iterations needed by the solver to converge.
         */
        template<std::size_t dim>
        int solve_optimization_problem(const scopi_container<dim>& particles,
                                       const std::vector<neighbor<dim>>& contacts,
                                       const std::vector<neighbor<dim>>& contacts_worms);
        /**
         * @brief Number of Lagrange multipliers > 0 (active constraints).
         */
        int get_nb_active_contacts() const;

        /**
         * @brief For some solvers (mostly OptimMosek), the vector \f$ \c \f$ contains more elements than just the a priori velocities. \c c_dec is the index of the first a priori velocity.
         */
        std::size_t m_c_dec;
    };

    template<class Derived, class problem_t>
    template<std::size_t dim>
    void OptimBase<Derived, problem_t>::run(const scopi_container<dim>& particles,
                                            const std::vector<neighbor<dim>>& contacts,
                                            const std::vector<neighbor<dim>>& contacts_worms,
                                            const std::size_t)
    {
        tic();
        create_vector_c(particles);
        this->create_vector_distances(contacts, contacts_worms);
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : vectors = " << duration;

        auto nbIter = solve_optimization_problem(particles, contacts, contacts_worms);
        PLOG_INFO << "iterations : " << nbIter;
        PLOG_INFO << "Contacts: " << contacts.size() << "  active contacts " << get_nb_active_contacts();
    }


    template<class Derived, class problem_t>
    OptimBase<Derived, problem_t>::OptimBase(std::size_t nparts, double dt, std::size_t cSize, std::size_t c_dec, const OptimParams<Derived>& optim_params, const ProblemParams<problem_t>& problem_params)
    : problem_t(nparts, dt, problem_params)
    , m_params(optim_params)
    , m_nparts(nparts)
    , m_c(xt::zeros<double>({cSize}))
    , m_c_dec(c_dec)
    {}

    template<class Derived, class problem_t>
    template<std::size_t dim>
    void OptimBase<Derived, problem_t>::create_vector_c(const scopi_container<dim>& particles)
    {
        std::size_t mass_dec = m_c_dec;
        std::size_t moment_dec = mass_dec + 3*particles.nb_active();

        auto active_offset = particles.nb_inactive();

        auto desired_velocity = particles.vd();
        auto desired_omega = particles.desired_omega();

        for (std::size_t i = 0; i < particles.nb_active(); ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_c(mass_dec + 3*i + d) = -particles.m()(active_offset + i)*desired_velocity(i + active_offset)[d];
            }
            auto omega = get_omega(desired_omega(i + active_offset));
            auto j = get_omega(particles.j()(active_offset+i));
            for (std::size_t d = 0; d < 3; ++d)
            {
                m_c(moment_dec + 3*i + d) = -j(d)*omega(d);
            }
        }
    }

    template<class Derived, class problem_t>
    template<std::size_t dim>
    int OptimBase<Derived, problem_t>::solve_optimization_problem(const scopi_container<dim>& particles,
                                                                  const std::vector<neighbor<dim>>& contacts,
                                                                  const std::vector<neighbor<dim>>& contacts_worms)
    {
        return static_cast<Derived&>(*this).solve_optimization_problem_impl(particles, contacts, contacts_worms);
    }

    template<class Derived, class problem_t>
    auto OptimBase<Derived, problem_t>::get_uadapt()
    {
        auto data = static_cast<Derived&>(*this).uadapt_data();
        return xt::adapt(reinterpret_cast<double*>(data), {this->m_nparts, 3UL});
    }

    template<class Derived, class problem_t>
    auto OptimBase<Derived, problem_t>::get_wadapt()
    {
        auto data = static_cast<Derived&>(*this).wadapt_data();
        return xt::adapt(reinterpret_cast<double*>(data), {this->m_nparts, 3UL});
    }

    template<class Derived, class problem_t>
    template<std::size_t dim>
    xt::xtensor<double, 2> OptimBase<Derived, problem_t>::get_constraint(const std::vector<neighbor<dim>>& contacts)
    {
        auto data = static_cast<Derived&>(*this).constraint_data();
        if (data)
            return xt::adapt(reinterpret_cast<double*>(data), {contacts.size(), 4UL});
        else
            return xt::xtensor<double, 2>{};
    }

    template<class Derived, class problem_t>
    template<std::size_t dim>
    auto OptimBase<Derived, problem_t>::get_lagrange_multiplier(const std::vector<neighbor<dim>>& contacts,
                                                                const std::vector<neighbor<dim>>& contacts_worms)
    {
        auto data = static_cast<Derived&>(*this).lagrange_multiplier_data();
        return xt::adapt(reinterpret_cast<double*>(data), {this->number_row_matrix(contacts, contacts_worms), 1UL});
    }

    template<class Derived, class problem_t>
    int OptimBase<Derived, problem_t>::get_nb_active_contacts() const
    {
        return static_cast<const Derived&>(*this).get_nb_active_contacts_impl();
    }
}


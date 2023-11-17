#pragma once

#include "plog/Initializers/RollingFileInitializer.h"
#include <cmath>
#include <plog/Log.h>
#include <xtensor/xfixed.hpp>
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../objects/neighbor.hpp"
#include "../params.hpp"
#include "../quaternion.hpp"
#include "../types.hpp"
#include "../utils.hpp"
#include "DryWithFrictionBase.hpp"

namespace scopi
{
    class DryWithFriction;

    /**
     * @class ProblemParams<DryWithFriction>
     * @brief Parameters for DryWithFriction
     *
     * Specialization of ProblemParams in params.hpp
     */
    template <>
    struct ProblemParams<DryWithFriction>
    {
        /**
         * @brief Default constructor.
         */
        ProblemParams();
        /**
         * @brief Copy constructor.
         *
         * @param params Parameters to by copied.
         */
        ProblemParams(const ProblemParams<DryWithFriction>& params);

        /**
         * @brief Friction coefficient.
         *
         * Default value is 0.
         * \note \c mu > 0
         */
        double mu;
    };

    /**
     * @class DryWithFriction
     * @brief Problem that models contacts with friction and without viscosity.
     *
     * See ProblemBase for the notations.
     * The constraint is
     * \f[
     *      \mathbf{d}_{ij} + \mathbb{B} \mathbf{u}_{ij} \ge ||\mathbb{T} \mathbf{u}_{ij}||
     * \f]
     * for all contacts \f$ ij \f$.
     * \f$ \mathbf{d} \in \mathbb{R}^{N_c} \f$, \f$ \mathbf{u} \in \mathbb{R}^{6N} \f$, \f$ \mathbb{B} \in \mathbb{R}^{N_c \times 6 N} \f$,
     * and \f$ \mathbb{T} \in R^{3 N_c \times 6N} \f$.
     *
     * Only one matrix is built.
     * It contains both matrices \f$ \mathbb{B} \f$ and \f$ \mathbb{T} \f$.
     * A contact \f$ ij \f$ corresponds to four rows in the matrix, one for \f$ \mathbb{B} \f$ and three for \f$ \mathbb{T} \f$.
     * Therefore, the matrix is in \f$ \mathbb{R}^{4N_c \times 6N} \f$ and \f$ \mathbf{d} \in \mathbb{R}^{4N_c} \f$.
     *
     */
    class DryWithFriction : public DryWithFrictionBase<ProblemParams<DryWithFriction>>
    {
      public:

        using base                = DryWithFrictionBase<ProblemParams<DryWithFriction>>;
        using contact_container_t = typename base::contact_container_t;

        /**
         * @brief Constructor.
         *
         * @param nparticles [in] Number of particles.
         * @param dt [in] Time step.
         * @param problem_params [in] Parameters.
         */
        DryWithFriction(std::size_t nparticles, double dt);

        /**
         * @brief Extra steps before solving the optimization problem.
         *
         * For compatibility with the other problems.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim, class optim_solver_t>
        void extra_steps_before_solve(const contact_container_t& contacts, optim_solver_t&);
        /**
         * @brief Extra steps after solving the optimization problem.
         *
         * For compatibility with the other problems.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param lambda [in] Lagrange multipliers.
         * @param u_tilde [in] Vector \f$ \mathbf{d} + \mathbb{B} \mathbf{u} - \mathbf{f}(\mathbf{u}) \f$, where \f$ \mathbf{u} \f$ is the
         * solution of the optimization problem.
         */
        template <std::size_t dim, class optim_solver_t>
        void extra_steps_after_solve(const contact_container_t& contacts, optim_solver_t& optim_solver);
        /**
         * @brief Get the number of rows in the matrix.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         *
         * @return Number of rows in the matrix.
         */
        template <std::size_t dim>
        std::size_t number_row_matrix(const contact_container_t& contacts) const;
        /**
         * @brief Whether the optimization problem should be solved.
         *
         * For compatibility with the other problems.
         */

        /**
         * @brief Create vector \f$ \mathbf{d} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim>
        void create_vector_distances(const contact_container_t& contacts);

        bool should_solve() const;
    };

    template <std::size_t dim, class optim_solver_t>
    void DryWithFriction::extra_steps_before_solve(const contact_container_t&, optim_solver_t&)
    {
        this->m_should_solve = true;
    }

    template <std::size_t dim, class optim_solver_t>
    void DryWithFriction::extra_steps_after_solve(const contact_container_t&, optim_solver_t&)
    {
        this->m_should_solve = false;
    }

    template <std::size_t dim>
    std::size_t DryWithFriction::number_row_matrix(const contact_container_t& contacts) const
    {
        return 4 * contacts.size();
    }

    template <std::size_t dim>
    void DryWithFriction::create_vector_distances(const contact_container_t& contacts)
    {
        this->m_distances = xt::zeros<double>({4 * contacts.size()});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            this->m_distances[4 * i] = contacts[i].dij;
        }
    }
}

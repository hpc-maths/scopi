#pragma once

#include <cmath>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include "../types.hpp"
#include "../container.hpp"
#include "../quaternion.hpp"
#include "../objects/neighbor.hpp"
#include "../utils.hpp"
#include "../params.hpp"
#include "DryWithFrictionBase.hpp"

namespace scopi
{
    class DryWithFriction;

    /**
     * @brief Parameters for \c DryWithFriction
     *
     * Specialization of ProblemParams in params.hpp
     */
    template<>
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
     * @brief Problem that models contacts with friction and without viscosity.
     *
     * See ProblemBase.hpp for the notations.
     * The constraint is 
     * \f[
     *      \mathbf{d}_{ij} + \mathbb{B} \u_{ij} \ge ||\mathbb{T} \u_{ij}||
     * \f]
     * for all contacts \f$ (ij) \f$.
     * \f$ \mathbf{d} \in \mathbb{R}^{N_c} \f$, \f$ \u \in \mathbb{R}^{6N} \f$, \f$ \mathbb{B} \in \mathbb{R}^{N_c \times 6 N} \f$, and \f$ \mathbb{T} \in R^{3 N_c \times 6N} \f$.
     *
     * Only one matrix is built.
     * It contains both matrices $\f$ \mathbb{B} \f$ and \f$ T \f$.
     * A contact \f$ (ij) \f$ corresponds to four rows in the matrix, one for \f$ \mathbb{B} \f$ and three for \f$ T \f$.
     * Therefore, the matrix is in \f$ \mathbb{R}^{4N_c \times 6N} \f$ and \f$ \mathbf{d} \in \mathbb{R}^{4N_c} \f$.
     *
     */
    class DryWithFriction : protected DryWithFrictionBase
    {
    protected:
        /**
         * @brief Constructor.
         *
         * @param nparticles [in] Number of particles.
         * @param dt [in] Time step.
         * @param problem_params [in] Parameters.
         */
        DryWithFriction(std::size_t nparticles, double dt, const ProblemParams<DryWithFriction>& problem_params);

        /**
         * @brief Construct the COO storage of the matrices \f$ \mathbb{B} \f$ and \f$ \mathbb{T} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for positions).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance (for compatibility with other problems).
         * @param firstCol [in] Index of the first column (solver-dependent).
         */
        template <std::size_t dim>
        void create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts,
                                          const std::vector<neighbor<dim>>& contacts_worms,
                                          std::size_t firstCol);
        /**
         * @brief Get the number of rows in the matrix.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance (for compatibility with other models).
         *
         * @return Number of rows in the matrix.
         */
        template <std::size_t dim>
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts,
                                      const std::vector<neighbor<dim>>& contacts_worms);
        /**
         * @brief Create vector \f$ \mathbf{d} \f$.
         *
         * See \c create_vector_distances for the order of the rows of the matrix.
         *
         * \f$ \mathbf{d} \in \mathbb{R}^{4N_c} \f$ can be seen as a block vector, each block has the form
         * \f$ (d_{ij}, 0, 0, 0) \f$,
         * where \f$ d_{ij} \f$ is the distance between particles \c i and \c j.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance (for compatibility with other models).
         */
        template<std::size_t dim>
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts, const std::vector<neighbor<dim>>& contacts_worms);

        /**
         * @brief Extra steps before solving the optimization problem.
         *
         * For compatibility with the other problems.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         */
        template<std::size_t dim>
        void extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts);
        /**
         * @brief Extra steps after solving the optimization problem.
         *
         * For compatibility with the other problems.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param lambda [in] Lagrange multipliers.
         * @param u_tilde [in] Vector \f$ \mathbf{d} + \mathbb{B} \u - \constraintFunction(\u) \f$, where \f$ \u \f$ is the solution of the optimization problem.
         */
        template<std::size_t dim>
        void extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                     const xt::xtensor<double, 1>& lambda,
                                     const xt::xtensor<double, 2>& u_tilde);
        /**
         * @brief Whether the optimization problem should be solved.
         *
         * For compatibility with the other problems.
         */
        bool should_solve_optimization_problem();

    private:
        /**
         * @brief Parameters, see <tt> ProblemParams<DryWithFriction> </tt>.
         */
        ProblemParams<DryWithFriction> m_params;
    };

    template<std::size_t dim>
    void DryWithFriction::create_vector_distances(const std::vector<neighbor<dim>>& contacts, const std::vector<neighbor<dim>>&)
    {
        this->m_distances = xt::zeros<double>({4*contacts.size()});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            this->m_distances[4*i] = contacts[i].dij;
        }
    }

    template<std::size_t dim>
    void DryWithFriction::extra_steps_before_solve(const std::vector<neighbor<dim>>&)
    {
        this->m_should_solve = true;
    }

    template<std::size_t dim>
    void DryWithFriction::extra_steps_after_solve(const std::vector<neighbor<dim>>&,
                                                  const xt::xtensor<double, 1>&,
                                                  const xt::xtensor<double, 2>&)
    {
        this->m_should_solve = false;
    }
  
}


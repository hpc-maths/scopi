#pragma once

#include <cmath>
#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <vector>
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
    class DryWithFrictionFixedPoint;

    /**
     * @brief Parameters for \c DryWithoutFriction.
     *
     * Specialization of ProblemParams in params.hpp.
     */
    template<>
    struct ProblemParams<DryWithFrictionFixedPoint>
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
        ProblemParams(const ProblemParams<DryWithFrictionFixedPoint>& params);

        /**
         * @brief Friction coefficient.
         *
         * Default value is 0.
         * \note \c mu > 0
         */
        double mu;
        /**
         * @brief Tolerance for the fixed point algorithm.
         *
         * Default value is \f$ 10^{-2} \f$.
         * \note \c tol_fixed_point > 0
         */
        double tol_fixed_point;
        /**
         * @brief Maximum number of iterations for the fixed point algorithm.
         *
         * Default value is 20.
         * \note \c max_iter_fixed_point > 0
         */
        std::size_t max_iter_fixed_point;
    };

    /**
     * @brief Problem that models contacts with friction and without viscosity. A fixed point algorithm is used to ensure \f$ D = 0 \f$.
     *
     * See ProblemBase.hpp for the notations.
     * The constraint is 
     * \f[ 
     *      \mathbf{d}_{ij} + \mathbb{B} \mathbf{u}_{ij} \ge \left( ||\mathbb{T} \mathbf{u}_{ij}|| - \mu \Delta t \mathbf{s}_{ij} \right),
     * \f]
     * for all contacts \f$ (ij) \f$, with \f$ \mathbf{s} \in \mathbb{R}^{N_c} \f$.
     * If \f$ \mathbf{u}^{\mathbf{s}} \f$ is the solution of the parametrized problem, then we consider
     * \f[
     *      \begin{aligned}
     *          \mathbf{F} : & \mathbb{R}^{N_c} \to \mathbb{R}^{N_c} \\
     *               & \mathbf{s}_{ij} \mapsto ||\mathbb{T} \mathbf{u}^{\mathbf{s}}_{ij}||,
     *      \end{aligned}
     * \f]
     * and search for a fixed point of \f$ \mathbf{F} \f$ : \f$ \mathbf{s} \in \mathbb{R}^{N_c} \f$ such that \f$ \mathbf{F}(\mathbf{s}) = \mathbf{s} \f$.
     *
     * This leads to the following algorithm:
     * - \f$ \sWithIndex{0} \f$;
     * - \f$ \indexFixedPoint = 0 \f$;
     * - While \f$ \frac{||\sWithIndex{\indexFixedPoint-1} - \sWithIndex{\indexFixedPoint}||}{||\sWithIndex{\indexFixedPoint}|| + 1} > \f$ \c tol_fixed_point and \f$ \indexFixedPoint < \f$ \c max_iter_fixed_point
     *   - Compute \f$ \usWithIndex{\indexFixedPoint} \f$ as the solution of the optimization problem under the constraint written above;
     *   - \f$ \sWithIndex{\indexFixedPoint+1}_{ij} = ||\mathbb{T} \usWithIndex{\indexFixedPoint}_{ij}|| \f$ for all contacts \f$ ij \f$;
     *   - \f$ \indexFixedPoint ++ \f$.
     * 
     * Only one matrix is built.
     * It contains both matrices $\f$ \mathbb{B} \f$ and \f$ \mathbb{T} \f$.
     * A contact \f$ (ij) \f$ corresponds to four rows in the matrix, one for \f$ \mathbb{B} \f$ and three for \f$ \mathbb{T} \f$.
     * Therefore, the matrix is in \f$ \mathbb{R}^{4N_c \times 6N} \f$ and \f$ \mathbf{d} \in \mathbb{R}^{4N_c} \f$.
     */
    class DryWithFrictionFixedPoint : protected DryWithFrictionBase
    {
    protected:
        /**
         * @brief Constructor.
         *
         * @param nparticles [in] Number of particles.
         * @param dt [in] Time step.
         * @param problem_params [in] Parameters.
         */
        DryWithFrictionFixedPoint(std::size_t nparticles, double dt, const ProblemParams<DryWithFrictionFixedPoint>& problem_params);

        /**
         * @brief Construct the COO storage of the matrices \f$ \mathbb{B} \f$ and \f$ \mathbb{T} \f$.
         *
         * \todo It should be the same function as in DryWithFriction.hpp
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
         * See \c create_vector_distances for the order of the rows of the matrix.
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
         * \f$ \mathbf{d} \in \mathbb{R}^{4N_c} \f$ can be seen as a block vector, each block has the form
         * \f$ (d_{ij} + \mu \Delta t \mathbf{s}_{ij}, 0, 0, 0) \f$,
         * where \f$ d_{ij} \f$ is the distance between particles \c i and \c j.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance (for compatibility with other models).
         */
        template<std::size_t dim>
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts, const std::vector<neighbor<dim>>& contacts_worms);

        /**
         * @brief Initialize variables for fixed-point algorithm.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         */
        template<std::size_t dim>
        void extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts);
        /**
         * @brief Compute \f$ \sWithIndex{\indexFixedPoint+1} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param lambda [in] Lagrange multipliers.
         * @param u_tilde [in] Vector \f$ \mathbf{d} + \mathbb{B} \mathbf{u} - \mathbf{f}(\mathbf{u}) \f$, where \f$ \mathbf{u} \f$ is the solution of the optimization problem.
         */
        template<std::size_t dim>
        void extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                     const xt::xtensor<double, 1>& lambda,
                                     const xt::xtensor<double, 2>& u_tilde);
        /**
         * @brief Stop criterion for the fixed point algorithm.
         *
         * @return Whether the fixed point algorithm has converged. 
         */
        bool should_solve_optimization_problem();

    private:
        /**
         * @brief Parameters.
         */
        ProblemParams<DryWithFrictionFixedPoint> m_params;
        /**
         * @brief \f$ \sWithIndex{\indexFixedPoint+1} \f$.
         */
        xt::xtensor<double, 1> m_s;
        /**
         * @brief \f$ \sWithIndex{\indexFixedPoint} \f$.
         */
        xt::xtensor<double, 1> m_s_old;
        /**
         * @brief Number of iterations in the fixed point algorithm (\f$ \indexFixedPoint \f$).
         */
        std::size_t m_nb_iter;
    };

    template<std::size_t dim>
    void DryWithFrictionFixedPoint::create_vector_distances(const std::vector<neighbor<dim>>& contacts, const std::vector<neighbor<dim>>&)
    {
        this->m_distances = xt::zeros<double>({4*contacts.size()});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            this->m_distances[4*i] = contacts[i].dij + m_params.mu*this->m_dt*m_s(i);
        }
    }

    template<std::size_t dim>
    void DryWithFrictionFixedPoint::extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts)
    {
        m_nb_iter = 0;
        m_s = xt::zeros<double>({contacts.size()});
        m_s_old = xt::ones<double>({contacts.size()});
    }

    template<std::size_t dim>
    void DryWithFrictionFixedPoint::extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                                            const xt::xtensor<double, 1>&,
                                                            const xt::xtensor<double, 2>& u_tilde)
    {
        m_nb_iter++;
        m_s_old = m_s;
        // TODO use xtensor functions to avoid loop
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            m_s(i) = xt::linalg::norm(xt::view(u_tilde, i, xt::range(1, _)))/(this->m_dt*m_params.mu);
        }
    }
  
}


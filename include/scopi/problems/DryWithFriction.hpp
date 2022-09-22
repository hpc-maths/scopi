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
     *      \d_{\ij} + \B \u_{\ij} \ge \norm{\T \u_{\ij}}
     * \f]
     * for all contacts \f$ (\ij) \f$.
     * \f$ \d \in \R^{\Nc} \f$, \f$ \u \in \R^{6\N} \f$, \f$ \B \in \R^{\Nc \times 6 N} \f$, and \f$ \T \in R^{3 \Nc \times 6\N} \f$.
     *
     * Only one matrix is built.
     * It contains both matrices $\f$ \B \f$ and \f$ T \f$.
     * A contact \f$ (\ij) \f$ corresponds to four rows in the matrix, one for \f$ \B \f$ and three for \f$ T \f$.
     * Therefore, the matrix is in \f$ \R^{4\Nc \times 6N} \f$ and \f$ \d \in \R^{4\Nc} \f$.
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
         * @brief Create vector \f$ \d \f$.
         *
         * See \c create_vector_distances for the order of the rows of the matrix.
         *
         * \f$ \d \in \R^{4\Nc} \f$ can be seen as a block vector, each block has the form
         * \f$ (d_{\ij}, 0, 0, 0) \f$,
         * where \f$ d_{\ij} \f$ is the distance between particles \c i and \c j.
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
         * @param u_tilde [in] Vector \f$ \d + \B \u - \constraintFunction(\u) \f$, where \f$ \u \f$ is the solution of the optimization problem.
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


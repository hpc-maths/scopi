#pragma once

#include "OptimUzawaBase.hpp"
#include <cstddef>
#include <omp.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include "plog/Initializers/RollingFileInitializer.h"
#include <plog/Log.h>

#include "../quaternion.hpp"
#include "../utils.hpp"

namespace scopi
{
    template <std::size_t dim, class problem_t>
    class OptimUzawaMatrixFreeOmp;

    /**
     * @brief Parameters for OptimUzawaMatrixFreeOmp<dim, problem_t>
     *
     * Specialization of ProblemParams.
     * See OptimParamsUzawaBase.
     *
     * @tparam problem_t Problem to be solved.
     */
    template <std::size_t dim, class problem_t>
    struct OptimParams<OptimUzawaMatrixFreeOmp<dim, problem_t>> : public OptimParamsUzawaBase
    {
    };

    /**
     * @brief Uzawa algorithm with matrix-free matrix-vector products parallelized with OpenMP.
     *
     * See OptimUzawaBase for the algorithm.
     * \warning Only the cases \c problem_t = DryWithoutFriction and \c problem_t = ViscousWithoutFriction are implemented.
     *
     * @tparam problem_t Problem to be solved.
     */
    template <std::size_t dim, class problem_t = DryWithoutFriction<dim>>
    class OptimUzawaMatrixFreeOmp : public OptimUzawaBase<OptimUzawaMatrixFreeOmp<dim, problem_t>, dim, problem_t>
    {
      public:

        using contact_container_t = typename problem_t::contact_container_t;
        /**
         * @brief Alias for the base class OptimUzawaBase.
         */
        using base_type = OptimUzawaBase<OptimUzawaMatrixFreeOmp<dim, problem_t>, dim, problem_t>;

        /**
         * @brief Constructor.
         *
         * @tparam dim Dimension (2 or 3).
         * @param nparts [in] Number of particles.
         * @param dt [in] Time step.
         * @param particles [in] Array of particles.
         * @param optim_params [in] Parameters.
         * @param problem_params [in] Parameters for the problem.
         */
        OptimUzawaMatrixFreeOmp(std::size_t nparts, double dt, const scopi_container<dim>& particles);

      public:

        /**
         * @brief Implements the product \f$ \mathbb{P}^{-1} \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for masses and moments of inertia).
         */
        void gemv_inv_P_impl(const scopi_container<dim>& particles);

        /**
         * @brief Implements the product \f$ \mathbf{r} = \mathbf{r} - \mathbb{B} \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        void gemv_A_impl(const scopi_container<dim>& particles, const contact_container_t& contacts);

        /**
         * @brief Implements the product \f$ \mathbf{u} = \mathbb{B}^T \mathbf{l} + \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        void gemv_transpose_A_impl(const scopi_container<dim>& particles, const contact_container_t& contacts);

        /**
         * @brief For compatibility with other methods to compute matrix-vector products.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for positions).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         */
        void init_uzawa_impl(const scopi_container<dim>& particles, const contact_container_t& contacts);
        /**
         * @brief For compatibility with other methods to compute matrix-vector products.
         */
        void finalize_uzawa_impl();
    };

    template <std::size_t dim, class problem_t>
    void OptimUzawaMatrixFreeOmp<dim, problem_t>::init_uzawa_impl(const scopi_container<dim>&, const contact_container_t&)
    {
    }

    template <std::size_t dim, class problem_t>
    void OptimUzawaMatrixFreeOmp<dim, problem_t>::finalize_uzawa_impl()
    {
    }

    template <std::size_t dim, class problem_t>
    void OptimUzawaMatrixFreeOmp<dim, problem_t>::gemv_inv_P_impl(const scopi_container<dim>& particles)
    {
        auto active_offset = particles.nb_inactive();
#pragma omp parallel for
        for (std::size_t i = 0; i < particles.nb_active(); ++i)
        {
            this->problem().matrix_free_gemv_inv_P(particles, this->m_U, active_offset, i);
        }
    }

    template <std::size_t dim, class problem_t>
    void OptimUzawaMatrixFreeOmp<dim, problem_t>::gemv_A_impl(const scopi_container<dim>& particles, const contact_container_t& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();
#pragma omp parallel for
        for (std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            auto& c = contacts[ic];
            this->problem().matrix_free_gemv_A(c, particles, this->m_U, this->m_R, active_offset, ic);
        }
    }

    template <std::size_t dim, class problem_t>
    void OptimUzawaMatrixFreeOmp<dim, problem_t>::gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                                                        const contact_container_t& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();
#pragma omp parallel for
        for (std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            auto& c = contacts[ic];
            this->problem().matrix_free_gemv_transpose_A(c, particles, this->m_L, this->m_U, active_offset, ic);
        }
    }

    template <std::size_t dim, class problem_t>
    OptimUzawaMatrixFreeOmp<dim, problem_t>::OptimUzawaMatrixFreeOmp(std::size_t nparts, double dt, const scopi_container<dim>&)
        : base_type(nparts, dt)
    {
    }

}

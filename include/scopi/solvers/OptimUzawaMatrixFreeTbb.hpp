#pragma once

#ifdef SCOPI_USE_TBB
#include "OptimUzawaBase.hpp"
#include <omp.h>
#include <tbb/tbb.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../quaternion.hpp"
#include "../utils.hpp"

namespace scopi
{
    template<class problem_t>
    class OptimUzawaMatrixFreeTbb;

    /**
     * @brief Parameters for OptimUzawaMatrixFreeTbb<problem_t>
     *
     * Specialization of ProblemParams.
     * See OptimParamsUzawaBase.
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t>
    struct OptimParams<OptimUzawaMatrixFreeTbb<problem_t>> : public OptimParamsUzawaBase
    {
    };

    /**
     * @brief Uzawa algorithm with matrix-free matrix-vector products parallelized with TBB.
     *
     * See OptimUzawaBase for the algorithm.
     * \warning Only the cases \c problem_t = DryWithoutFriction and \c problem_t = ViscousWithoutFriction are implemented.
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t = DryWithoutFriction>
    class OptimUzawaMatrixFreeTbb : public OptimUzawaBase<OptimUzawaMatrixFreeTbb<problem_t>, problem_t>
    {
    public:
        /**
         * @brief Alias for the problem.
         */
        using problem_type = problem_t;
    private:
        /**
         * @brief Alias for the base class OptimUzawaBase.
         */
        using base_type = OptimUzawaBase<OptimUzawaMatrixFreeTbb<problem_t>, problem_t>;

    protected:
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
        template <std::size_t dim>
        OptimUzawaMatrixFreeTbb(std::size_t nparts,
                                double dt,
                                const scopi_container<dim>& particles,
                                const OptimParams<OptimUzawaMatrixFreeTbb<problem_t>>& optim_params,
                                const ProblemParams<problem_t>& problem_params);

    public:
        /**
         * @brief Implements the product \f$ \mathbb{P}^{-1} \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for masses and moments of inertia).
         */
        template <std::size_t dim>
        void gemv_inv_P_impl(const scopi_container<dim>& particles);

        /**
         * @brief Implements the product \f$ \mathbf{r} = \mathbf{r} - \mathbb{B} \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim>
        void gemv_A_impl(const scopi_container<dim>& particles,
                         const std::vector<neighbor<dim>>& contacts);

        /**
         * @brief Implements the product \f$ \mathbf{u} = \mathbb{B}^T \mathbf{l} + \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim>
        void gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                   const std::vector<neighbor<dim>>& contacts);

        /**
         * @brief For compatibility with other methods to compute matrix-vector products.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for positions).
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim>
        void init_uzawa_impl(const scopi_container<dim>& particles,
                             const std::vector<neighbor<dim>>& contacts);
        /**
         * @brief For compatibility with other methods to compute matrix-vector products.
         */
        void finalize_uzawa_impl();

    };

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb<problem_t>::init_uzawa_impl(const scopi_container<dim>&,
                                                             const std::vector<neighbor<dim>>&)
    {}

    template <class problem_t>
    void OptimUzawaMatrixFreeTbb<problem_t>::finalize_uzawa_impl()
    {}

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb<problem_t>::gemv_inv_P_impl(const scopi_container<dim>& particles)
    {
        auto active_offset = particles.nb_inactive();
        tbb::parallel_for(std::size_t(0), this->m_nparts, [&](std::size_t i) {
            this->matrix_free_gemv_inv_P(particles, this->m_U, active_offset, i);
        });
    }

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb<problem_t>::gemv_A_impl(const scopi_container<dim>& particles,
                                                         const std::vector<neighbor<dim>>& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();

        tbb::parallel_for(std::size_t(0), contacts.size(), [&](std::size_t ic)
        {
            auto &c = contacts[ic];
            this->matrix_free_gemv_A(c, particles, this->m_U, this->m_R, active_offset, ic);
        });
    }

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb<problem_t>::gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                                                   const std::vector<neighbor<dim>>& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();

        tbb::parallel_for(std::size_t(0), contacts.size(), [&](std::size_t ic)
        {
            auto &c = contacts[ic];
            this->matrix_free_gemv_transpose_A(c, particles, this->m_L, this->m_U, active_offset, ic);
        });
    }

    template <class problem_t>
    template <std::size_t dim>
    OptimUzawaMatrixFreeTbb<problem_t>::OptimUzawaMatrixFreeTbb(std::size_t nparts,
                                                                double dt,
                                                                const scopi_container<dim>&,
                                                                const OptimParams<OptimUzawaMatrixFreeTbb<problem_t>>& optim_params,
                                                                const ProblemParams<problem_t>& problem_params)
    : base_type(nparts, dt, optim_params, problem_params)
    {}

}
#endif

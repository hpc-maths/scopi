#pragma once

#include "OptimUzawaBase.hpp"
#include <cstddef>
#include <omp.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../quaternion.hpp"
#include "../utils.hpp"

namespace scopi{
    template<class problem_t>
    class OptimUzawaMatrixFreeOmp;

    /**
     * @brief Parameters for \c OptimUzawaMatrixFreeOmp<problem_t>
     *
     * Specialization of ProblemParams.
     * See OptimParamsUzawaBase.
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t>
    struct OptimParams<OptimUzawaMatrixFreeOmp<problem_t>> : public OptimParamsUzawaBase
    {
    };

    /**
     * @brief Uzawa algorithm with matrix-free matrix-vector products parallelized with OpenMP.
     *
     * See OptimUzawaBase for the algorithm.
     * \warning Only the cases <tt> problem_t = DryWithoutFriction </tt> and <tt> problem_t = ViscousWithoutFriction<dim> </tt> are implemented.
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t = DryWithoutFriction>
    class OptimUzawaMatrixFreeOmp :public OptimUzawaBase<OptimUzawaMatrixFreeOmp<problem_t>, problem_t>
    {
    public:
        /**
         * @brief Alias for the problem.
         */
        using problem_type = problem_t; 
    private:
        /**
         * @brief Alias for the base class \c OptimUzawaBase.
         */
        using base_type = OptimUzawaBase<OptimUzawaMatrixFreeOmp<problem_t>, problem_t>;

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
        OptimUzawaMatrixFreeOmp(std::size_t nparts,
                                double dt,
                                const scopi_container<dim>& particles,
                                const OptimParams<OptimUzawaMatrixFreeOmp<problem_t>>& optim_params,
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
         * @brief Implements the product \f$ \mathbf{u} = \mathbb{B}^T \l + \mathbf{u} \f$.
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
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         */
        template <std::size_t dim>
        void init_uzawa_impl(const scopi_container<dim>& particles,
                             const std::vector<neighbor<dim>>& contacts,
                             const std::vector<neighbor<dim>>& contacts_worms);
        /**
         * @brief For compatibility with other methods to compute matrix-vector products.
         */
        void finalize_uzawa_impl();

    };

    template <class problem_t>
    template <std::size_t dim>
    void OptimUzawaMatrixFreeOmp<problem_t>::init_uzawa_impl(const scopi_container<dim>&,
                                                             const std::vector<neighbor<dim>>&,
                                                             const std::vector<neighbor<dim>>&)
    {}

    template <class problem_t>
    void OptimUzawaMatrixFreeOmp<problem_t>::finalize_uzawa_impl()
    {}

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp<problem_t>::gemv_inv_P_impl(const scopi_container<dim>& particles)
    {
        auto active_offset = particles.nb_inactive();
        #pragma omp parallel for
        for (std::size_t i = 0; i < particles.nb_active(); ++i)
        {
            this->matrix_free_gemv_inv_P(particles, this->m_U, active_offset, i);
        }
    }

    template <class problem_t>
    template <std::size_t dim>
    void OptimUzawaMatrixFreeOmp<problem_t>::gemv_A_impl(const scopi_container<dim>& particles,
                                                         const std::vector<neighbor<dim>>& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();
        #pragma omp parallel for
        for (std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            auto &c = contacts[ic];
            this->matrix_free_gemv_A(c, particles, this->m_U, this->m_R, active_offset, ic);
        }
    }

    template <class problem_t>
    template <std::size_t dim>
    void OptimUzawaMatrixFreeOmp<problem_t>::gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                                                   const std::vector<neighbor<dim>>& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();
        #pragma omp parallel for
        for(std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            auto &c = contacts[ic];
            this->matrix_free_gemv_transpose_A(c, particles, this->m_L, this->m_U, active_offset, ic);
        }
    }

    template <class problem_t>
    template <std::size_t dim>
    OptimUzawaMatrixFreeOmp<problem_t>::OptimUzawaMatrixFreeOmp(std::size_t nparts,
                                                                double dt,
                                                                const scopi_container<dim>&,
                                                                const OptimParams<OptimUzawaMatrixFreeOmp<problem_t>>& optim_params,
                                                                const ProblemParams<problem_t>& problem_params)
    : base_type(nparts, dt, optim_params, problem_params)
    {}

}

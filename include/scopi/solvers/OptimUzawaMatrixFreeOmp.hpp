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

    template<class problem_t>
    struct OptimParams<OptimUzawaMatrixFreeOmp<problem_t>> : public OptimParamsUzawaBase
    {
    };

    template<class problem_t = DryWithoutFriction>
    class OptimUzawaMatrixFreeOmp :public OptimUzawaBase<OptimUzawaMatrixFreeOmp<problem_t>, problem_t>
    {
    protected:
        using problem_type = problem_t; 
    private:
        using base_type = OptimUzawaBase<OptimUzawaMatrixFreeOmp<problem_t>, problem_t>;

    protected:
        template <std::size_t dim>
        OptimUzawaMatrixFreeOmp(std::size_t nparts,
                                double dt,
                                const scopi_container<dim>& particles,
                                const OptimParams<OptimUzawaMatrixFreeOmp<problem_t>>& optim_params,
                                const ProblemParams<problem_t>& problem_params);

    public:
        template <std::size_t dim>
        void gemv_inv_P_impl(const scopi_container<dim>& particles);

        template <std::size_t dim>
        void gemv_A_impl(const scopi_container<dim>& particles,
                         const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void gemv_transpose_A_impl(const scopi_container<dim>& particles,
                                   const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void init_uzawa_impl(const scopi_container<dim>& particles,
                             const std::vector<neighbor<dim>>& contacts,
                             const std::vector<neighbor<dim>>& contacts_worms);
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

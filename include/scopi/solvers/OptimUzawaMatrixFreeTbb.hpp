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

    template<class problem_t>
    struct OptimParams<OptimUzawaMatrixFreeTbb<problem_t>> : public OptimParamsUzawaBase
    {
    };

    template<class problem_t = DryWithoutFriction>
    class OptimUzawaMatrixFreeTbb : public OptimUzawaBase<OptimUzawaMatrixFreeTbb<problem_t>, problem_t>
    {
    protected:
        using problem_type = problem_t; 
    private:
        using base_type = OptimUzawaBase<OptimUzawaMatrixFreeTbb<problem_t>, problem_t>;

    protected:
        template <std::size_t dim>
        OptimUzawaMatrixFreeTbb(std::size_t nparts,
                                double dt,
                                const scopi_container<dim>& particles,
                                const OptimParams<OptimUzawaMatrixFreeTbb<problem_t>>& optim_params,
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
    template<std::size_t dim>
    void OptimUzawaMatrixFreeTbb<problem_t>::init_uzawa_impl(const scopi_container<dim>&,
                                                             const std::vector<neighbor<dim>>&,
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

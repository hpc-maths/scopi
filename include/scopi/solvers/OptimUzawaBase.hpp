#pragma once

#ifdef SCOPI_USE_TBB
#include "OptimBase.hpp"
#include <omp.h>
#include "tbb/tbb.h"

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi{
    template<class D, std::size_t dim>
    class OptimUzawaBase : public crtp_base<D>, public OptimBase<OptimUzawaBase<D, dim>, dim>
    {
    public:
        OptimUzawaBase(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, double tol = 1e-9);
        void create_matrix_constraint_impl(const std::vector<neighbor<dim>>& contacts);
        void create_matrix_mass_impl();
        int solve_optimization_problem_impl(const std::vector<neighbor<dim>>& contacts);
        auto get_uadapt_impl();
        auto get_wadapt_impl();
        void allocate_memory_impl(const std::size_t nc);
        void free_memory_impl();
        int get_nb_active_contacts_impl();

    private:
        using base_type = crtp_base<D>;

    protected:
        void gemv_inv_P();
        void gemv_A(const std::vector<neighbor<dim>>& contacts);
        void gemv_transpose_A(const std::vector<neighbor<dim>>& contacts);

        const double m_tol;
        const std::size_t m_max_iter;
        const double m_rho;
        const double m_dmin;
        xt::xtensor<double, 1> m_U;
        xt::xtensor<double, 1> m_L;
        xt::xtensor<double, 1> m_R;
    };

    template<class D, std::size_t dim>
    OptimUzawaBase<D, dim>::OptimUzawaBase(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, double tol)
    : OptimBase<OptimUzawaBase<D, dim>, dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive, 0)
    , m_tol(tol)
    , m_max_iter(40000)
    , m_rho(2000.)
    , m_dmin(0.)
    , m_U(xt::zeros<double>({6*Nactive}))
    {}

    template<class D, std::size_t dim>
    void OptimUzawaBase<D, dim>::create_matrix_constraint_impl(const std::vector<neighbor<dim>>& contacts)
    {
        this->base_type::derived_cast().create_matrix_constraint_impl(contacts);
    }

    template<class D, std::size_t dim>
    void OptimUzawaBase<D, dim>::create_matrix_mass_impl()
    {
        this->base_type::derived_cast().create_matrix_mass_impl();
    }

    template<class D, std::size_t dim>
    int OptimUzawaBase<D, dim>::solve_optimization_problem_impl(const std::vector<neighbor<dim>>& contacts)
    {
        m_L = xt::zeros_like(this->m_distances);
        m_R = xt::zeros_like(this->m_distances);

        double time_assign_u = 0.;
        double time_gemv_transpose_A = 0.;
        double time_gemv_inv_P = 0.;
        double time_assign_r = 0.;
        double time_gemv_A = 0.;
        double time_assign_l = 0.;
        double time_compute_cmax = 0.;

        std::size_t cc = 0;
        double cmax = -1000.0;
        while ( (cmax<=-m_tol) && (cc <= m_max_iter) )
        {
            tic();
            m_U = this->m_c;
            time_assign_u += toc();

            tic();
            gemv_transpose_A(contacts); // U = A^T * L + U
            time_gemv_transpose_A += toc();

            tic();
            gemv_inv_P();  // U = - P^-1 * U
            time_gemv_inv_P += toc();

            tic();
            m_R = this->m_distances - m_dmin;
            time_assign_r += toc();

            tic();
            gemv_A(contacts); // R = - A * U + R
            time_gemv_A += toc();

            tic();
            m_L = xt::maximum( m_L-m_rho*m_R, 0);
            time_assign_l += toc();

            tic();
            cmax = double((xt::amin(m_R))(0));
            time_compute_cmax += toc();
            cc += 1;

            PLOG_VERBOSE << "-- C++ -- Projection : minimal constraint : " << cc << '\t' << cmax;
        }

        PLOG_ERROR_IF(cc >= m_max_iter) << "Uzawa does not converge";

        PLOG_INFO << "----> CPUTIME : solve (U = c) = " << time_assign_u;
        PLOG_INFO << "----> CPUTIME : solve (U = A^T*L+U) = " << time_gemv_transpose_A;
        PLOG_INFO << "----> CPUTIME : solve (U = -P^-1*U) = " << time_gemv_inv_P; 
        PLOG_INFO << "----> CPUTIME : solve (R = d) = " << time_assign_r;
        PLOG_INFO << "----> CPUTIME : solve (R = -A*U+R) = " << time_gemv_A;
        PLOG_INFO << "----> CPUTIME : solve (L = max(L-rho*R, 0)) = " << time_assign_l;
        PLOG_INFO << "----> CPUTIME : solve (cmax = min(R)) = " << time_compute_cmax;

        return cc;
    }

    template<class D, std::size_t dim>
    auto OptimUzawaBase<D, dim>::get_uadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_U.data()), {this->m_Nactive, 3UL});
    }

    template<class D, std::size_t dim>
    auto OptimUzawaBase<D, dim>::get_wadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_U.data()+3*this->m_Nactive), {this->m_Nactive, 3UL});
    }

    template<class D, std::size_t dim>
    void OptimUzawaBase<D, dim>::allocate_memory_impl(const std::size_t)
    {}

    template<class D, std::size_t dim>
    void OptimUzawaBase<D, dim>::free_memory_impl()
    {
        this->base_type::derived_cast().free_memory_impl();
    }

    template<class D, std::size_t dim>
    int OptimUzawaBase<D, dim>::get_nb_active_contacts_impl()
    {
        return xt::sum(xt::where(m_L > 0., xt::ones_like(m_L), xt::zeros_like(m_L)))();
    }

    template<class D, std::size_t dim>
    void OptimUzawaBase<D, dim>::gemv_inv_P()
    {
        this->base_type::derived_cast().gemv_inv_P_impl();
    }

    template<class D, std::size_t dim>
    void OptimUzawaBase<D, dim>::gemv_A(const std::vector<neighbor<dim>>& contacts)
    {
        this->base_type::derived_cast().gemv_A_impl(contacts);
    }

    template<class D, std::size_t dim>
    void OptimUzawaBase<D, dim>::gemv_transpose_A(const std::vector<neighbor<dim>>& contacts)
    {
        this->base_type::derived_cast().gemv_transpose_A_impl(contacts);
    }
}
#endif

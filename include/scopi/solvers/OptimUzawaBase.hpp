#pragma once

#include "OptimBase.hpp"

#include <omp.h>
#include "tbb/tbb.h"

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi{
    template<class Derived>
    class OptimUzawaBase: public OptimBase<Derived>
    {
    public:
        using base_type = OptimBase<Derived>;
        OptimUzawaBase(std::size_t nparts, double dt, double tol = 1e-9);

        template <std::size_t dim>
        int solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                            const std::vector<neighbor<dim>>& contacts);

        auto get_uadapt_impl();
        auto get_wadapt_impl();

        int get_nb_active_contacts_impl() const;

    protected:
        template <std::size_t dim>
        void gemv_inv_P(const scopi_container<dim>& particles);

        template <std::size_t dim>
        void gemv_A(const scopi_container<dim>& particles,
                    const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void gemv_transpose_A(const scopi_container<dim>& particles,
                              const std::vector<neighbor<dim>>& contacts);

        template <std::size_t dim>
        void init_uzawa(const std::vector<neighbor<dim>>& contacts);

        const double m_tol;
        const std::size_t m_max_iter;
        const double m_rho;
        const double m_dmin;
        xt::xtensor<double, 1> m_U;
        xt::xtensor<double, 1> m_L;
        xt::xtensor<double, 1> m_R;
    };

    template<class Derived>
    OptimUzawaBase<Derived>::OptimUzawaBase(std::size_t nparts, double dt, double tol)
    : base_type(nparts, dt, 2*3*nparts, 0)
    , m_tol(tol)
    , m_max_iter(40000)
    , m_rho(2000.)
    , m_dmin(0.)
    , m_U(xt::zeros<double>({6*nparts}))
    {}

    template<class Derived>
    template <std::size_t dim>
    int OptimUzawaBase<Derived>::solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                           const std::vector<neighbor<dim>>& contacts)
    {
        init_uzawa(contacts);
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
            gemv_transpose_A(particles, contacts); // U = A^T * L + U
            time_gemv_transpose_A += toc();

            tic();
            gemv_inv_P(particles);  // U = - P^-1 * U
            time_gemv_inv_P += toc();

            tic();
            m_R = this->m_distances - m_dmin;
            time_assign_r += toc();

            tic();
            gemv_A(particles, contacts); // R = - A * U + R
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

    template<class Derived>
    auto OptimUzawaBase<Derived>::get_uadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_U.data()), {this->m_nparts, 3UL});
    }

    template<class Derived>
    auto OptimUzawaBase<Derived>::get_wadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_U.data()+3*this->m_nparts), {this->m_nparts, 3UL});
    }

    template<class Derived>
    int OptimUzawaBase<Derived>::get_nb_active_contacts_impl() const
    {
        return xt::sum(xt::where(m_L > 0., xt::ones_like(m_L), xt::zeros_like(m_L)))();
    }

    template<class Derived>
    template <std::size_t dim>
    void OptimUzawaBase<Derived>::gemv_inv_P(const scopi_container<dim>& particles)
    {
        static_cast<Derived&>(*this).gemv_inv_P_impl(particles);
    }

    template<class Derived>
    template <std::size_t dim>
    void OptimUzawaBase<Derived>::gemv_A(const scopi_container<dim>& particles,
                                   const std::vector<neighbor<dim>>& contacts)
    {
        static_cast<Derived&>(*this).gemv_A_impl(particles, contacts);
    }

    template<class Derived>
    template <std::size_t dim>
    void OptimUzawaBase<Derived>::gemv_transpose_A(const scopi_container<dim>& particles,
                                             const std::vector<neighbor<dim>>& contacts)
    {
        static_cast<Derived&>(*this).gemv_transpose_A_impl(particles, contacts);
    }

    template<class Derived>
    template <std::size_t dim>
    void OptimUzawaBase<Derived>::init_uzawa(const std::vector<neighbor<dim>>& contacts)
    {
        static_cast<Derived&>(*this).init_uzawa_impl(contacts);
    }
}

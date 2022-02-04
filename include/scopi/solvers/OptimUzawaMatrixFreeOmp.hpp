#pragma once

#include "OptimBase.hpp"
#include <omp.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

#include "../quaternion.hpp"

namespace scopi{
    template<std::size_t dim>
    class OptimUzawaMatrixFreeOmp : public OptimBase<OptimUzawaMatrixFreeOmp<dim>, dim>
    {
    public:
        using base_type = OptimBase<OptimUzawaMatrixFreeOmp<dim>, dim>;

        OptimUzawaMatrixFreeOmp(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
        void create_matrix_constraint_impl(const std::vector<neighbor<dim>>& contacts);
        void create_matrix_mass_impl();
        int solve_optimization_problem_impl(const std::vector<neighbor<dim>>& contacts);
        auto get_uadapt_impl();
        auto get_wadapt_impl();
        void allocate_memory_impl(const std::size_t nc);
        void free_memory_impl();
        int get_nb_active_contacts_impl();

    private:
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
        int m_nb_active_contacts = 0;
    };

    template<std::size_t dim>
    OptimUzawaMatrixFreeOmp<dim>::OptimUzawaMatrixFreeOmp(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr)
    : OptimBase<OptimUzawaMatrixFreeOmp<dim>, dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive, 0)
    , m_tol(1.0e-11)
    , m_max_iter(40000)
    , m_rho(2000.)
    , m_dmin(0.)
    , m_U(xt::zeros<double>({6*Nactive}))
    {}

    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp<dim>::create_matrix_constraint_impl(const std::vector<neighbor<dim>>& contacts)
    {}

    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp<dim>::create_matrix_mass_impl()
    {}

    template<std::size_t dim>
    int OptimUzawaMatrixFreeOmp<dim>::solve_optimization_problem_impl(const std::vector<neighbor<dim>>& contacts)
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
        while ( (cmax<=-m_tol)&&(cc <= m_max_iter) )
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
            m_L = xt::maximum( m_L - m_rho*m_R, 0);
            time_assign_l += toc();

            tic();
            cmax = double((xt::amin(m_R))(0));
            time_compute_cmax += toc();
            cc += 1;
            // std::cout << "-- C++ -- Projection : minimal constraint : " << cmax << std::endl;
        }

        // if (cc > =m_max_iter)
        // {
        //     std::cout<<"\n-- C++ -- Projection : ********************** WARNING **********************"<<std::endl;
        //     std::cout<<  "-- C++ -- Projection : *************** Uzawa does not converge ***************"<<std::endl;
        //     std::cout<<  "-- C++ -- Projection : ********************** WARNING **********************\n"<<std::endl;
        // }

        // std::cout << "----> CPUTIME : solve (U = c) = " << time_assign_u << std::endl;
        // std::cout << "----> CPUTIME : solve (U = A^T*L+U) = " << time_gemv_transpose_A << std::endl;
        // std::cout << "----> CPUTIME : solve (U = -P^-1*U) = " << time_gemv_inv_P << std::endl;
        // std::cout << "----> CPUTIME : solve (R = d) = " << time_assign_r << std::endl;
        // std::cout << "----> CPUTIME : solve (R = -A*U+R) = " << time_gemv_A << std::endl;
        // std::cout << "----> CPUTIME : solve (L = max(L-rho*R, 0)) = " << time_assign_l << std::endl;
        // std::cout << "----> CPUTIME : solve (cmax = min(R)) = " << time_compute_cmax << std::endl;

        return cc;
    }

    template<std::size_t dim>
    auto OptimUzawaMatrixFreeOmp<dim>::get_uadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_U.data()), {this->m_Nactive, 3UL});
    }

    template<std::size_t dim>
    auto OptimUzawaMatrixFreeOmp<dim>::get_wadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_U.data()+3*this->m_Nactive), {this->m_Nactive, 3UL});
    }

    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp<dim>::allocate_memory_impl(const std::size_t nc)
    {}

    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp<dim>::free_memory_impl()
    {}

    template<std::size_t dim>
    int OptimUzawaMatrixFreeOmp<dim>::get_nb_active_contacts_impl()
    {
        return xt::sum(xt::where(m_L > 0., xt::ones_like(m_L), xt::zeros_like(m_L)))();
    }

    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp<dim>::gemv_inv_P()
    {
        #pragma omp parallel for
        for (std::size_t i = 0; i < this->m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                m_U(3*i + d) /= (-1. * this->m_mass); // TODO: add mass into particles
                m_U(3*this->m_Nactive + 3*i + d) /= (-1. * this->m_moment);
            }
        }
    }

    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp<dim>::gemv_A(const std::vector<neighbor<dim>>& contacts)
    {
        #pragma omp parallel for
        for (std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            auto &c = contacts[ic];
            for (std::size_t d = 0; d < 3; ++d)
            {
                if (c.i >= this->m_active_ptr)
                {
                    m_R(ic) -= (-this->m_dt*c.nij[d]) * m_U((c.i - this->m_active_ptr)*3 + d);
                }
                if (c.j >= this->m_active_ptr)
                {
                    m_R(ic) -= (this->m_dt*c.nij[d]) * m_U((c.j - this->m_active_ptr)*3 + d);
                }
            }

            auto r_i = c.pi - this->m_particles.pos()(c.i);
            auto r_j = c.pj - this->m_particles.pos()(c.j);

            xt::xtensor_fixed<double, xt::xshape<3, 3>> ri_cross, rj_cross;

            if (dim == 2)
            {
                ri_cross = {{      0,      0, r_i(1)},
                            {      0,      0, -r_i(0)},
                            {-r_i(1), r_i(0),       0}};

                rj_cross = {{      0,      0,  r_j(1)},
                            {      0,      0, -r_j(0)},
                            {-r_j(1), r_j(0),       0}};
            }
            else
            {
                ri_cross = {{      0, -r_i(2),  r_i(1)},
                            { r_i(2),       0, -r_i(0)},
                            {-r_i(1),  r_i(0),       0}};

                rj_cross = {{      0, -r_j(2),  r_j(1)},
                            { r_j(2),       0, -r_j(0)},
                            {-r_j(1),  r_j(0),       0}};
            }

            auto Ri = rotation_matrix<3>(this->m_particles.q()(c.i));
            auto Rj = rotation_matrix<3>(this->m_particles.q()(c.j));

            if (c.i >= this->m_active_ptr)
            {
                std::size_t ind_part = c.i - this->m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                for (std::size_t ip=0; ip<3; ++ip)
                {
                    m_R(ic) -= (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * m_U(3*this->m_Nactive + 3*ind_part + ip);
                }
            }

            if (c.j >= this->m_active_ptr)
            {
                std::size_t ind_part = c.j - this->m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    m_R(ic) -= (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * m_U(3*this->m_Nactive + 3*ind_part + ip);
                }
            }

        }
    }

    template<std::size_t dim>
    void OptimUzawaMatrixFreeOmp<dim>::gemv_transpose_A(const std::vector<neighbor<dim>>& contacts)
    {
        #pragma omp parallel for
        for(std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            auto &c = contacts[ic];

            for (std::size_t d = 0; d < 3; ++d)
            {
                if (c.i >= this->m_active_ptr)
                {
                    #pragma omp atomic
                    m_U((c.i - this->m_active_ptr)*3 + d) += m_L(ic) * (-this->m_dt*c.nij[d]);
                }
                if (c.j >= this->m_active_ptr)
                {
                    #pragma omp atomic
                    m_U((c.j - this->m_active_ptr)*3 + d) += m_L(ic) * (this->m_dt*c.nij[d]);
                }
            }

            auto r_i = c.pi - this->m_particles.pos()(c.i);
            auto r_j = c.pj - this->m_particles.pos()(c.j);

            xt::xtensor_fixed<double, xt::xshape<3, 3>> ri_cross, rj_cross;

            if (dim == 2)
            {
                ri_cross = {{      0,      0, r_i(1)},
                            {      0,      0, -r_i(0)},
                            {-r_i(1), r_i(0),       0}};

                rj_cross = {{      0,      0,  r_j(1)},
                            {      0,      0, -r_j(0)},
                            {-r_j(1), r_j(0),       0}};
            }
            else
            {
                ri_cross = {{      0, -r_i(2),  r_i(1)},
                            { r_i(2),       0, -r_i(0)},
                            {-r_i(1),  r_i(0),       0}};

                rj_cross = {{      0, -r_j(2),  r_j(1)},
                            { r_j(2),       0, -r_j(0)},
                            {-r_j(1),  r_j(0),       0}};
            }

            auto Ri = rotation_matrix<3>(this->m_particles.q()(c.i));
            auto Rj = rotation_matrix<3>(this->m_particles.q()(c.j));

            if (c.i >= this->m_active_ptr)
            {
                std::size_t ind_part = c.i - this->m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    #pragma omp atomic
                    m_U(3*this->m_Nactive + 3*ind_part + ip) += m_L(ic) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }

            if (c.j >= this->m_active_ptr)
            {
                std::size_t ind_part = c.j - this->m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    #pragma omp atomic
                    m_U(3*this->m_Nactive + 3*ind_part + ip) += m_L(ic) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }
        }
    }
}

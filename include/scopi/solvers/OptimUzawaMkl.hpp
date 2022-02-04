#pragma once

#ifdef SCOPI_USE_MKL
#include "OptimBase.hpp"
#include "mkl_service.h"
#include "mkl_spblas.h"
#include <stdio.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

namespace scopi{
    template<std::size_t dim>
        class OptimUzawaMkl : public OptimBase<OptimUzawaMkl<dim>, dim>
    {
        public:
            using base_type = OptimBase<OptimUzawaMkl<dim>, dim>;

            OptimUzawaMkl(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
            void create_matrix_constraint_impl(const std::vector<scopi::neighbor<dim>>& contacts);
            void create_matrix_mass_impl();
            int solve_optimization_problem_impl(const std::vector<scopi::neighbor<dim>>& contacts);
            auto get_uadapt_impl();
            auto get_wadapt_impl();
            void allocate_memory_impl(const std::size_t nc);
            void free_memory_impl();
            int get_nb_active_contacts_impl();

        private:
            void print_csr_matrix(const sparse_matrix_t);

            const double m_tol;
            const std::size_t m_max_iter;
            const double m_rho;
            const double m_dmin;
            xt::xtensor<double, 1> m_U;
            xt::xtensor<double, 1> m_L;
            xt::xtensor<double, 1> m_R;
            int m_nb_active_contacts = 0;
            sparse_matrix_t m_A;
            struct matrix_descr m_descrA;
            std::vector<MKL_INT> m_A_coo_row;
            std::vector<MKL_INT> m_A_coo_col;
            std::vector<double> m_A_coo_val;
            sparse_matrix_t m_inv_P;
            struct matrix_descr m_descr_inv_P;
            sparse_status_t m_status;

    };

    template<std::size_t dim>
    OptimUzawaMkl<dim>::OptimUzawaMkl(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr)
    : OptimBase<OptimUzawaMkl<dim>, dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive, 0)
    , m_tol(1.0e-11)
    , m_max_iter(40000)
    , m_rho(2000.)
    , m_dmin(0.)
    , m_U(xt::zeros<double>({6*Nactive}))
    {}

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::create_matrix_constraint_impl(const std::vector<scopi::neighbor<dim>>& contacts)
    {
        this->create_matrix_constraint_coo(contacts, m_A_coo_row, m_A_coo_col, m_A_coo_val, 0);

        sparse_matrix_t A_coo;
        m_status =  mkl_sparse_d_create_coo(&A_coo,
                                           SPARSE_INDEX_BASE_ZERO,
                                           contacts.size(), // number of rows
                                           6*this->m_Nactive, // number of cols
                                           m_A_coo_val.size(), // number of non-zero elements
                                           m_A_coo_row.data(),
                                           m_A_coo_col.data(),
                                           m_A_coo_val.data());
        if (m_status != SPARSE_STATUS_SUCCESS)
        {
            std::cout << " Error in mkl_sparse_d_create_coo for matrix A: " << m_status << std::endl;
        }

        m_status = mkl_sparse_convert_csr(A_coo,
                                          SPARSE_OPERATION_NON_TRANSPOSE,
                                          &m_A);
        if (m_status != SPARSE_STATUS_SUCCESS)
        {
            std::cout << " Error in mkl_sparse_convert_csr for matrix A: " << m_status << std::endl;
        }

        m_descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

        m_status = mkl_sparse_set_mv_hint(m_A, SPARSE_OPERATION_NON_TRANSPOSE, m_descrA, 1 );
        if (m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED)
        {
            std::cout << " Error in mkl_sparse_set_mv_hint for matrix A SPARSE_OPERATION_NON_TRANSPOSE: " << m_status << std::endl;
        }

        m_status = mkl_sparse_set_mv_hint(m_A, SPARSE_OPERATION_TRANSPOSE, m_descrA, 1 );
        if (m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED)
        {
            std::cout << " Error in mkl_sparse_set_mv_hint for matrix A SPARSE_OPERATION_TRANSPOSE: " << m_status << std::endl;
        }

        m_status = mkl_sparse_optimize ( m_A );
        if (m_status != SPARSE_STATUS_SUCCESS)
        {
            std::cout << " Error in mkl_sparse_optimize for matrix A: " << m_status << std::endl;
        }
    }

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::create_matrix_mass_impl()
    {
        std::vector<MKL_INT> invP_csr_row;
        std::vector<MKL_INT> invP_csr_col;
        std::vector<double> invP_csr_val;
        invP_csr_col.reserve(6*this->m_Nactive);
        invP_csr_row.reserve(6*this->m_Nactive+1);
        invP_csr_val.reserve(6*this->m_Nactive);

        for (std::size_t i = 0; i < this->m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*i + d);
                invP_csr_col.push_back(3*i + d);
                invP_csr_val.push_back(1./this->m_mass); // TODO: add mass into particles
            }
        }
        for (std::size_t i = 0; i < this->m_Nactive; ++i)
        {
            for (std::size_t d  =0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*this->m_Nactive + 3*i + d);
                invP_csr_col.push_back(3*this->m_Nactive + 3*i + d);
                invP_csr_val.push_back(1./this->m_moment);
            }
        }
        invP_csr_row.push_back(6*this->m_Nactive);

        m_status = mkl_sparse_d_create_csr(&m_inv_P,
                                           SPARSE_INDEX_BASE_ZERO,
                                           6*this->m_Nactive, // number of rows
                                           6*this->m_Nactive, // number of cols
                                           invP_csr_row.data(),
                                           invP_csr_row.data()+1,
                                           invP_csr_col.data(),
                                           invP_csr_val.data());
        if (m_status != SPARSE_STATUS_SUCCESS)
        {
            std::cout << " Error in mkl_sparse_d_create_csr for matrix invP: " << m_status << std::endl;
        }

        m_descr_inv_P.type = SPARSE_MATRIX_TYPE_DIAGONAL;
        m_descr_inv_P.diag = SPARSE_DIAG_NON_UNIT;

        m_status = mkl_sparse_set_mv_hint(m_inv_P, SPARSE_OPERATION_NON_TRANSPOSE, m_descr_inv_P, 1 );
        if (m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED)
        {
            std::cout << " Error in mkl_sparse_set_mv_hint for matrix invP: " << m_status << std::endl;
        }

        m_status = mkl_sparse_optimize ( m_inv_P );
        if (m_status != SPARSE_STATUS_SUCCESS)
        {
            std::cout << " Error in mkl_sparse_optimize for matrix invP: " << m_status << std::endl;
        }
    }

    template<std::size_t dim>
    int OptimUzawaMkl<dim>::solve_optimization_problem_impl(const std::vector<scopi::neighbor<dim>>& contacts)
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
            m_status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., m_A, m_descrA, m_L.data(), 1., m_U.data()); // U = A^T * L + U
            if (m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED)
            {
                std::cout << " Error in mkl_sparse_d_mv for U = A^T * L + U: " << m_status << std::endl;
                return -1;
            }
            time_gemv_transpose_A += toc();

            tic();
            m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_inv_P, m_descr_inv_P, m_U.data(), 0., m_U.data()); // U = - P^-1 * U
            if (m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED)
            {
                std::cout << " Error in mkl_sparse_d_mv for U = - P^-1 * U: " << m_status << std::endl;
                return -1;
            }
            time_gemv_inv_P += toc();

            tic();
            m_R = this->m_distances - m_dmin;
            time_assign_r += toc();

            tic();
            m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_A, m_descrA, m_U.data(), 1., m_R.data()); // R = - A * U + R
            if (m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED)
            {
                std::cout << " Error in mkl_sparse_d_mv for R = - A * U + R: " << m_status << std::endl;
                return -1;
            }
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

        // if (cc>=m_max_iter)
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
    auto OptimUzawaMkl<dim>::get_uadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_U.data()), {this->m_Nactive, 3UL});
    }

    template<std::size_t dim>
    auto OptimUzawaMkl<dim>::get_wadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_U.data()+3*this->m_Nactive), {this->m_Nactive, 3UL});
    }

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::allocate_memory_impl(const std::size_t nc)
    {}

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::free_memory_impl()
    {
        mkl_sparse_destroy ( m_inv_P );
        mkl_sparse_destroy ( m_A );
        m_A_coo_row.clear();
        m_A_coo_col.clear();
        m_A_coo_val.clear();
    }

    template<std::size_t dim>
    int OptimUzawaMkl<dim>::get_nb_active_contacts_impl()
    {
        return xt::sum(xt::where(m_L > 0., xt::ones_like(m_L), xt::zeros_like(m_L)))();
    }

    template<std::size_t dim>
    void OptimUzawaMkl<dim>::print_csr_matrix(const sparse_matrix_t A)
    {
        MKL_INT* csr_row_begin_ptr = NULL;
        MKL_INT* csr_row_end_ptr = NULL;
        MKL_INT* csr_col_ptr = NULL;
        double* csr_val_ptr = NULL;
        sparse_index_base_t indexing;
        MKL_INT nbRows;
        MKL_INT nbCols;
        m_status = mkl_sparse_d_export_csr(A, 
                                           &indexing,
                                           &nbRows,
                                           &nbCols,
                                           &csr_row_begin_ptr,
                                           &csr_row_end_ptr,
                                           &csr_col_ptr,
                                           &csr_val_ptr);

        if (m_status != SPARSE_STATUS_SUCCESS)
        {
            std::cout << " Error in mkl_sparse_d_export_csr: " << m_status << std::endl;
        }

        std::cout << "\nMatrix with " << nbRows << " rows and " << nbCols << " columns\n";
        std::cout << "RESULTANT MATRIX:\nrow# : (column, value) (column, value)\n";
        int ii = 0;
        for( int i = 0; i < nbRows; i++ )
        {
            std::cout << "row#" << i << ": ";
            for(MKL_INT j = csr_row_begin_ptr[i]; j < csr_row_end_ptr[i]; j++ )
            {
                std::cout << " (" << csr_col_ptr[ii] << ", " << csr_val_ptr[ii] << ")";
                ii++;
            }
            std::cout << std::endl;
        }
        std::cout << "_____________________________________________________________________  \n" ;
    }
}
#endif

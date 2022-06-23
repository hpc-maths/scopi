#pragma once

#ifdef SCOPI_USE_MKL
#include <mkl_spblas.h>

#include "OptimBase.hpp"

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../params/OptimParams.hpp"
#include "../problems/DryWithoutFriction.hpp"
#include "gradient/projected_gradient.hpp"

namespace scopi{
    template<class problem_t, class gradient_t>
    class OptimProjectedGradient;

    template<class problem_t, class gradient_t>
    class OptimParams<OptimProjectedGradient<problem_t, gradient_t>>
    {
    public:
        OptimParams();
        OptimParams(const OptimParams<OptimProjectedGradient<problem_t, gradient_t>>& params);

        ProblemParams<problem_t> m_problem_params;
        double m_tol_dg;
        double m_tol_l;
        std::size_t m_max_iter;
        double m_rho;
    };

    void print_csr_matrix(const sparse_matrix_t A)
    {
        // TODO free memory
        MKL_INT* csr_row_begin = NULL;
        MKL_INT* csr_row_end = NULL;
        MKL_INT* csr_col = NULL;
        double* csr_val = NULL;
        sparse_index_base_t indexing;
        MKL_INT nbRows;
        MKL_INT nbCols;
        auto status = mkl_sparse_d_export_csr(A,
                                           &indexing,
                                           &nbRows,
                                           &nbCols,
                                           &csr_row_begin,
                                           &csr_row_end,
                                           &csr_col,
                                           &csr_val);

        PLOG_ERROR_IF(status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_export_csr: " << status;

        std::cout << "\nMatrix with " << nbRows << " rows and " << nbCols << " columns\n";
        std::cout << "RESULTANT MATRIX:\nrow# : (column, value) (column, value)\n";
        int ii = 0;
        for( int i = 0; i < nbRows; i++ )
        {
            std::cout << "row#" << i << ": ";
            for(MKL_INT j = csr_row_begin[i]; j < csr_row_end[i]; j++ )
            {
                std::cout << " (" << csr_col[ii] << ", " << csr_val[ii] << ")";
                ii++;
            }
            std::cout << std::endl;
        }
        std::cout << "_____________________________________________________________________  \n" ;
    }

    template<class problem_t = DryWithoutFriction, class gradient_t = projected_gradient<projection_max>>
    class OptimProjectedGradient: public OptimBase<OptimProjectedGradient<problem_t, gradient_t>>
                                , public gradient_t
    {
    public:
        using base_type = OptimBase<OptimProjectedGradient<problem_t, gradient_t>>;
        using problem_type = problem_t; 
        template<std::size_t dim>
        OptimProjectedGradient(std::size_t nparts, double dt, const scopi_container<dim>& particles, const OptimParams<OptimProjectedGradient<problem_t, gradient_t>>& optim_params);

        template <std::size_t dim>
        std::size_t solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                    const std::vector<neighbor<dim>>& contacts,
                                                    problem_t& problem);

        auto uadapt_data();
        auto wadapt_data();
        auto lagrange_multiplier_data();

        int get_nb_active_contacts_impl() const;

    private:
        void set_moment_matrix(std::size_t nparts,
                               std::vector<MKL_INT>& invP_csr_row,
                               std::vector<MKL_INT>& invP_csr_col,
                               std::vector<double>& invP_csr_val,
                               const scopi_container<2>& particles);
        void set_moment_matrix(std::size_t nparts,
                               std::vector<MKL_INT>& invP_csr_row,
                               std::vector<MKL_INT>& invP_csr_col,
                               std::vector<double>& invP_csr_val,
                               const scopi_container<3>& particles);
        template <std::size_t dim>
        void create_matrix_B(const scopi_container<dim>& particles,
                             const std::vector<neighbor<dim>>& contacts,
                             problem_t& problem);
        void create_matrix_A();

        xt::xtensor<double, 1> m_l;
        xt::xtensor<double, 1> m_e; // vector c in 220517_PbDual_MiniForces.pdf
        xt::xtensor<double, 1> m_u;
        xt::xtensor<double, 1> m_bl;

        sparse_matrix_t m_A;
        struct matrix_descr m_descrA;
        sparse_matrix_t m_B;
        struct matrix_descr m_descrB;
        sparse_matrix_t m_inv_P;
        struct matrix_descr m_descr_inv_P;
        sparse_status_t m_status;
    };

    template<class problem_t, class gradient_t>
    template<std::size_t dim>
    OptimProjectedGradient<problem_t, gradient_t>::OptimProjectedGradient(std::size_t nparts, double dt, const scopi_container<dim>& particles, const OptimParams<OptimProjectedGradient<problem_t, gradient_t>>& optim_params)
    : base_type(nparts, dt, 2*3*nparts, 0, optim_params)
    , gradient_t(optim_params.m_max_iter, optim_params.m_rho, optim_params.m_tol_dg, optim_params.m_tol_l)
    , m_u(xt::zeros<double>({6*nparts}))
    , m_bl(xt::zeros<double>({6*nparts}))
    {
        std::vector<MKL_INT> invP_csr_row;
        std::vector<MKL_INT> invP_csr_col;
        std::vector<double> invP_csr_val;
        invP_csr_col.reserve(6*nparts);
        invP_csr_row.reserve(6*nparts+1);
        invP_csr_val.reserve(6*nparts);

        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*i + d);
                invP_csr_col.push_back(3*i + d);
                invP_csr_val.push_back(1./particles.m()(active_offset + i));
            }
        }
        set_moment_matrix(nparts, invP_csr_row, invP_csr_col, invP_csr_val, particles);
        invP_csr_row.push_back(6*nparts);

        m_status = mkl_sparse_d_create_csr(&m_inv_P,
                                           SPARSE_INDEX_BASE_ZERO,
                                           6*nparts, // number of rows
                                           6*nparts, // number of cols
                                           invP_csr_row.data(),
                                           invP_csr_row.data()+1,
                                           invP_csr_col.data(),
                                           invP_csr_val.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_create_csr for matrix invP: " << m_status;

        m_descr_inv_P.type = SPARSE_MATRIX_TYPE_DIAGONAL;
        m_descr_inv_P.diag = SPARSE_DIAG_NON_UNIT;

        m_status = mkl_sparse_set_mv_hint(m_inv_P, SPARSE_OPERATION_NON_TRANSPOSE, m_descr_inv_P, 2 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix invP: " << m_status;
        // m_status = mkl_sparse_set_mm_hint( m_inv_P, SPARSE_OPERATION_NON_TRANSPOSE, m_descr_inv_P, SPARSE_LAYOUT_ROW_MAJOR, 0, 1);
        // PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mm_hint for matrix invP: " << m_status;
        m_status = mkl_sparse_optimize ( m_inv_P );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix invP: " << m_status;
    }

    template<class problem_t, class gradient_t>
    template <std::size_t dim>
    std::size_t OptimProjectedGradient<problem_t, gradient_t>::solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                                                              const std::vector<neighbor<dim>>& contacts,
                                                                                              problem_t& problem)
    {
        xt::noalias(m_l) = xt::zeros<double>({problem.number_row_matrix(contacts)});
        // u = P^{-1}*c = vap
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1., m_inv_P, m_descr_inv_P, this->m_c.data(), 0., m_u.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for u = P^{-1}*c: " << m_status;

        create_matrix_B(particles, contacts, problem);
        create_matrix_A();

        // e = -B*u+distances
        xt::noalias(m_e) = problem.m_distances;
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_B, m_descrB, m_u.data(), 1., m_e.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for e = B*u+d: " << m_status;

        std::size_t nb_iter = this->projection(m_A, m_descrA, m_e, m_l);

        // u = u - P^{-1}*B^T*l
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., m_B, m_descrB, m_l.data(), 0., m_bl.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for bl = B^T*l: " << m_status;
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_inv_P, m_descr_inv_P, m_bl.data(), 1., m_u.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for u = vap - P^{-1}*bl: " << m_status;

        m_status = mkl_sparse_destroy ( m_B );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix B: " << m_status;
        m_status = mkl_sparse_destroy ( m_A );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix A: " << m_status;

        return nb_iter;
    }

    template<class problem_t, class gradient_t>
    auto OptimProjectedGradient<problem_t, gradient_t>::uadapt_data()
    {
        return m_u.data();
    }

    template<class problem_t, class gradient_t>
    auto OptimProjectedGradient<problem_t, gradient_t>::wadapt_data()
    {
        return m_u.data() + 3*this->m_nparts;
    }

    template<class problem_t, class gradient_t>
    auto OptimProjectedGradient<problem_t, gradient_t>::lagrange_multiplier_data()
    {
        return m_l.data();
    }

    template<class problem_t, class gradient_t>
    int OptimProjectedGradient<problem_t, gradient_t>::get_nb_active_contacts_impl() const
    {
        return xt::sum(xt::where(m_l > 0., xt::ones_like(m_l), xt::zeros_like(m_l)))();
    }

    template <class problem_t, class gradient_t>
    template <std::size_t dim>
    void OptimProjectedGradient<problem_t, gradient_t>::create_matrix_B(const scopi_container<dim>& particles,
                                                                        const std::vector<neighbor<dim>>& contacts,
                                                                        problem_t& problem)
    {
        m_descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t B_coo;

        problem.create_matrix_constraint_coo(particles, contacts, 0);
        m_status =  mkl_sparse_d_create_coo(&B_coo,
                                           SPARSE_INDEX_BASE_ZERO,
                                           problem.number_row_matrix(contacts), // number of rows
                                           6*this->m_nparts, // number of cols
                                           problem.m_A_values.size(), // number of non-zero elements
                                           problem.m_A_rows.data(),
                                           problem.m_A_cols.data(),
                                           problem.m_A_values.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_create_coo for matrix B: " << m_status;
        m_status = mkl_sparse_convert_csr(B_coo, SPARSE_OPERATION_NON_TRANSPOSE, &m_B);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_convert_csr for matrix B: " << m_status;
        m_status = mkl_sparse_destroy ( B_coo );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix B_coo: " << m_status;
        m_status = mkl_sparse_order(m_B);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_order for matrix B: " << m_status;
        m_status = mkl_sparse_set_mv_hint(m_B, SPARSE_OPERATION_NON_TRANSPOSE, m_descrB, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix B SPARSE_OPERATION_NON_TRANSPOSE: " << m_status;
        m_status = mkl_sparse_set_mv_hint(m_B, SPARSE_OPERATION_TRANSPOSE, m_descrB, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix B SPARSE_OPERATION_TRANSPOSE: " << m_status;
        // m_status = mkl_sparse_set_mm_hint(m_B, SPARSE_OPERATION_NON_TRANSPOSE, m_descrB, SPARSE_LAYOUT_ROW_MAJOR, 0, 1);
        // PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mm_hint for matrix B SPARSE_OPERATION_NON_TRANSPOSE: " << m_status;
        // m_status = mkl_sparse_set_mm_hint(m_B, SPARSE_OPERATION_TRANSPOSE, m_descrB, SPARSE_LAYOUT_ROW_MAJOR, 0, 1);
        // PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mm_hint for matrix B SPARSE_OPERATION_TRANSPOSE: " << m_status;
        m_status = mkl_sparse_optimize ( m_B );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix B: " << m_status;
    }

    template <class problem_t, class gradient_t>
    void OptimProjectedGradient<problem_t, gradient_t>::create_matrix_A()
    {
        m_descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t tmp;
        // tmp = P^{-1}*B
        m_descr_inv_P.type = SPARSE_MATRIX_TYPE_GENERAL;
        m_status = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, m_descr_inv_P, m_inv_P,
                                  SPARSE_OPERATION_TRANSPOSE, m_descrB, m_B,
                                  SPARSE_STAGE_FULL_MULT, &tmp);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_sp2m for tmp = P^{-1}*B^T: " << m_status;
        m_descr_inv_P.type = SPARSE_MATRIX_TYPE_DIAGONAL;
        m_descr_inv_P.diag = SPARSE_DIAG_NON_UNIT;
        m_status = mkl_sparse_order(tmp);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_order for matrix tmp: " << m_status;
        // A = B^T*tmp
        m_status = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, m_descrB, m_B,
                                  SPARSE_OPERATION_NON_TRANSPOSE, m_descrB, tmp,
                                  SPARSE_STAGE_FULL_MULT, &m_A);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_sp2m for A = B*tmp: " << m_status;
        m_status = mkl_sparse_destroy ( tmp );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix tmp: " << m_status;
        m_status = mkl_sparse_set_mv_hint(m_A, SPARSE_OPERATION_NON_TRANSPOSE, m_descrA, 2 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix A: " << m_status;
        m_status = mkl_sparse_optimize ( m_A );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix A: " << m_status;
    }

    template <class problem_t, class gradient_t>
    void OptimProjectedGradient<problem_t, gradient_t>::set_moment_matrix(std::size_t nparts,
                                                                          std::vector<MKL_INT>& invP_csr_row,
                                                                          std::vector<MKL_INT>& invP_csr_col,
                                                                          std::vector<double>& invP_csr_val,
                                                                          const scopi_container<2>& particles)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 2; ++d)
            {
                invP_csr_row.push_back(3*nparts + 3*i + d);
                invP_csr_col.push_back(3*nparts + 3*i + d);
                invP_csr_val.push_back(0.);
            }
            invP_csr_row.push_back(3*nparts + 3*i + 2);
            invP_csr_col.push_back(3*nparts + 3*i + 2);
            invP_csr_val.push_back(1./particles.j()(active_offset + i));
        }
    }

    template <class problem_t, class gradient_t>
    void OptimProjectedGradient<problem_t, gradient_t>::set_moment_matrix(std::size_t nparts,
                                                                         std::vector<MKL_INT>& invP_csr_row,
                                                                         std::vector<MKL_INT>& invP_csr_col,
                                                                         std::vector<double>& invP_csr_val,
                                                                         const scopi_container<3>& particles)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*nparts + 3*i + d);
                invP_csr_col.push_back(3*nparts + 3*i + d);
                invP_csr_val.push_back(1./particles.j()(active_offset + i)(d));
            }
        }
    }

    template<class problem_t, class gradient_t>
    OptimParams<OptimProjectedGradient<problem_t, gradient_t>>::OptimParams(const OptimParams<OptimProjectedGradient<problem_t, gradient_t>>& params)
    : m_problem_params(params.m_problem_params)
    , m_tol_dg(params.m_tol_dg)
    , m_tol_l(params.m_tol_l)
    , m_max_iter(params.m_max_iter)
    , m_rho(params.m_rho)
    {}

    template<class problem_t, class gradient_t>
    OptimParams<OptimProjectedGradient<problem_t, gradient_t>>::OptimParams()
    : m_problem_params()
    , m_tol_dg(1e-9)
    , m_tol_l(1e-9)
    , m_max_iter(40000)
    , m_rho(2000.)
    {}

}
#endif

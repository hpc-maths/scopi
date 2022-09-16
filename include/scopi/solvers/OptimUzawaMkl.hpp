#pragma once

#ifdef SCOPI_USE_MKL
#include "OptimUzawaBase.hpp"
#include "mkl_service.h"
#include "mkl_spblas.h"
#include "../problems/DryWithoutFriction.hpp"
#include <stdio.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{
    template<class problem_t>
    class OptimUzawaMkl;

    /**
     * @brief Parameters for \c OptimUzawaMatrixFreeOmp<problem_t>
     *
     * Specialization of ProblemParams.
     * See OptimParamsUzawaBase.
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t>
    struct OptimParams<OptimUzawaMkl<problem_t>> : public OptimParamsUzawaBase
    {
    };

    /**
     * @brief Uzawa algorithm where the matrices are stored and matrix-vector products are computed with the MKL.
     *
     * See OptimUzawaBase for the algorithm.
     * \warning Only the cases <tt> problem_t = DryWithoutFriction </tt> and <tt> problem_t = ViscousWithoutFriction<dim> </tt> are implemented.
     *
     * @tparam problem_t Problem to be solved.
     */
    template <class problem_t = DryWithoutFriction>
    class OptimUzawaMkl: public OptimUzawaBase<OptimUzawaMkl<problem_t>, problem_t>
    {
    protected:
        /**
         * @brief Alias for the problem.
         */
        using problem_type = problem_t; 
    private:
        /**
         * @brief Alias for the base class \c OptimUzawaBase.
         */
        using base_type = OptimUzawaBase<OptimUzawaMkl<problem_t>, problem_t>;

    protected:
        /**
         * @brief Constructor.
         *
         * Buils the matrix \f$ \mathbb{P}^{-1} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param nparts [in] Number of particles.
         * @param dt [in] Time step.
         * @param particles [in] Array of particles.
         * @param optim_params [in] Parameters.
         * @param problem_params [in] Parameters for the problem.
         */
        template <std::size_t dim>
        OptimUzawaMkl(std::size_t nparts,
                      double dt,
                      const scopi_container<dim>& particles,
                      const OptimParams<OptimUzawaMkl<problem_t>>& optim_params,
                      const ProblemParams<problem_t>& problem_params);
        /**
         * @brief Free the memory allocated for the matrix \f$ \mathbb{P}^{-1} \f$..
         */
        ~OptimUzawaMkl();

    public:
        /**
         * @brief Initialize the matrices for matrix-vector products with stored matrix.
         *
         * Builds the matrix \f$ \mathbb{B} \f$.
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
         * @brief Free the memory allocated for the matrix \f$ \mathbb{B} \f$..
         */
        void finalize_uzawa_impl();

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

    private:
        /**
         * @brief Prints a matrix on standard output.
         *
         * @param A [in] Matrix to print.
         */
        void print_csr_matrix(const sparse_matrix_t A);

        /**
         * @brief 2D implementation to set the moments of inertia in the matrix \f$ \mathbb{P}^{-1} \f$.
         *
         * @param nparts [in] Number of particles.
         * @param invP_csr_row [out] Rows' indicies of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param invP_csr_col [out] Columns' indicies of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param invP_csr_val [out] Values of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param particles [in] Array for particles (for moments of inertia).
         */
        void set_moment_matrix(std::size_t nparts,
                               std::vector<MKL_INT>& invP_csr_row,
                               std::vector<MKL_INT>& invP_csr_col,
                               std::vector<double>& invP_csr_val,
                               const scopi_container<2>& particles);
        /**
         * @brief 3D implementation to set the moments of inertia in the matrix \f$ \mathbb{P}^{-1} \f$.
         *
         * @param nparts [in] Number of particles.
         * @param invP_csr_row [out] Rows' indicies of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param invP_csr_col [out] Columns' indicies of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param invP_csr_val [out] Values of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param particles [in] Array for particles (for moments of inertia).
         */
        void set_moment_matrix(std::size_t nparts,
                               std::vector<MKL_INT>& invP_csr_row,
                               std::vector<MKL_INT>& invP_csr_col,
                               std::vector<double>& invP_csr_val,
                               const scopi_container<3>& particles);

        /**
         * @brief MKL's data structure for the matrix \f$ \mathbb{B} \f$.
         */
        sparse_matrix_t m_A;
        /**
         * @brief Structure specifying \f$ \mathbb{B} \f$ properties. 
         */
        struct matrix_descr m_descrA;
        /**
         * @brief MKL's data structure for the matrix \f$ \mathbb{P}^{-1} \f$.
         */
        sparse_matrix_t m_inv_P;
        /**
         * @brief Structure specifying \f$ \mathbb{P}^{-1} \f$ properties. 
         */
        struct matrix_descr m_descr_inv_P;
        /**
         * @brief Value indicating whether the operation was successful or not, and why.
         */
        sparse_status_t m_status;

    };

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMkl<problem_t>::init_uzawa_impl(const scopi_container<dim>& particles,
                                                  const std::vector<scopi::neighbor<dim>>& contacts,
                                                  const std::vector<scopi::neighbor<dim>>& contacts_worms)
    {
        this->create_matrix_constraint_coo(particles, contacts, contacts_worms, 0);

        sparse_matrix_t A_coo;
        m_status =  mkl_sparse_d_create_coo(&A_coo,
                                           SPARSE_INDEX_BASE_ZERO,
                                           this->number_row_matrix(contacts, contacts_worms), // number of rows
                                           6*this->m_nparts, // number of cols
                                           this->m_A_values.size(), // number of non-zero elements
                                           this->m_A_rows.data(),
                                           this->m_A_cols.data(),
                                           this->m_A_values.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_create_coo for matrix A: " << m_status;

        m_status = mkl_sparse_convert_csr(A_coo,
                                          SPARSE_OPERATION_NON_TRANSPOSE,
                                          &m_A);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_convert_csr for matrix A: " << m_status;

        m_descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

        m_status = mkl_sparse_set_mv_hint(m_A, SPARSE_OPERATION_NON_TRANSPOSE, m_descrA, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix A SPARSE_OPERATION_NON_TRANSPOSE: " << m_status;

        m_status = mkl_sparse_set_mv_hint(m_A, SPARSE_OPERATION_TRANSPOSE, m_descrA, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_set_mv_hint for matrix A SPARSE_OPERATION_TRANSPOSE: " << m_status;

        m_status = mkl_sparse_optimize ( m_A );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix A: " << m_status;
    }

    template <class problem_t>
    void OptimUzawaMkl<problem_t>::finalize_uzawa_impl()
    {
        m_status = mkl_sparse_destroy ( m_A );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix A: " << m_status;
    }

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMkl<problem_t>::gemv_inv_P_impl(const scopi_container<dim>&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_inv_P, m_descr_inv_P, this->m_U.data(), 0., this->m_U.data()); // U = - P^-1 * U
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for U = - P^-1 * U: " << m_status;
    }

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMkl<problem_t>::gemv_A_impl(const scopi_container<dim>&,
                                               const std::vector<neighbor<dim>>&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_A, m_descrA, this->m_U.data(), 1., this->m_R.data()); // R = - A * U + R
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for R = - A * U + R: " << m_status;
    }

    template <class problem_t>
    template<std::size_t dim>
    void OptimUzawaMkl<problem_t>::gemv_transpose_A_impl(const scopi_container<dim>&,
                                                         const std::vector<neighbor<dim>>&)
    {
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., m_A, m_descrA, this->m_L.data(), 1., this->m_U.data()); // U = A^T * L + U
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << " Error in mkl_sparse_d_mv for U = A^T * L + U: " << m_status;
    }

    template <class problem_t>
    template<std::size_t dim>
    OptimUzawaMkl<problem_t>::OptimUzawaMkl(std::size_t nparts,
                                            double dt,
                                            const scopi_container<dim>& particles,
                                            const OptimParams<OptimUzawaMkl<problem_t>>& optim_params,
                                            const ProblemParams<problem_t>& problem_params)
    : base_type(nparts, dt, optim_params, problem_params)
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

        m_status = mkl_sparse_set_mv_hint(m_inv_P, SPARSE_OPERATION_NON_TRANSPOSE, m_descr_inv_P, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix invP: " << m_status;

        m_status = mkl_sparse_optimize ( m_inv_P );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix invP: " << m_status;
    }

    template <class problem_t>
    OptimUzawaMkl<problem_t>::~OptimUzawaMkl()
    {
        m_status = mkl_sparse_destroy ( m_inv_P );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix P^-1: " << m_status;
    }

    template <class problem_t>
    void OptimUzawaMkl<problem_t>::print_csr_matrix(const sparse_matrix_t A)
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

        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_export_csr: " << m_status;

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

    template <class problem_t>
    void OptimUzawaMkl<problem_t>::set_moment_matrix(std::size_t nparts,
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

    template <class problem_t>
    void OptimUzawaMkl<problem_t>::set_moment_matrix(std::size_t nparts,
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
}
#endif

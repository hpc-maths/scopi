#pragma once

#include "OptimBase.hpp"
#include "mkl_service.h"
#include "mkl_spblas.h"
#include <stdio.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

namespace scopi{
    template<std::size_t dim>
        class OptimUzawa : public OptimBase<OptimUzawa<dim>, dim>
    {
        public:
            using base_type = OptimBase<OptimUzawa<dim>, dim>;

            OptimUzawa(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
            void createMatrixConstraint_impl(const std::vector<scopi::neighbor<dim>>& contacts);
            void createMatrixMass_impl();
            int solveOptimizationProblem_impl(const std::vector<scopi::neighbor<dim>>& contacts);
            auto getUadapt_impl();
            auto getWadapt_impl();
            void allocateMemory_impl(const std::size_t nc);
            void freeMemory_impl();
            int getNbActiveContacts_impl();

        private:
            int testMkl();
            void printCrsMatrix(const sparse_matrix_t);

            const double _tol;
            const std::size_t _maxiter;
            const double _rho;
            const double _dmin;
            xt::xtensor<double, 1> _U;
            int _nbActiveContacts = 0;
            sparse_matrix_t _A;
            struct matrix_descr _descrA;
            sparse_matrix_t _invP;
            struct matrix_descr _descrInvP;

    };

    template<std::size_t dim>
        OptimUzawa<dim>::OptimUzawa(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr) : 
            OptimBase<OptimUzawa<dim>, dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive, 0),
            _tol(1.0e-2), _maxiter(40000), _rho(2000.), _dmin(0.),
            _U(xt::zeros<double>({6*Nactive}))
            {
            }

    template<std::size_t dim>
        void OptimUzawa<dim>::createMatrixConstraint_impl(const std::vector<scopi::neighbor<dim>>& contacts)
        {
            std::vector<MKL_INT> _coo_rows;
            std::vector<MKL_INT> _coo_cols;
            std::vector<double> _coo_vals;
            this->createMatrixConstraintCoo(contacts, _coo_rows, _coo_cols, _coo_vals, 0);

            sparse_matrix_t A_coo;
            auto _status =  mkl_sparse_d_create_coo
                (&A_coo,
                 SPARSE_INDEX_BASE_ZERO,
                 contacts.size(),    // number of rows
                 6*this->_Nactive,   // number of cols
                 _coo_vals.size(),   // number of non-zero elements
                 _coo_rows.data(),
                 _coo_cols.data(),
                 _coo_vals.data()
                );
            if (_status != SPARSE_STATUS_SUCCESS)
            {
                std::cout << " Error in mkl_sparse_d_create_coo for matrix A: " << _status << std::endl;
            }

            _status = mkl_sparse_convert_csr
                (A_coo,
                 SPARSE_OPERATION_NON_TRANSPOSE,
                 &_A
                );
            if (_status != SPARSE_STATUS_SUCCESS)
            {
                std::cout << " Error in mkl_sparse_convert_csr for matrix A: " << _status << std::endl;
            }

            _descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

            _status = mkl_sparse_set_mv_hint(_A, SPARSE_OPERATION_NON_TRANSPOSE, _descrA, 1 );
            if (_status != SPARSE_STATUS_SUCCESS && _status != SPARSE_STATUS_NOT_SUPPORTED)
            {
                std::cout << " Error in mkl_sparse_set_mv_hint for matrix A SPARSE_OPERATION_NON_TRANSPOSE: " << _status << std::endl;
            }

            _status = mkl_sparse_set_mv_hint(_A, SPARSE_OPERATION_TRANSPOSE, _descrA, 1 );
            if (_status != SPARSE_STATUS_SUCCESS && _status != SPARSE_STATUS_NOT_SUPPORTED)
            {
                std::cout << " Error in mkl_sparse_set_mv_hint for matrix A SPARSE_OPERATION_TRANSPOSE: " << _status << std::endl;
            }

            _status = mkl_sparse_optimize ( _A );
            if (_status != SPARSE_STATUS_SUCCESS)
            {
                std::cout << " Error in mkl_sparse_optimize for matrix A: " << _status << std::endl;
            }
        }

    template<std::size_t dim>
        void OptimUzawa<dim>::createMatrixMass_impl()
        {
            std::vector<MKL_INT> _row;
            std::vector<MKL_INT> _col;
            std::vector<double> _val;
            _col.reserve(6*this->_Nactive);
            _row.reserve(6*this->_Nactive+1);
            _val.reserve(6*this->_Nactive);

            for (std::size_t i=0; i<this->_Nactive; ++i)
            {
                for (std::size_t d=0; d<3; ++d)
                {
                    _row.push_back(3*i + d);
                    _col.push_back(3*i + d);
                    _val.push_back(1./this->_mass); // TODO: add mass into particles
                }
            }
            for (std::size_t i=0; i<this->_Nactive; ++i)
            {
                for (std::size_t d=0; d<3; ++d)
                {
                    _row.push_back(3*this->_Nactive + 3*i + d);
                    _col.push_back(3*this->_Nactive + 3*i + d);
                    _val.push_back(1./this->_moment);
                }
            }
            _row.push_back(6*this->_Nactive);

            auto _status = mkl_sparse_d_create_csr( &_invP,
                    SPARSE_INDEX_BASE_ZERO,
                    6*this->_Nactive,    // number of rows
                    6*this->_Nactive,    // number of cols
                    _row.data(),
                    _row.data()+1,
                    _col.data(),
                    _val.data() );
            if (_status != SPARSE_STATUS_SUCCESS)
            {
                std::cout << " Error in mkl_sparse_d_create_csr for matrix invP: " << _status << std::endl;
            }

            _descrInvP.type = SPARSE_MATRIX_TYPE_DIAGONAL;
            _descrInvP.diag = SPARSE_DIAG_NON_UNIT;

            _status = mkl_sparse_set_mv_hint(_invP, SPARSE_OPERATION_NON_TRANSPOSE, _descrInvP, 1 );
            if (_status != SPARSE_STATUS_SUCCESS && _status != SPARSE_STATUS_NOT_SUPPORTED)
            {
                std::cout << " Error in mkl_sparse_set_mv_hint for matrix invP: " << _status << std::endl;
            }

            _status = mkl_sparse_optimize ( _invP );
            if (_status != SPARSE_STATUS_SUCCESS)
            {
                std::cout << " Error in mkl_sparse_optimize for matrix invP: " << _status << std::endl;
            }
        }

    template<std::size_t dim>
        int OptimUzawa<dim>::solveOptimizationProblem_impl(const std::vector<scopi::neighbor<dim>>& contacts)
        {
            auto L = xt::zeros_like(this->_distances);
            auto R = xt::zeros_like(this->_distances);

            std::size_t cc = 0;
            double cmax = -1000.0;
            while ( (cmax<=-_tol)&&(cc <= _maxiter) )
            {
                _U = this->_c;

                auto _status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., _A, _descrA, &L[0], 1., &_U[0]); // U = A^T * L + U
                if (_status != SPARSE_STATUS_SUCCESS && _status != SPARSE_STATUS_NOT_SUPPORTED)
                {
                    std::cout << " Error in mkl_sparse_d_mv for U = A^T * L + U: " << _status << std::endl;
                    return -1;
                }

                _status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., _invP, _descrInvP, &_U[0], 0., &_U[0]); // U = - P^-1 * U
                if (_status != SPARSE_STATUS_SUCCESS && _status != SPARSE_STATUS_NOT_SUPPORTED)
                {
                    std::cout << " Error in mkl_sparse_d_mv for U = - P^-1 * U: " << _status << std::endl;
                    return -1;
                }

                R = this->_distances - _dmin;

                _status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., _A, _descrA, &_U[0], 1., &R[0]); // R = - A * U + R
                if (_status != SPARSE_STATUS_SUCCESS && _status != SPARSE_STATUS_NOT_SUPPORTED)
                {
                    printf(" Error in mkl_sparse_d_mv R = A U: %d \n", _status);
                    std::cout << " Error in mkl_sparse_d_mv for R = - A * U + R: " << _status << std::endl;
                    return -1;
                }

                L = xt::maximum( L-_rho*R, 0);

                cmax = double((xt::amin(R))(0));
                cc += 1;
                // std::cout << "-- C++ -- Projection : minimal constraint : " << cmax << std::endl;
            }

            if (cc>=_maxiter)
            {
                std::cout<<"\n-- C++ -- Projection : ********************** WARNING **********************"<<std::endl;
                std::cout<<  "-- C++ -- Projection : *************** Uzawa does not converge ***************"<<std::endl;
                std::cout<<  "-- C++ -- Projection : ********************** WARNING **********************\n"<<std::endl;
            }

            // TODO use xt functions such as xt::where
            _nbActiveContacts = 0;
            for(std::size_t i = 0; i < contacts.size(); ++i)
            {
                if(L(i) > 0.)
                {
                    _nbActiveContacts++;
                }
            }

            mkl_sparse_destroy ( _invP );
            mkl_sparse_destroy ( _A );

            return cc;
        }

    template<std::size_t dim>
        int OptimUzawa<dim>::testMkl()
        {
#ifdef MKL_ILP64
#define INT_PRINT_FORMAT "%lld"
#else
#define INT_PRINT_FORMAT "%d"
#endif
            //*******************************************************************************
            //     Definition arrays for sparse representation of the matrix A in
            //     the coordinate format:
            //*******************************************************************************
#define M 5    /* nrows = ncols = M */
#define NNZ 13
            MKL_INT m = M;

            MKL_INT colIndex[M+1] = {   0,               3,              6,              9,        11,       13};
            MKL_INT rows[NNZ]     = {   0,   1,    3,    0,   1,   4,    0,   2,   3,    2,   3,    2,   4};
            double values[NNZ]    = { 1.0 -2.0, -4.0, -1.0, 5.0, 8.0, -3.0, 4.0, 2.0,  6.0, 7.0, 4.0, -5.0};

            //*******************************************************************************
            //    Declaration of local variables :
            //*******************************************************************************


            double      sol_vec[M]  = {1.0, 1.0, 1.0, 1.0, 1.0};
            double      rhs_vec[M]  = {0.0, 0.0, 0.0, 0.0, 0.0};

            double      alpha = 1.0, beta = 0.0;
            MKL_INT     i;
            struct matrix_descr descrA;
            sparse_matrix_t cscA;

            sparse_status_t status;
            int exit_status = 0;

            printf( "\n EXAMPLE PROGRAM FOR CSC format routines from IE Sparse BLAS\n" );
            printf( "-------------------------------------------------------\n" );

            //*******************************************************************************
            //   Create CSC sparse matrix handle and analyze step
            //*******************************************************************************

            status = mkl_sparse_d_create_csc( &cscA,
                    SPARSE_INDEX_BASE_ZERO,
                    m,    // number of rows
                    m,    // number of cols
                    colIndex,
                    colIndex+1,
                    rows,
                    values );

            if (status != SPARSE_STATUS_SUCCESS) {
                printf(" Error in mkl_sparse_d_create_csc: %d \n", status);
                exit_status = 1;
                goto exit;
            }

            //*******************************************************************************
            // First we set hints for the different operations before calling the
            // mkl_sparse_optimize() api which actually does the analyze step.  Not all
            // configurations have optimized steps, so the hint apis may return status
            // MKL_SPARSE_STATUS_NOT_SUPPORTED (=6) if no analysis stage is actually available
            // for that configuration.
            //*******************************************************************************

            //*******************************************************************************
            // Set hints for Task 2: Upper triangular transpose MV and SV solve
            // with unit diagonal
            //*******************************************************************************
            descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

            status = mkl_sparse_set_mv_hint(cscA, SPARSE_OPERATION_NON_TRANSPOSE, descrA, 1 );
            if (status != SPARSE_STATUS_SUCCESS && status != SPARSE_STATUS_NOT_SUPPORTED) {
                printf(" Error in set hints for Task 2: mkl_sparse_set_mv_hint: %d \n", status);
                exit_status = 1;
                goto exit;
            }

            status = mkl_sparse_set_sv_hint(cscA, SPARSE_OPERATION_NON_TRANSPOSE, descrA, 1 );
            if (status != SPARSE_STATUS_SUCCESS && status != SPARSE_STATUS_NOT_SUPPORTED) {
                printf(" Error in set hints for Task 2: mkl_sparse_set_sv_hint: %d \n", status);
                exit_status = 1;
                goto exit;
            }
            //*******************************************************************************
            // Analyze sparse matrix; choose proper kernels and workload balancing strategy
            //*******************************************************************************
            status = mkl_sparse_optimize ( cscA );
            if (status != SPARSE_STATUS_SUCCESS) {
                printf(" Error in mkl_sparse_optimize: %d \n", status);
                exit_status = 1;
                goto exit;
            }
            //*******************************************************************************
            //    Task 2.    Obtain Triangular matrix-vector multiply (U+I) *sol --> rhs
            //    and solve triangular system   (U+I) *tmp = rhs with single right hand sides
            //    Array tmp must be equal to the array sol
            //*******************************************************************************
            printf("                                     \n");
            printf("   TASK 2:                           \n");
            printf("   INPUT DATA FOR mkl_sparse_d_mv    \n");
            printf("   WITH UNIT UPPER TRIANGULAR MATRIX \n");
            printf("     ALPHA = %4.1f  BETA = %4.1f     \n", alpha, beta);
            printf("     SPARSE_OPERATION_NON_TRANSPOSE  \n" );
            printf("   Input vector                      \n");
            for (i = 0; i < m; i++) {
                printf("%7.1f\n", sol_vec[i]);
            };

            descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

            status = mkl_sparse_d_mv( SPARSE_OPERATION_NON_TRANSPOSE, alpha, cscA, descrA, sol_vec, beta, rhs_vec);
            if (status != SPARSE_STATUS_SUCCESS) {
                printf(" Error in Task 2 mkl_sparse_d_mv: %d \n", status);
                exit_status = 1;
                goto exit;
            }

            printf("                                   \n");
            printf("   OUTPUT DATA FOR mkl_sparse_d_mv \n");
            printf("   WITH TRIANGULAR MATRIX          \n");
            for (i = 0; i < m; i++) {
                printf("%7.1f\n", rhs_vec[i]);
            };


exit:
            // Release matrix handle and deallocate matrix
            mkl_sparse_destroy ( cscA );

            return exit_status;

        }

    template<std::size_t dim>
        auto OptimUzawa<dim>::getUadapt_impl()
        {
            return xt::adapt(reinterpret_cast<double*>(_U.data()), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        auto OptimUzawa<dim>::getWadapt_impl()
        {
            return xt::adapt(reinterpret_cast<double*>(_U.data()+3*this->_Nactive), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        void OptimUzawa<dim>::allocateMemory_impl(const std::size_t nc)
        {
            std::ignore = nc;
        }

    template<std::size_t dim>
        void OptimUzawa<dim>::freeMemory_impl()
        {
        }

    template<std::size_t dim>
        int OptimUzawa<dim>::getNbActiveContacts_impl()
        {
            // TODO L as a member of the class and do the computation here
            return _nbActiveContacts;
        }

    template<std::size_t dim>
        void OptimUzawa<dim>::printCrsMatrix(const sparse_matrix_t A)
        {
            MKL_INT* csr_row_begin_ptr = NULL;
            MKL_INT* csr_row_end_ptr = NULL;
            MKL_INT* csr_col_ptr = NULL;
            double* csr_val_ptr = NULL;
            sparse_index_base_t indexing;
            MKL_INT nbRows;
            MKL_INT nbCols;
            auto status = mkl_sparse_d_export_csr
                (
                 A,
                 &indexing,
                 &nbRows,
                 &nbCols,
                 &csr_row_begin_ptr,
                 &csr_row_end_ptr,
                 &csr_col_ptr,
                 &csr_val_ptr
                );

            if (status != SPARSE_STATUS_SUCCESS)
            {
                std::cout << " Error in mkl_sparse_d_export_csr: " << status << std::endl;
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

#pragma once

#include "mkl_service.h"
#include "mkl_spblas.h"
#include <stdio.h>

namespace scopi{
    template<std::size_t dim>
        class UzawaSolver : public OptimizationSolver<dim>
    {
        public:
            UzawaSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
            void createMatrixConstraint(std::vector<scopi::neighbor<dim>>& contacts);
            void createMatrixMass();
            int solveOptimizationProbelm(std::vector<scopi::neighbor<dim>>& contacts);
            auto getUadapt();
            auto getWadapt();
            void allocateMemory(std::size_t nc);
            void freeMemory();

        private:
            int testMkl();
    };

    template<std::size_t dim>
        UzawaSolver<dim>::UzawaSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr) : 
            OptimizationSolver<dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive + 2*3*Nactive, 0)
    {
    }

    template<std::size_t dim>
        void UzawaSolver<dim>::createMatrixConstraint(std::vector<scopi::neighbor<dim>>& contacts)
        {
        }

    template<std::size_t dim>
        void UzawaSolver<dim>::createMatrixMass()
        {
        }

    template<std::size_t dim>
        int UzawaSolver<dim>::solveOptimizationProbelm(std::vector<scopi::neighbor<dim>>& contacts)
        {
            return 0;
        }

    template<std::size_t dim>
        int UzawaSolver<dim>::testMkl()
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
        auto UzawaSolver<dim>::getUadapt()
        {
            std::vector<double> v(this->_Nactive * 3UL, 0.);
            return xt::adapt(v, {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        auto UzawaSolver<dim>::getWadapt()
        {
            std::vector<double> v(this->_Nactive * 3UL, 0.);
            return xt::adapt(v, {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        void UzawaSolver<dim>::allocateMemory(std::size_t nc)
        {
            std::ignore = nc;
        }

    template<std::size_t dim>
        void UzawaSolver<dim>::freeMemory()
        {
        }
}

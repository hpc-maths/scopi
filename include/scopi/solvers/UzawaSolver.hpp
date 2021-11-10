#pragma once

// #include "mkl_service.h"
#include "mkl_spblas.h"

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
            /*
            constexpr MKL_INT M = 4;
            constexpr MKL_INT N = 6;
            constexpr MKL_INT NNZ = 8;

            float   csrVal[NNZ]      = { 1., 2., 3., 4., 5., 6., 7., 8. };
            MKL_INT csrColInd[NNZ]   = { 0, 1, 1, 3, 2, 3, 4, 5 };
            MKL_INT csrRowPtr[M + 1] = { 0, 2, 4, 7, 8 };

            sparse_matrix_t csrA;
            mkl_sparse_s_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO, N, M,
                    csrRowPtr, csrRowPtr + 1, csrColInd, csrVal);

            float x[N]  = { 0., 1., 2., 3., 4., 5. };
            float y[N]  = { 0., 0., 0., 0., 0., 0. };
            float alpha = 1., beta = 0.;

            matrix_descr descrA;
            descrA.type = SPARSE_MATRIX_TYPE_GENERAL;

            mkl_sparse_optimize(csrA);
            mkl_sparse_s_mv(SPARSE_OPERATION_NON_TRANSPOSE, alpha, csrA, descrA, x, beta, y);
            mkl_sparse_destroy(csrA);
            */

            return 0;
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

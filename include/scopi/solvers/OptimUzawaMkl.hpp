#pragma once

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
            void createMatrixConstraint_impl(const std::vector<scopi::neighbor<dim>>& contacts);
            void createMatrixMass_impl();
            int solveOptimizationProblem_impl();
            auto getUadapt_impl();
            auto getWadapt_impl();
            void allocateMemory_impl(const std::size_t nc);
            void freeMemory_impl();
            int getNbActiveContacts_impl();

        private:
            void printCrsMatrix(const sparse_matrix_t);

            const double _tol;
            const std::size_t _maxiter;
            const double _rho;
            const double _dmin;
            xt::xtensor<double, 1> _U;
            xt::xtensor<double, 1> _L;
            xt::xtensor<double, 1> _R;
            int _nbActiveContacts = 0;
            sparse_matrix_t _A;
            struct matrix_descr _descrA;
            std::vector<MKL_INT> _A_coo_row;
            std::vector<MKL_INT> _A_coo_col;
            std::vector<double> _A_coo_val;
            sparse_matrix_t _invP;
            struct matrix_descr _descrInvP;
            sparse_status_t _status;

    };

    template<std::size_t dim>
        OptimUzawaMkl<dim>::OptimUzawaMkl(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr) : 
            OptimBase<OptimUzawaMkl<dim>, dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive, 0),
            _tol(1.0e-2), _maxiter(40000), _rho(200.), _dmin(0.),
            _U(xt::zeros<double>({6*Nactive}))
            {
            }

    template<std::size_t dim>
        void OptimUzawaMkl<dim>::createMatrixConstraint_impl(const std::vector<scopi::neighbor<dim>>& contacts)
        {
            this->createMatrixConstraintCoo(contacts, _A_coo_row, _A_coo_col, _A_coo_val, 0);

            sparse_matrix_t A_coo;
            _status =  mkl_sparse_d_create_coo
                (&A_coo,
                 SPARSE_INDEX_BASE_ZERO,
                 contacts.size(),    // number of rows
                 6*this->_Nactive,   // number of cols
                 _A_coo_val.size(),   // number of non-zero elements
                 _A_coo_row.data(),
                 _A_coo_col.data(),
                 _A_coo_val.data()
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
        void OptimUzawaMkl<dim>::createMatrixMass_impl()
        {
            std::vector<MKL_INT> invP_csr_row;
            std::vector<MKL_INT> invP_csr_col;
            std::vector<double> invP_csr_val;
            invP_csr_col.reserve(6*this->_Nactive);
            invP_csr_row.reserve(6*this->_Nactive+1);
            invP_csr_val.reserve(6*this->_Nactive);

            for (std::size_t i=0; i<this->_Nactive; ++i)
            {
                for (std::size_t d=0; d<3; ++d)
                {
                    invP_csr_row.push_back(3*i + d);
                    invP_csr_col.push_back(3*i + d);
                    invP_csr_val.push_back(1./this->_mass); // TODO: add mass into particles
                }
            }
            for (std::size_t i=0; i<this->_Nactive; ++i)
            {
                for (std::size_t d=0; d<3; ++d)
                {
                    invP_csr_row.push_back(3*this->_Nactive + 3*i + d);
                    invP_csr_col.push_back(3*this->_Nactive + 3*i + d);
                    invP_csr_val.push_back(1./this->_moment);
                }
            }
            invP_csr_row.push_back(6*this->_Nactive);

            _status = mkl_sparse_d_create_csr( &_invP,
                    SPARSE_INDEX_BASE_ZERO,
                    6*this->_Nactive,    // number of rows
                    6*this->_Nactive,    // number of cols
                    invP_csr_row.data(),
                    invP_csr_row.data()+1,
                    invP_csr_col.data(),
                    invP_csr_val.data() );
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
        int OptimUzawaMkl<dim>::solveOptimizationProblem_impl()
        {
            _L = xt::zeros_like(this->_distances);
            _R = xt::zeros_like(this->_distances);

            std::size_t cc = 0;
            double cmax = -1000.0;
            while ( (cmax<=-_tol)&&(cc <= _maxiter) )
            {
                _U = this->_c;

                _status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., _A, _descrA, _L.data(), 1., _U.data()); // U = A^T * L + U
                if (_status != SPARSE_STATUS_SUCCESS && _status != SPARSE_STATUS_NOT_SUPPORTED)
                {
                    std::cout << " Error in mkl_sparse_d_mv for U = A^T * L + U: " << _status << std::endl;
                    return -1;
                }

                _status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., _invP, _descrInvP, _U.data(), 0., _U.data()); // U = - P^-1 * U
                if (_status != SPARSE_STATUS_SUCCESS && _status != SPARSE_STATUS_NOT_SUPPORTED)
                {
                    std::cout << " Error in mkl_sparse_d_mv for U = - P^-1 * U: " << _status << std::endl;
                    return -1;
                }

                _R = this->_distances - _dmin;

                _status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., _A, _descrA, _U.data(), 1., _R.data()); // R = - A * U + R
                if (_status != SPARSE_STATUS_SUCCESS && _status != SPARSE_STATUS_NOT_SUPPORTED)
                {
                    printf(" Error in mkl_sparse_d_mv R = A U: %d \n", _status);
                    std::cout << " Error in mkl_sparse_d_mv for R = - A * U + R: " << _status << std::endl;
                    return -1;
                }

                _L = xt::maximum( _L-_rho*_R, 0);

                cmax = double((xt::amin(_R))(0));
                cc += 1;
                // std::cout << "-- C++ -- Projection : minimal constraint : " << cmax << std::endl;
            }

            if (cc>=_maxiter)
            {
                std::cout<<"\n-- C++ -- Projection : ********************** WARNING **********************"<<std::endl;
                std::cout<<  "-- C++ -- Projection : *************** Uzawa does not converge ***************"<<std::endl;
                std::cout<<  "-- C++ -- Projection : ********************** WARNING **********************\n"<<std::endl;
            }

            return cc;
        }

    template<std::size_t dim>
        auto OptimUzawaMkl<dim>::getUadapt_impl()
        {
            return xt::adapt(reinterpret_cast<double*>(_U.data()), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        auto OptimUzawaMkl<dim>::getWadapt_impl()
        {
            return xt::adapt(reinterpret_cast<double*>(_U.data()+3*this->_Nactive), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        void OptimUzawaMkl<dim>::allocateMemory_impl(const std::size_t nc)
        {
            std::ignore = nc;
        }

    template<std::size_t dim>
        void OptimUzawaMkl<dim>::freeMemory_impl()
        {
            mkl_sparse_destroy ( _invP );
            mkl_sparse_destroy ( _A );
            _A_coo_row.clear();
            _A_coo_col.clear();
            _A_coo_val.clear();
        }

    template<std::size_t dim>
        int OptimUzawaMkl<dim>::getNbActiveContacts_impl()
        {
            return xt::sum(xt::where(_L > 0., xt::ones_like(_L), xt::zeros_like(_L)))();
        }

    template<std::size_t dim>
        void OptimUzawaMkl<dim>::printCrsMatrix(const sparse_matrix_t A)
        {
            MKL_INT* csr_row_begin_ptr = NULL;
            MKL_INT* csr_row_end_ptr = NULL;
            MKL_INT* csr_col_ptr = NULL;
            double* csr_val_ptr = NULL;
            sparse_index_base_t indexing;
            MKL_INT nbRows;
            MKL_INT nbCols;
            _status = mkl_sparse_d_export_csr
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

            if (_status != SPARSE_STATUS_SUCCESS)
            {
                std::cout << " Error in mkl_sparse_d_export_csr: " << _status << std::endl;
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

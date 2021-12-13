#pragma once

#include "OptimBase.hpp"
#include "mkl_service.h"
#include "mkl_spblas.h"
#include <stdio.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>

namespace scopi{
    template<std::size_t dim>
        class OptimUzawaMatrixFree : public OptimBase<OptimUzawaMatrixFree<dim>, dim>
    {
        public:
            using base_type = OptimBase<OptimUzawaMatrixFree<dim>, dim>;

            OptimUzawaMatrixFree(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
            void createMatrixConstraint_impl(const std::vector<scopi::neighbor<dim>>& contacts);
            void createMatrixMass_impl();
            int solveOptimizationProblem_impl(const std::vector<scopi::neighbor<dim>>& contacts);
            auto getUadapt_impl();
            auto getWadapt_impl();
            void allocateMemory_impl(const std::size_t nc);
            void freeMemory_impl();
            int getNbActiveContacts_impl();

        private:
            void printCrsMatrix(const sparse_matrix_t);
            void gemv_invP();
            void gemv_A(const std::vector<scopi::neighbor<dim>>& contacts);
            void gemv_transposeA(const std::vector<scopi::neighbor<dim>>& contacts);

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
        OptimUzawaMatrixFree<dim>::OptimUzawaMatrixFree(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr) : 
            OptimBase<OptimUzawaMatrixFree<dim>, dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive, 0),
            _tol(1.0e-2), _maxiter(40000), _rho(200.), _dmin(0.),
            _U(xt::zeros<double>({6*Nactive}))
            {
            }

    template<std::size_t dim>
        void OptimUzawaMatrixFree<dim>::createMatrixConstraint_impl(const std::vector<scopi::neighbor<dim>>& contacts)
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
        void OptimUzawaMatrixFree<dim>::createMatrixMass_impl()
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
        int OptimUzawaMatrixFree<dim>::solveOptimizationProblem_impl(const std::vector<scopi::neighbor<dim>>& contacts)
        {
            _L = xt::zeros_like(this->_distances);
            _R = xt::zeros_like(this->_distances);

            std::size_t cc = 0;
            double cmax = -1000.0;
            while ( (cmax<=-_tol)&&(cc <= _maxiter) )
            {
                _U = this->_c;
                gemv_transposeA(contacts); // U = A^T * L + U
                gemv_invP();  // U = - P^-1 * U
                _R = this->_distances - _dmin;
                gemv_A(contacts); // R = - A * U + R
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
        auto OptimUzawaMatrixFree<dim>::getUadapt_impl()
        {
            return xt::adapt(reinterpret_cast<double*>(_U.data()), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        auto OptimUzawaMatrixFree<dim>::getWadapt_impl()
        {
            return xt::adapt(reinterpret_cast<double*>(_U.data()+3*this->_Nactive), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        void OptimUzawaMatrixFree<dim>::allocateMemory_impl(const std::size_t nc)
        {
            std::ignore = nc;
        }

    template<std::size_t dim>
        void OptimUzawaMatrixFree<dim>::freeMemory_impl()
        {
            mkl_sparse_destroy ( _invP );
            mkl_sparse_destroy ( _A );
            _A_coo_row.clear();
            _A_coo_col.clear();
            _A_coo_val.clear();
        }

    template<std::size_t dim>
        int OptimUzawaMatrixFree<dim>::getNbActiveContacts_impl()
        {
            return xt::sum(xt::where(_L > 0., xt::ones_like(_L), xt::zeros_like(_L)))();
        }

    template<std::size_t dim>
        void OptimUzawaMatrixFree<dim>::printCrsMatrix(const sparse_matrix_t A)
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

    template<std::size_t dim>
        void OptimUzawaMatrixFree<dim>::gemv_invP()
        {
            // for loops instead of xtensor functions to control exactly the parallelism
#pragma omp parallel for
            for(std::size_t i = 0; i < this->_Nactive; ++i)
            {
                for (std::size_t d=0; d<3; ++d)
                {
                    _U(3*i + d) /= (-1. * this->_mass); // TODO: add mass into particles
                    _U(3*this->_Nactive + 3*i + d) /= (-1. * this->_moment);
                }
            }
        }

    template<std::size_t dim>
        void OptimUzawaMatrixFree<dim>::gemv_A(const std::vector<scopi::neighbor<dim>>& contacts)
        {
#pragma omp parallel for
            for(std::size_t ic = 0; ic < contacts.size(); ++ic)
            {
                auto &c = contacts[ic];
                for (std::size_t d=0; d<3; ++d)
                {
                    if (c.i >= this->_active_ptr)
                    {
                        _R(ic) -= (-this->_dt*c.nij[d]) * _U((c.i - this->_active_ptr)*3 + d);
                    }
                    if (c.j >= this->_active_ptr)
                    {
                        _R(ic) -= (this->_dt*c.nij[d]) * _U((c.j - this->_active_ptr)*3 + d);
                    }
                }

                auto r_i = c.pi - this->_particles.pos()(c.i);
                auto r_j = c.pj - this->_particles.pos()(c.j);

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

                auto Ri = scopi::rotation_matrix<3>(this->_particles.q()(c.i));
                auto Rj = scopi::rotation_matrix<3>(this->_particles.q()(c.j));

                if (c.i >= this->_active_ptr)
                {
                    std::size_t ind_part = c.i - this->_active_ptr;
                    auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                    for (std::size_t ip=0; ip<3; ++ip)
                    {
                        _R(ic) -= (this->_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * _U(3*this->_Nactive + 3*ind_part + ip);
                    }
                }

                if (c.j >= this->_active_ptr)
                {
                    std::size_t ind_part = c.j - this->_active_ptr;
                    auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                    for (std::size_t ip=0; ip<3; ++ip)
                    {
                        _R(ic) -= (-this->_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * _U(3*this->_Nactive + 3*ind_part + ip);
                    }
                }

            }
        }

    template<std::size_t dim>
        void OptimUzawaMatrixFree<dim>::gemv_transposeA(const std::vector<scopi::neighbor<dim>>& contacts)
        {
// #pragma omp parallel for
            for(std::size_t ic = 0; ic < contacts.size(); ++ic)
            {
                auto &c = contacts[ic];

                for (std::size_t d=0; d<3; ++d)
                {
                    if (c.i >= this->_active_ptr)
                    {
                         _U((c.i - this->_active_ptr)*3 + d) += _L(ic) * (-this->_dt*c.nij[d]);
                    }
                    if (c.j >= this->_active_ptr)
                    {
                        _U((c.j - this->_active_ptr)*3 + d) += _L(ic) * (this->_dt*c.nij[d]);
                    }
                }

                auto r_i = c.pi - this->_particles.pos()(c.i);
                auto r_j = c.pj - this->_particles.pos()(c.j);

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

                auto Ri = scopi::rotation_matrix<3>(this->_particles.q()(c.i));
                auto Rj = scopi::rotation_matrix<3>(this->_particles.q()(c.j));

                if (c.i >= this->_active_ptr)
                {
                    std::size_t ind_part = c.i - this->_active_ptr;
                    auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                    for (std::size_t ip=0; ip<3; ++ip)
                    {
                        _U(3*this->_Nactive + 3*ind_part + ip) += _L(ic) * (this->_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                }

                if (c.j >= this->_active_ptr)
                {
                    std::size_t ind_part = c.j - this->_active_ptr;
                    auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                    for (std::size_t ip=0; ip<3; ++ip)
                    {
                        _U(3*this->_Nactive + 3*ind_part + ip) += _L(ic) * (-this->_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                }
            }
        }
}

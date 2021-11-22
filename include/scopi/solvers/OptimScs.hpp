#pragma once

#include "OptimBase.hpp"
#include <scs.h>

namespace scopi{
    template<std::size_t dim>
        class OptimScs: public OptimBase<OptimScs<dim>, dim>
    {
        public:
            using base_type = OptimBase<OptimScs<dim>, dim>;

            OptimScs(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
            ~OptimScs();
            void createMatrixConstraint_impl(const std::vector<scopi::neighbor<dim>>& contacts);
            void createMatrixMass_impl();
            int solveOptimizationProblem_impl(const std::vector<scopi::neighbor<dim>>& contacts);
            auto getUadapt_impl();
            auto getWadapt_impl();
            void allocateMemory_impl(const std::size_t nc);
            void freeMemory_impl();
            int getNbActiveContacts_impl();

        private:
            ScsMatrix _P;
            ScsMatrix _A;
            ScsData _d;
            ScsCone _k;
            ScsSolution _sol;
            ScsInfo _info;
            ScsSettings _stgs;
            OptimScs(const OptimScs &);
            OptimScs & operator=(const OptimScs &);
    };

    template<std::size_t dim>
        OptimScs<dim>::OptimScs(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr) : 
            OptimBase<OptimScs<dim>, dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive, 0)
    {
        _P.x = new scs_float[6*this->_Nactive];
        _P.i = new scs_int[6*this->_Nactive];
        _P.p = new scs_int[6*this->_Nactive+1];
        _sol.x = new double[6*this->_Nactive];
        _A.p = new scs_int[6*this->_Nactive+1];
        // default values not set
        // use values given by
        // https://www.cvxgrp.org/scs/api/settings.html#settings
        _stgs.normalize = 1;
        _stgs.scale = 0.1;
        _stgs.adaptive_scale = 1;
        _stgs.rho_x = 1e-6;
        _stgs.max_iters = 1e5;
        _stgs.eps_abs = 1e-4;
        _stgs.eps_rel = 1e-4;
        _stgs.eps_infeas = 1e-7;
        _stgs.alpha = 1.5;
        _stgs.time_limit_secs = 0.;
        _stgs.verbose = 1;
        _stgs.warm_start = 0;
        _stgs.acceleration_lookback = 0;
        _stgs.acceleration_interval = 1;
        _stgs.write_data_filename = NULL;
        _stgs.log_csv_filename = NULL;

    }

    template<std::size_t dim>
        OptimScs<dim>::~OptimScs()
        {
            delete[] _P.x;
            delete[] _P.i;
            delete[] _P.p;
        }

    template<std::size_t dim>
        void OptimScs<dim>::createMatrixConstraint_impl(const std::vector<scopi::neighbor<dim>>& contacts)
        {
            // COO storage to CSR storage is easy to write, e.g.
            // The CSC storage of A is the CSR storage of A^T
            // reverse the role of row and column pointers to have the transpose
            std::vector<int> coo_rows;
            std::vector<int> coo_cols;
            std::vector<double> coo_vals;
            this->createMatrixConstraintCoo(contacts, coo_rows, coo_cols, coo_vals, 0);

            std::vector<int> csc_row;
            std::vector<int> csc_col;
            std::vector<double> csc_val;
            this->cooToCsr(coo_cols, coo_rows, coo_vals, csc_col, csc_row, csc_val);

            for(std::size_t i = 0; i < csc_val.size(); ++i)
                _A.x[i] = csc_val[i];
            for(std::size_t i = 0; i < csc_row.size(); ++i)
                _A.i[i] = csc_row[i];
            for(std::size_t i = 0; i < csc_col.size(); ++i)
                _A.p[i] = csc_col[i];
            _A.m = contacts.size();
            _A.n = 6*this->_Nactive;
        }

    template<std::size_t dim>
        void OptimScs<dim>::createMatrixMass_impl()
        {
            std::vector<scs_int> col;
            std::vector<scs_int> row;
            std::vector<scs_float> val;
            row.reserve(6*this->_Nactive);
            col.reserve(6*this->_Nactive+1);
            val.reserve(6*this->_Nactive);

            for (std::size_t i=0; i<this->_Nactive; ++i)
            {
                for (std::size_t d=0; d<3; ++d)
                {
                    row.push_back(3*i + d);
                    col.push_back(3*i + d);
                    val.push_back(this->_mass); // TODO: add mass into particles
                }
            }
            for (std::size_t i=0; i<this->_Nactive; ++i)
            {
                for (std::size_t d=0; d<3; ++d)
                {
                    row.push_back(3*this->_Nactive + 3*i + d);
                    col.push_back(3*this->_Nactive + 3*i + d);
                    val.push_back(this->_moment);
                }
            }
            col.push_back(6*this->_Nactive);

            // TODO allocation in constructor
            // There is a segfault if the memory is allocated in the constructor
            for(std::size_t i = 0; i < val.size(); ++i)
                _P.x[i] = val[i];
            for(std::size_t i = 0; i < row.size(); ++i)
                _P.i[i] = row[i];
            for(std::size_t i = 0; i < col.size(); ++i)
                _P.p[i] = col[i];
            _P.m = 6*this->_Nactive;
            _P.n = 6*this->_Nactive;
        }

    template<std::size_t dim>
        int OptimScs<dim>::solveOptimizationProblem_impl(const std::vector<scopi::neighbor<dim>>& contacts)
        {
            _d.m = contacts.size();
            _d.n = 6*this->_Nactive;
            _d.A = &_A;
            _d.P = &_P;
            _d.b = this->_distances.data();
            _d.c = this->_c.data();

            _k.z = 0; // 0 linear equality constraints
            _k.l = contacts.size(); // s >= 0
            _k.bu = NULL; 
            _k.bl = NULL; 
            _k.bsize = 0;
            _k.q = NULL;
            _k.qsize = 0;
            _k.s = NULL;
            _k.ssize = 0;
            _k.ep = 0;
            _k.ed = 0;
            _k.p = NULL;
            _k.psize = 0;

            scs(&_d, &_k, &_stgs, &_sol, &_info);

            // if(info.iter == -1)
            //     std::abort();

            auto nbIter = _info.iter;
            return nbIter;
        }

    template<std::size_t dim>
        auto OptimScs<dim>::getUadapt_impl()
        {
            return xt::adapt(reinterpret_cast<double*>(_sol.x), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        auto OptimScs<dim>::getWadapt_impl()
        {
            return xt::adapt(reinterpret_cast<double*>(_sol.x+3*this->_Nactive), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        void OptimScs<dim>::allocateMemory_impl(const std::size_t nc)
        {
            _A.x = new scs_float[2*6*nc];
            _A.i = new scs_int[2*6*nc];
            _sol.y = new scs_float[nc];
            _sol.s = new scs_float[nc];
        }

    template<std::size_t dim>
        void OptimScs<dim>::freeMemory_impl()
        {
            // TODO check that the memory was indeed allocated before freeing it
            delete[] _A.x;
            delete[] _A.i;
            delete[] _sol.y;
            delete[] _sol.s;
        }

    template<std::size_t dim>
        int OptimScs<dim>::getNbActiveContacts_impl()
        {
            int nbActiveContacts = 0;
            for(std::size_t i = 0; i < this->_distances.size(); ++i)
            {
                if(_sol.y[i] > 0.)
                {
                    nbActiveContacts++;
                }
            }
            return nbActiveContacts;
        }

}

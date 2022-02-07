#pragma once

#ifdef SCOPI_USE_SCS
#include "OptimBase.hpp"
#include <scs.h>

namespace scopi{
    template<std::size_t dim>
    class OptimScs: public OptimBase<OptimScs<dim>, dim>
    {
    public:
        using base_type = OptimBase<OptimScs<dim>, dim>;

        OptimScs(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, double tol = 1e-7);
        ~OptimScs();
        void create_matrix_constraint_impl(const std::vector<neighbor<dim>>& contacts);
        void create_matrix_mass_impl();
        int solve_optimization_problem_impl(const std::vector<neighbor<dim>>& contacts);
        auto get_uadapt_impl();
        auto get_wadapt_impl();
        void allocate_memory_impl(const std::size_t nc);
        void free_memory_impl();
        int get_nb_active_contacts_impl();

    private:
        void coo_to_csr(std::vector<int> coo_rows, std::vector<int> coo_cols, std::vector<double> coo_vals, std::vector<int>& csr_rows, std::vector<int>& csr_cols, std::vector<double>& csr_vals);

        ScsMatrix m_P;
        ScsMatrix m_A;
        ScsData m_d;
        ScsCone m_k;
        ScsSolution m_sol;
        ScsInfo m_info;
        ScsSettings m_stgs;
        OptimScs(const OptimScs &);
        OptimScs & operator=(const OptimScs &);
    };

    template<std::size_t dim>
    OptimScs<dim>::OptimScs(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, double tol) 
    : OptimBase<OptimScs<dim>, dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive, 0)
    {
        m_P.x = new scs_float[6*this->m_Nactive];
        m_P.i = new scs_int[6*this->m_Nactive];
        m_P.p = new scs_int[6*this->m_Nactive+1];
        m_sol.x = new double[6*this->m_Nactive];
        m_A.p = new scs_int[6*this->m_Nactive+1];
        // default values not set
        // use values given by
        // https://www.cvxgrp.org/scs/api/settings.html#settings
        m_stgs.normalize = 1;
        m_stgs.scale = 0.1;
        m_stgs.adaptive_scale = 1;
        m_stgs.rho_x = 1e-6;
        m_stgs.max_iters = 1e5;
        m_stgs.eps_abs = tol;
        m_stgs.eps_rel = tol;
        m_stgs.eps_infeas = tol*1e-3;
        m_stgs.alpha = 1.5;
        m_stgs.time_limit_secs = 0.;
        m_stgs.verbose = 0;
        m_stgs.warm_start = 0;
        m_stgs.acceleration_lookback = 0;
        m_stgs.acceleration_interval = 1;
        m_stgs.write_data_filename = NULL;
        m_stgs.log_csv_filename = NULL;

    }

    template<std::size_t dim>
    OptimScs<dim>::~OptimScs()
    {
        delete[] m_P.x;
        delete[] m_P.i;
        delete[] m_P.p;
    }

    template<std::size_t dim>
    void OptimScs<dim>::create_matrix_constraint_impl(const std::vector<neighbor<dim>>& contacts)
    {
        // COO storage to CSR storage is easy to write, e.g.
        // The CSC storage of A is the CSR storage of A^T
        // reverse the role of row and column pointers to have the transpose
        std::vector<int> coo_rows;
        std::vector<int> coo_cols;
        std::vector<double> coo_vals;
        this->create_matrix_constraint_coo(contacts, coo_rows, coo_cols, coo_vals, 0);

        std::vector<int> csc_row;
        std::vector<int> csc_col;
        std::vector<double> csc_val;
        this->coo_to_csr(coo_cols, coo_rows, coo_vals, csc_col, csc_row, csc_val);

        for (std::size_t i = 0; i < csc_val.size(); ++i)
            m_A.x[i] = csc_val[i];
        for (std::size_t i = 0; i < csc_row.size(); ++i)
            m_A.i[i] = csc_row[i];
        for (std::size_t i = 0; i < csc_col.size(); ++i)
            m_A.p[i] = csc_col[i];
        m_A.m = contacts.size();
        m_A.n = 6*this->m_Nactive;
    }

    template<std::size_t dim>
    void OptimScs<dim>::create_matrix_mass_impl()
    {
        std::vector<scs_int> col;
        std::vector<scs_int> row;
        std::vector<scs_float> val;
        row.reserve(6*this->m_Nactive);
        col.reserve(6*this->m_Nactive+1);
        val.reserve(6*this->m_Nactive);

        for (std::size_t i = 0; i < this->m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                row.push_back(3*i + d);
                col.push_back(3*i + d);
                val.push_back(this->m_mass); // TODO: add mass into particles
            }
        }
        for (std::size_t i = 0; i < this->m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                row.push_back(3*this->m_Nactive + 3*i + d);
                col.push_back(3*this->m_Nactive + 3*i + d);
                val.push_back(this->m_moment);
            }
        }
        col.push_back(6*this->m_Nactive);

        // TODO allocation in constructor
        // There is a segfault if the memory is allocated in the constructor
        for (std::size_t i = 0; i < val.size(); ++i)
            m_P.x[i] = val[i];
        for (std::size_t i = 0; i < row.size(); ++i)
            m_P.i[i] = row[i];
        for (std::size_t i = 0; i < col.size(); ++i)
            m_P.p[i] = col[i];
        m_P.m = 6*this->m_Nactive;
        m_P.n = 6*this->m_Nactive;
    }

    template<std::size_t dim>
    int OptimScs<dim>::solve_optimization_problem_impl(const std::vector<neighbor<dim>>&)
    {
        m_d.m = this->m_distances.size();
        m_d.n = 6*this->m_Nactive;
        m_d.A = &m_A;
        m_d.P = &m_P;
        m_d.b = this->m_distances.data();
        m_d.c = this->m_c.data();

        m_k.z = 0; // 0 linear equality constraints
        m_k.l = this->m_distances.size(); // s >= 0
        m_k.bu = NULL; 
        m_k.bl = NULL; 
        m_k.bsize = 0;
        m_k.q = NULL;
        m_k.qsize = 0;
        m_k.s = NULL;
        m_k.ssize = 0;
        m_k.ep = 0;
        m_k.ed = 0;
        m_k.p = NULL;
        m_k.psize = 0;

        scs(&m_d, &m_k, &m_stgs, &m_sol, &m_info);

        return m_info.iter;
    }

    template<std::size_t dim>
    auto OptimScs<dim>::get_uadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_sol.x), {this->m_Nactive, 3UL});
    }

    template<std::size_t dim>
    auto OptimScs<dim>::get_wadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_sol.x+3*this->m_Nactive), {this->m_Nactive, 3UL});
    }

    template<std::size_t dim>
    void OptimScs<dim>::allocate_memory_impl(const std::size_t nc)
    {
        m_A.x = new scs_float[2*6*nc];
        m_A.i = new scs_int[2*6*nc];
        m_sol.y = new scs_float[nc];
        m_sol.s = new scs_float[nc];
    }

    template<std::size_t dim>
    void OptimScs<dim>::free_memory_impl()
    {
        // TODO check that the memory was indeed allocated before freeing it
        delete[] m_A.x;
        delete[] m_A.i;
        delete[] m_sol.y;
        delete[] m_sol.s;
    }

    template<std::size_t dim>
    int OptimScs<dim>::get_nb_active_contacts_impl()
    {
        int nb_active_contacts = 0;
        for(std::size_t i = 0; i < this->m_distances.size(); ++i)
        {
            if(m_sol.y[i] > 0.)
            {
                nb_active_contacts++;
            }
        }
        return nb_active_contacts;
    }

    template<std::size_t dim>
    void OptimScs<dim>::coo_to_csr(std::vector<int> coo_rows, std::vector<int> coo_cols, std::vector<double> coo_vals, std::vector<int>& csr_rows, std::vector<int>& csr_cols, std::vector<double>& csr_vals)
    {
        // https://www-users.cse.umn.edu/~saad/software/SPARSKIT/
        std::size_t nrow = 6*this->m_Nactive;
        std::size_t nnz = coo_vals.size();
        csr_rows.resize(nrow+1);
        std::fill(csr_rows.begin(), csr_rows.end(), 0);
        csr_cols.resize(nnz);
        csr_vals.resize(nnz);

        // determine row-lengths.
        for (std::size_t k = 0; k < nnz; ++k)
        {
            csr_rows[coo_rows[k]]++;
        }

        // starting position of each row..
        {
            int k = 0;
            for (std::size_t j = 0; j < nrow+1; ++j)
            {
                int k0 = csr_rows[j];
                csr_rows[j] = k;
                k += k0;
            }
        }

        // go through the structure  once more. Fill in output matrix.
        for (std::size_t k = 0; k < nnz; ++k)
        {
            int i = coo_rows[k];
            int j = coo_cols[k];
            double x = coo_vals[k];
            int iad = csr_rows[i];
            csr_vals[iad] = x;
            csr_cols[iad] = j;
            csr_rows[i] = iad+1;
        }

        // shift back iao
        for (std::size_t j = nrow; j >= 1; --j)
        {
            csr_rows[j] = csr_rows[j-1];
        }
        csr_rows[0] = 0;
    }
}
#endif

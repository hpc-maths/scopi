#pragma once

#ifdef SCOPI_USE_SCS
#include "OptimBase.hpp"
#include "MatrixOptimSolver.hpp"
#include <scs.h>

namespace scopi{
    template<std::size_t dim>
    class OptimScs: public OptimBase<OptimScs<dim>, dim>
                  , public MatrixOptimSolver<dim>
    {
    public:
        OptimScs(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, double tol = 1e-7);
        int solve_optimization_problem_impl(const std::vector<neighbor<dim>>& contacts);
        auto get_uadapt_impl();
        auto get_wadapt_impl();
        void setup_impl(const std::vector<neighbor<dim>>& contacts);
        int get_nb_active_contacts_impl();

    private:
        using base_type = OptimBase<OptimScs<dim>, dim>;
        void coo_to_csr(std::vector<int> coo_rows, std::vector<int> coo_cols, std::vector<double> coo_vals, std::vector<int>& csr_rows, std::vector<int>& csr_cols, std::vector<double>& csr_vals);

        ScsMatrix m_P;
        std::vector<scs_float> m_P_x;
        std::vector<scs_int> m_P_i;
        std::vector<scs_int> m_P_p;

        ScsMatrix m_A;
        std::vector<scs_float> m_A_x;
        std::vector<scs_int> m_A_i;
        std::vector<scs_int> m_A_p;

        ScsData m_d;
        ScsCone m_k;

        ScsSolution m_sol;
        std::vector<scs_float> m_sol_x;
        std::vector<scs_float> m_sol_y;
        std::vector<scs_float> m_sol_s;

        ScsInfo m_info;
        ScsSettings m_stgs;
        OptimScs(const OptimScs &);
        OptimScs & operator=(const OptimScs &);
    };

    template<std::size_t dim>
    OptimScs<dim>::OptimScs(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, double tol) 
    : OptimBase<OptimScs<dim>, dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive, 0)
    , MatrixOptimSolver<dim>(particles, dt, Nactive, active_ptr)
    , m_P_x(6*Nactive)
    , m_P_i(6*Nactive)
    , m_P_p(6*Nactive+1)
    , m_A_p(6*Nactive+1)
    , m_sol_x(6*Nactive)
    {
        std::size_t index = 0;
        for (std::size_t i = 0; i < base_type::m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                m_P_i[index] = 3*i + d;
                m_P_p[index] = 3*i + d;
                m_P_x[index] = this->m_mass; // TODO: add mass into particles
                index++;
            }
        }
        for (std::size_t i = 0; i < base_type::m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                m_P_i[index] = 3*base_type::m_Nactive + 3*i + d;
                m_P_p[index] = 3*base_type::m_Nactive + 3*i + d;
                m_P_x[index] = this->m_moment;
                index++;
            }
        }
        m_P_p[index] = 6*base_type::m_Nactive;

        m_P.x = m_P_x.data();
        m_P.i = m_P_i.data();
        m_P.p = m_P_p.data();
        m_P.m = 6*base_type::m_Nactive;
        m_P.n = 6*base_type::m_Nactive;

        scs_set_default_settings(&m_stgs);
        m_stgs.eps_abs = tol;
        m_stgs.eps_rel = tol;
        m_stgs.eps_infeas = tol*1e-3;
        m_stgs.verbose = 0;
    }

    template<std::size_t dim>
    void OptimScs<dim>::setup_impl(const std::vector<neighbor<dim>>& contacts)
    {
        this->create_matrix_constraint_coo(contacts, 0);

        // COO storage to CSR storage is easy to write
        // The CSC storage of A is the CSR storage of A^T
        // reverse the role of row and column pointers to have the transpose
        this->coo_to_csr(this->m_A_cols, this->m_A_rows, this->m_A_values, m_A_p, m_A_i, m_A_x);

        m_A.x = m_A_x.data();
        m_A.i = m_A_i.data();
        m_A.p = m_A_p.data();
        m_A.m = contacts.size();
        m_A.n = 6*base_type::m_Nactive;
    }

    template<std::size_t dim>
    int OptimScs<dim>::solve_optimization_problem_impl(const std::vector<neighbor<dim>>&)
    {
        m_d.m = this->m_distances.size();
        m_d.n = 6*base_type::m_Nactive;
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

        m_sol.x = m_sol_x.data();
        m_sol_y.resize(this->m_distances.size());
        m_sol.y = m_sol_y.data();
        m_sol_s.resize(this->m_distances.size());
        m_sol.s = m_sol_s.data();

        scs(&m_d, &m_k, &m_stgs, &m_sol, &m_info);

        return m_info.iter;
    }

    template<std::size_t dim>
    auto OptimScs<dim>::get_uadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_sol.x), {base_type::m_Nactive, 3UL});
    }

    template<std::size_t dim>
    auto OptimScs<dim>::get_wadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_sol.x+3*base_type::m_Nactive), {base_type::m_Nactive, 3UL});
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
        std::size_t nrow = 6*base_type::m_Nactive;
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

#pragma once

#ifdef SCOPI_USE_SCS
#include "OptimBase.hpp"
#include "MatrixOptimSolver.hpp"
#include <scs.h>

namespace scopi
{
    class OptimScs: public OptimBase<OptimScs>
                  , public MatrixOptimSolver
    {
    public:
        using base_type = OptimBase<OptimScs>;

        OptimScs(std::size_t nparts, double dt, double tol = 1e-7);

        template <std::size_t dim>
        int solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                            const std::vector<neighbor<dim>>& contacts);
        double* uadapt_data();
        double* wadapt_data();
        int get_nb_active_contacts_impl() const;

    private:
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
    int OptimScs::solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                 const std::vector<neighbor<dim>>& contacts)
    {
        this->create_matrix_constraint_coo(particles, contacts, 0);
        // COO storage to CSR storage is easy to write
        // The CSC storage of A is the CSR storage of A^T
        // reverse the role of row and column pointers to have the transpose
        this->coo_to_csr(this->m_A_cols, this->m_A_rows, this->m_A_values, m_A_p, m_A_i, m_A_x);
        m_A.x = m_A_x.data();
        m_A.i = m_A_i.data();
        m_A.p = m_A_p.data();
        m_A.m = contacts.size();
        m_A.n = 6*this->m_nparts;

        m_d.m = this->m_distances.size();
        m_d.n = 6*this->m_nparts;
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
}
#endif

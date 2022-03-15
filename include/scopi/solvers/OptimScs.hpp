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

        template <std::size_t dim>
        OptimScs(std::size_t nparts, double dt, const scopi_container<dim>& particles, double tol = 1e-7);

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
        tic();
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
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : SCS matrix = " << duration;

        tic();
        scs(&m_d, &m_k, &m_stgs, &m_sol, &m_info);
        duration = toc();
        PLOG_INFO << "----> CPUTIME : SCS solve = " << duration;

        return m_info.iter;
    }

    template<std::size_t dim>
    OptimScs::OptimScs(std::size_t nparts, double dt, const scopi_container<dim>& particles, double tol)
    : base_type(nparts, dt, 2*3*nparts, 0)
    , MatrixOptimSolver(nparts, dt)
    , m_P_x(6*nparts)
    , m_P_i(6*nparts)
    , m_P_p(6*nparts+1)
    , m_A_p(6*nparts+1)
    , m_sol_x(6*nparts)
    {
        auto active_offset = particles.nb_inactive();
        std::size_t index = 0;
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_P_i[index] = 3*i + d;
                m_P_p[index] = 3*i + d;
                m_P_x[index] = particles.m()(active_offset + i);
                index++;
            }
            for (std::size_t d = dim; d < 3; ++d)
            {
                m_P_i[index] = 3*i + d;
                m_P_p[index] = 3*i + d;
                m_P_x[index] = 0.;
                index++;
            }
        }
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_P_i[index] = 3*nparts + 3*i + d;
                m_P_p[index] = 3*nparts + 3*i + d;
                m_P_x[index] = particles.j()(active_offset + i);
                index++;
            }
            for (std::size_t d = dim; d < 3; ++d)
            {
                m_P_i[index] = 3*nparts + 3*i + d;
                m_P_p[index] = 3*nparts + 3*i + d;
                m_P_x[index] = 0.;
                index++;
            }
        }
        m_P_p[index] = 6*nparts;

        m_P.x = m_P_x.data();
        m_P.i = m_P_i.data();
        m_P.p = m_P_p.data();
        m_P.m = 6*nparts;
        m_P.n = 6*nparts;

        scs_set_default_settings(&m_stgs);
        m_stgs.eps_abs = tol;
        m_stgs.eps_rel = tol;
        m_stgs.eps_infeas = tol*1e-3;
        m_stgs.verbose = 0;
    }

}
#endif

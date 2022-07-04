#pragma once

#ifdef SCOPI_USE_SCS
#include "OptimBase.hpp"
#include "../problems/DryWithoutFriction.hpp"
#include <scs.h>

namespace scopi
{
    template<class problem_t>
    class OptimScs;

    template<class problem_t>
    struct OptimParams<OptimScs<problem_t>>
    {
        OptimParams();
        OptimParams(const OptimParams<OptimScs<problem_t>>& params);

        ProblemParams<problem_t> problem_params;
        double tol;
        double tol_infeas;
    };

    template <class problem_t = DryWithoutFriction>
    class OptimScs: public OptimBase<OptimScs<problem_t>, problem_t>
    {
    protected:
        using problem_type = problem_t; 
    private:
        using base_type = OptimBase<OptimScs<problem_t>, problem_t>;

    protected:
        template <std::size_t dim>
        OptimScs(std::size_t nparts, double dt, const scopi_container<dim>& particles, const OptimParams<OptimScs>& optim_params);

    public:
        template <std::size_t dim>
        int solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                            const std::vector<neighbor<dim>>& contacts, 
                                            const std::vector<neighbor<dim>>& contacts_worms);
        double* uadapt_data();
        double* wadapt_data();
        double* lagrange_multiplier_data();
        double* constraint_data();
        int get_nb_active_contacts_impl() const;

    private:
        void coo_to_csr(std::vector<int> coo_rows, std::vector<int> coo_cols, std::vector<double> coo_vals, std::vector<int>& csr_rows, std::vector<int>& csr_cols, std::vector<double>& csr_vals);

        void set_moment_matrix(std::size_t nparts, const scopi_container<2>& particles, std::size_t& index);
        void set_moment_matrix(std::size_t nparts, const scopi_container<3>& particles, std::size_t& index);
        

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

    template <class problem_t>
    template<std::size_t dim>
    int OptimScs<problem_t>::solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                             const std::vector<neighbor<dim>>& contacts,
                                                             const std::vector<neighbor<dim>>& contacts_worms)
    {
        tic();
        this->create_matrix_constraint_coo(particles, contacts, contacts_worms, 0);
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

    template <class problem_t>
    template<std::size_t dim>
    OptimScs<problem_t>::OptimScs(std::size_t nparts, double dt, const scopi_container<dim>& particles, const OptimParams<OptimScs>& optim_params)
    : base_type(nparts, dt, 2*3*nparts, 0, optim_params)
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
        set_moment_matrix(nparts, particles, index);
        m_P_p[index] = 6*nparts;

        m_P.x = m_P_x.data();
        m_P.i = m_P_i.data();
        m_P.p = m_P_p.data();
        m_P.m = 6*nparts;
        m_P.n = 6*nparts;

        scs_set_default_settings(&m_stgs);
        m_stgs.eps_abs = this->m_params.tol;
        m_stgs.eps_rel = this->m_params.tol;
        m_stgs.eps_infeas = this->m_params.tol_infeas;
        m_stgs.verbose = 0;
    }

    template <class problem_t>
    double* OptimScs<problem_t>::uadapt_data()
    {
        return m_sol.x;
    }

    template <class problem_t>
    double* OptimScs<problem_t>::wadapt_data()
    {
        return m_sol.x + 3*this->m_nparts;
    }

    template <class problem_t>
    double* OptimScs<problem_t>::lagrange_multiplier_data()
    {
        return m_sol.y;
    }

    template<class problem_t>
    double* OptimScs<problem_t>::constraint_data()
    {
        return NULL;
    }

    template <class problem_t>
    int OptimScs<problem_t>::get_nb_active_contacts_impl() const
    {
        int nb_active_contacts = 0;
        for(int i = 0; i < m_k.l; ++i)
        {
            if(m_sol.y[i] > 0.)
            {
                nb_active_contacts++;
            }
        }
        return nb_active_contacts;
    }

    template <class problem_t>
    void OptimScs<problem_t>::coo_to_csr(std::vector<int> coo_rows, std::vector<int> coo_cols, std::vector<double> coo_vals,
                              std::vector<int>& csr_rows, std::vector<int>& csr_cols, std::vector<double>& csr_vals)
    {
        // https://www-users.cse.umn.edu/~saad/software/SPARSKIT/
        std::size_t nrow = 6*this->m_nparts;
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

    template <class problem_t>
    void OptimScs<problem_t>::set_moment_matrix(std::size_t nparts, const scopi_container<2>& particles, std::size_t& index)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 2; ++d)
            {
                m_P_i[index] = 3*nparts + 3*i + d;
                m_P_p[index] = 3*nparts + 3*i + d;
                m_P_x[index] = 0.;
                index++;
            }
            m_P_i[index] = 3*nparts + 3*i + 2;
            m_P_p[index] = 3*nparts + 3*i + 2;
            m_P_x[index] = particles.j()(active_offset + i);
            index++;
        }
    }

    template <class problem_t>
    void OptimScs<problem_t>::set_moment_matrix(std::size_t nparts, const scopi_container<3>& particles, std::size_t& index)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 2; ++d)
            {
                m_P_i[index] = 3*nparts + 3*i + d;
                m_P_p[index] = 3*nparts + 3*i + d;
                m_P_x[index] = particles.j()(active_offset + i)[d];
                index++;
            }
        }
    }

    template<class problem_t>
    OptimParams<OptimScs<problem_t>>::OptimParams()
    : problem_params()
    , tol(1e-7)
    , tol_infeas(1e-10)
    {}

    template<class problem_t>
    OptimParams<OptimScs<problem_t>>::OptimParams(const OptimParams<OptimScs<problem_t>>& params)
    : problem_params(params.problem_params)
    , tol(params.tol)
    , tol_infeas(params.tol_infeas)
    {}
}
#endif

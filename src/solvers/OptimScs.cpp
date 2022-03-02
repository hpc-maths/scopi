#include "scopi/solvers/OptimScs.hpp"

#ifdef SCOPI_USE_SCS

namespace scopi
{
    OptimScs::OptimScs(std::size_t nparts, double dt, double tol)
    : base_type(nparts, dt, 2*3*nparts, 0)
    , MatrixOptimSolver(nparts, dt)
    , m_P_x(6*nparts)
    , m_P_i(6*nparts)
    , m_P_p(6*nparts+1)
    , m_A_p(6*nparts+1)
    , m_sol_x(6*nparts)
    {
        std::size_t index = 0;
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                m_P_i[index] = 3*i + d;
                m_P_p[index] = 3*i + d;
                m_P_x[index] = this->m_mass; // TODO: add mass into particles
                index++;
            }
        }
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                m_P_i[index] = 3*nparts + 3*i + d;
                m_P_p[index] = 3*nparts + 3*i + d;
                m_P_x[index] = this->m_moment;
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

    double* OptimScs::uadapt_data()
    {
        return m_sol.x;
    }

    double* OptimScs::wadapt_data()
    {
        return m_sol.x + 3*this->m_nparts;
    }

    int OptimScs::get_nb_active_contacts_impl() const
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

    void OptimScs::coo_to_csr(std::vector<int> coo_rows, std::vector<int> coo_cols, std::vector<double> coo_vals,
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
}
#endif

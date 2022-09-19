#pragma once

#ifdef SCOPI_USE_SCS
#include <scs.h>

#include "OptimBase.hpp"
#include "../problems/DryWithoutFriction.hpp"

namespace scopi
{
    template<class problem_t>
    class OptimScs;

    /**
     * @class OptimParams>
     * @brief Parameters for \c OptimScs<problem_t>
     *
     * Specialization of ProblemParams.
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t>
    struct OptimParams<OptimScs<problem_t>>
    {
        /**
         * @brief Default constructor.
         */
        OptimParams();
        /**
         * @brief Copy constructor.
         *
         * @param params Parameters to by copied.
         */
        OptimParams(const OptimParams<OptimScs<problem_t>>& params);

        /**
         * @brief Tolerance of the solver.
         *
         * Default value is \f$ 10^{-7} \f$.
         * \note \c tol > 0
         */
        double tol;
        /**
         * @brief Infeasible convergence tolerance.
         *
         * Default value is \f$ 10^{-10} \f$
         * \note \c tol_infeas > 0
         */
        double tol_infeas;
    };

    /**
     * @class OptimScs
     * @brief Solve optimization problem using Mosek.
     *
     * See ProblemBase for the notations.
     *
     * The documentation of SCS is available here: https://www.cvxgrp.org/scs/.
     * Matrices are stored using CSC storage.
     * \warning Only the cases <tt> problem_t = DryWithoutFriction </tt> and <tt> problem_t = ViscousWithoutFriction </tt> are implemented.
     *
     * @tparam problem_t Problem to be solved.
     */
    template <class problem_t = DryWithoutFriction>
    class OptimScs: public OptimBase<OptimScs<problem_t>, problem_t>
    {
    protected:
        /**
         * @brief Alias for the problem.
         */
        using problem_type = problem_t; 
    private:
        /**
         * @brief Alias for the base class \c OptimBase
         */
        using base_type = OptimBase<OptimScs<problem_t>, problem_t>;

    protected:
        /**
         * @brief Constructor.
         *
         * Build the matrix \f$ \mathbb{P} \f$ with SCS' data structure.
         *
         * @tparam dim Dimension (2 or 3).
         * @param nparts [in] Number of particles.
         * @param dt [in] Time step.
         * @param particles [in] Array of particles.
         * @param optim_params [in] Parameters.
         * @param problem_params [in] Parameters for the problem.
         */
        template <std::size_t dim>
        OptimScs(std::size_t nparts,
                 double dt,
                 const scopi_container<dim>& particles,
                 const OptimParams<OptimScs<problem_t>>& optim_params,
                 const ProblemParams<problem_t>& problem_params);

    public:
        /**
         * @brief Solve the optimization problem.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         *
         * @return Number of iterations SCS' solver needed to converge.
         */
        template <std::size_t dim>
        int solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                            const std::vector<neighbor<dim>>& contacts, 
                                            const std::vector<neighbor<dim>>& contacts_worms);
        /**
         * @brief \f$ \mathbf{u} \in \mathbb{R}^{6N} \f$ contains the velocities and the rotations of the particles, the function returns the velocities solution of the optimization problem..
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ 3 N \f$ elements.
         */
        double* uadapt_data();
        /**
         * @brief \f$ \mathbf{u} \in \mathbb{R}^{6N} \f$ contains the velocities and the rotations of the particles, the function returns the rotations solution of the optimization problem..
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ 3 N \f$ elements.
         */
        double* wadapt_data();
        /**
         * @brief Returns the Lagrange multipliers (solution of the dual problem) when the optimization is solved.
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ N_c \f$ elements.
         */
        double* lagrange_multiplier_data();
        /**
         * @brief Returns \f$ \mathbf{d} + \mathbb{B} \mathbf{u} \f$, where \f$ \mathbf{u} \f$ is the solution of the optimization problem.
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         * \warning The method is not implemented, it is defined so all solvers have the same interface.
         *
         * @return Null pointer instead of \f$ N_c \f$ elements.
         */
        double* constraint_data();
        /**
         * @brief Number of Lagrange multipliers > 0 (active constraints).
         */
        int get_nb_active_contacts_impl() const;

    private:
        /**
         * @brief Convert a matrix stored in COO format to CSR format.
         *
         * This function is used knowing that
         *  - the CSC storage of a matrix is the CSR storage of its transpose;
         *  - the COO storage of the transpose of a matrix is obtained by inversing the row and columns array of the matrix in COO storage.
         *
         * @param coo_rows [in] Rows' indicies of to COO storage.
         * @param coo_cols [in] Column's indicies of the COO storage.
         * @param coo_vals [in] Values of the COO storage.
         * @param csr_rows [out] Rows' indicies of the CSR storage.
         * @param csr_cols [out] Column's indicies of the CSR storage.
         * @param csr_vals [out] Values of the CSR storage.
         */
        void coo_to_csr(std::vector<int> coo_rows, std::vector<int> coo_cols, std::vector<double> coo_vals, std::vector<int>& csr_rows, std::vector<int>& csr_cols, std::vector<double>& csr_vals);

        /**
         * @brief 2D implementation to set the moments of inertia in the matrix \f$ \mathbb{P} \f$.
         *
         * The matrix \f$ \mathbb{P} \f$ is diagonale and \f$ \mathbb{P} = diag(m_0, m_0, 0, \dots, m_{N}, m_{N}, 0, 0, 0, J_0, \dots, 0, 0, J_{N}) \f$,
         * where \f$ m_i \f$ (resp. \f$ J_i \f$) is the mass (resp. moment of inertia) of the particle \f$ i \f$.
         * This function set the second part of the matrix.
         *
         * @param nparts [in] Number of particles.
         * @param particles [in] Array of particles (for the moments of inertia).
         * @param index [in] Index of the first row with moments.
         */
        void set_moment_matrix(std::size_t nparts, const scopi_container<2>& particles, std::size_t& index);
        /**
         * @brief 3D implementation to set the moments of inertia in the matrix \f$ \mathbb{P} \f$.
         *
         * The matrix \f$ \mathbb{P} \f$ is diagonale and \f$ \mathbb{P} = diag(m_0, m_0, m_0, \dots, m_{N}, m_{N}, m_{N}, J_0^x, J_0^y, J_0^z, \dots, J_{N}^x, J_{N}^y, J_{N}^z) \f$,
         * where \f$ m_i \f$ (resp. \f$ \mathbf{J}_i = (J_i^x, J_i^y, J_i^z) \f$) is the mass (resp. moment of inertia) of the particle \f$ i \f$.
         * This function set the second part of the matrix.
         *
         * @param nparts [in] Number of particles.
         * @param particles [in] Array of particles (for the moments of inertia).
         * @param index [in] Index of the first row with moments.
         */
        void set_moment_matrix(std::size_t nparts, const scopi_container<3>& particles, std::size_t& index);
        

        /**
         * @brief SCS' data structure for the matrix \f$ \mathbb{P} \f$.
         */
        ScsMatrix m_P;
        /**
         * @brief Values of \f$ \mathbb{P} \f$ in CSC storage.
         */
        std::vector<scs_float> m_P_x;
        /**
         * @brief Row indices of \f$ \mathbb{P} \f$ in CSC storage.
         */
        std::vector<scs_int> m_P_i;
        /**
         * @brief Column indices of \f$ \mathbb{P} \f$ in CSC storage.
         */
        std::vector<scs_int> m_P_p;

        /**
         * @brief SCS' data structure for the matrix \f$ \mathbb{B} \f$.
         */
        ScsMatrix m_A;
        /**
         * @brief Values of \f$ \mathbb{B} \f$ in CSC storage.
         */
        std::vector<scs_float> m_A_x;
        /**
         * @brief Row indices of \f$ \mathbb{B} \f$ in CSC storage.
         */
        std::vector<scs_int> m_A_i;
        /**
         * @brief Column indices of \f$ \mathbb{B} \f$ in CSC storage.
         */
        std::vector<scs_int> m_A_p;

        /**
         * @brief SCS' data structure for \f$ \mathbf{d} \f$.
         */
        ScsData m_d;
        /**
         * @brief SCS' data structure for the constraints.
         */
        ScsCone m_k;

        /**
         * @brief SCS' data structure for the solution of the optimization problem.
         */
        ScsSolution m_sol;
        /**
         * @brief Solution of the optimization problem.
         */
        std::vector<scs_float> m_sol_x;
        /**
         * @brief Solution of the dual problem.
         */
        std::vector<scs_float> m_sol_y;
        /**
         * @brief Slack variable.
         */
        std::vector<scs_float> m_sol_s;

        /**
         * @brief Contains information about the solve run at termination.
         */
        ScsInfo m_info;
        /**
         * @brief Struct containing all settings. 
         */
        ScsSettings m_stgs;
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
    OptimScs<problem_t>::OptimScs(std::size_t nparts,
                                  double dt,
                                  const scopi_container<dim>& particles,
                                  const OptimParams<OptimScs<problem_t>>& optim_params,
                                  const ProblemParams<problem_t>& problem_params)
    : base_type(nparts, dt, 2*3*nparts, 0, optim_params, problem_params)
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
    void OptimScs<problem_t>::coo_to_csr(std::vector<int> coo_rows,   std::vector<int> coo_cols,  std::vector<double> coo_vals,
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
    : tol(1e-7)
    , tol_infeas(1e-10)
    {}

    template<class problem_t>
    OptimParams<OptimScs<problem_t>>::OptimParams(const OptimParams<OptimScs<problem_t>>& params)
    : tol(params.tol)
    , tol_infeas(params.tol_infeas)
    {}
}
#endif

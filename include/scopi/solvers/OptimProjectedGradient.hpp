#pragma once

#ifdef SCOPI_USE_MKL
#include <mkl_spblas.h>

#include "OptimBase.hpp"

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../problems/DryWithoutFriction.hpp"
#include "gradient/pgd.hpp"

namespace scopi{
    template<class problem_t, template <class> class gradient_t>
    class OptimProjectedGradient;

    /**
     * @brief Parameters for \c OptimProjectedGradient<problem_t, gradient_t>
     *
     * Specialization of ProblemParams in params.hpp
     *
     * @tparam problem_t Problem to be solved.
     * @tparam gradient_t Gradient descent algorithm.
     */
    template<class problem_t, template <class> class gradient_t>
    class OptimParams<OptimProjectedGradient<problem_t, gradient_t>>
    {
    public:
        /**
         * @brief Default constructor.
         */
        OptimParams();
        /**
         * @brief Copy constructor.
         *
         * @param params Parameters to by copied.
         */
        OptimParams(const OptimParams<OptimProjectedGradient<problem_t, gradient_t>>& params);

        /**
         * @brief Tolerance for \f$ \mathbf{dg} \f$ criterion.
         *
         * Default value is \f$ 10^{-9} \f$.
         * \note \c tol_dg > 0
         */
        double tol_dg;
        /**
         * @brief Tolerance for \f$ \l \f$ criterion.
         *
         * Default value is \f$ 10^{-9} \f$.
         * \note \c tol_l > 0
         */
        double tol_l;
        /**
         * @brief Maximal number of iterations.
         *
         * Default value is 40000.
         */
        std::size_t max_iter;
        /**
         * @brief Step for the gradient descent.
         *
         * Default value is 2000.
         * \note \c rho > 0
         */
        double rho;
        /**
         * @brief Whether to compute and print the function cost.
         *
         * Sould be used with <tt> PLOG_VERBOSE </tt>.
         */
        bool verbose;
    };

    /**
     * @brief Prints a matrix on standard output.
     *
     * @param A [in] Matrix to print.
     */
    void print_csr_matrix(const sparse_matrix_t A);

    /**
     * @brief Solve the optimization problem with gradients-like algorithms.
     *
     * See ProblemBase for the notations.
     * The implemented algorithm is:
     *  - \f$ \A = \mathbb{B}^T \mathbb{P}^{-1} \mathbb{B} \f$;
     *  - \f$ \l = \text{ gradient algorithm } \left( \A, \mathbf{d} - \mathbb{B} \u \right) \f$;
     *  - \f$ \u = \mathbb{P}^{-1} \left( \c - \mathbb{B}^T \l \right) \f$.
     *
     *  The gradient algorithm is given by \c gradient_t.
     *
     *  All matrices and matrix-vector products use the MKL.
     *
     * @tparam problem_t Problem to be solved.
     * @tparam gradient_t Gradient algorithm.
     */
    template<class problem_t = DryWithoutFriction, template <class> class gradient_t = pgd>
    class OptimProjectedGradient: public OptimBase<OptimProjectedGradient<problem_t, gradient_t>, problem_t>
                                , public gradient_t<problem_t>
    {
    public:
        /**
         * @brief Alias for the problem.
         */
        using problem_type = problem_t; 
    private:
        /**
         * @brief Alias for the base class \c OptimBase.
         */
        using base_type = OptimBase<OptimProjectedGradient<problem_t, gradient_t>, problem_t>;

    protected:
        /**
         * @brief Constructor.
         *
         * Build the matrix \f$ \AzMosek \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param nparts [in] Number of particles.
         * @param dt [in] Time step.
         * @param particles [in] Array of particles.
         * @param optim_params [in] Parameters.
         * @param problem_params [in] Parameters for the problem.
         */
        template<std::size_t dim>
        OptimProjectedGradient(std::size_t nparts,
                               double dt,
                               const scopi_container<dim>& particles,
                               const OptimParams<OptimProjectedGradient<problem_t, gradient_t>>& optim_params,
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
         * @return Number of iterations gradient descent algorithm needed to converge.
         */
        template <std::size_t dim>
        std::size_t solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                    const std::vector<neighbor<dim>>& contacts,
                                                    const std::vector<neighbor<dim>>& contacts_worms);
        /**
         * @brief \f$ \u \in \mathbb{R}^{6N} \f$ contains the velocities and the rotations of the particles, the function returns the velocities solution of the optimization problem..
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ 3 N \f$ elements.
         */
        auto uadapt_data();
        /**
         * @brief \f$ \u \in \mathbb{R}^{6N} \f$ contains the velocities and the rotations of the particles, the function returns the rotations solution of the optimization problem..
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ 3 N \f$ elements.
         */
        auto wadapt_data();
        /**
         * @brief Returns the Lagrange multipliers (solution of the dual problem) when the optimization is solved.
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ N_c \f$ elements.
         */
        auto lagrange_multiplier_data();
        /**
         * @brief Returns \f$ \mathbf{d} + \mathbb{B} \u \f$, where \f$ \u \f$ is the solution of the optimization problem.
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ N_c \f$ elements.
         */
        double* constraint_data();
        /**
         * @brief Number of Lagrange multipliers > 0 (active constraints).
         */
        int get_nb_active_contacts_impl() const;

    private:
        /**
         * @brief 2D implementation to set the moments of inertia in the matrix \f$ \mathbb{P}^{-1} \f$.
         *
         * @param nparts [in] Number of particles.
         * @param invP_csr_row [out] Rows' indicies of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param invP_csr_col [out] Columns' indicies of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param invP_csr_val [out] Values of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param particles [in] Array for particles (for moments of inertia).
         */
        void set_moment_matrix(std::size_t nparts,
                               std::vector<MKL_INT>& invP_csr_row,
                               std::vector<MKL_INT>& invP_csr_col,
                               std::vector<double>& invP_csr_val,
                               const scopi_container<2>& particles);
        /**
         * @brief 3D implementation to set the moments of inertia in the matrix \f$ \mathbb{P}^{-1} \f$.
         *
         * @param nparts [in] Number of particles.
         * @param invP_csr_row [out] Rows' indicies of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param invP_csr_col [out] Columns' indicies of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param invP_csr_val [out] Values of the matrix \f$ \mathbb{P}^{-1} \f$.
         * @param particles [in] Array for particles (for moments of inertia).
         */
        void set_moment_matrix(std::size_t nparts,
                               std::vector<MKL_INT>& invP_csr_row,
                               std::vector<MKL_INT>& invP_csr_col,
                               std::vector<double>& invP_csr_val,
                               const scopi_container<3>& particles);
        /**
         * @brief Build the matrix \f$ \mathbb{B} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         */
        template <std::size_t dim>
        void create_matrix_B(const scopi_container<dim>& particles,
                             const std::vector<neighbor<dim>>& contacts,
                             const std::vector<neighbor<dim>>& contacts_worms);
        /**
         * @brief Build matrix \f$ \A = \mathbb{B}^T \mathbb{P}^{-1} \mathbb{B} \f$.
         */
        void create_matrix_A();

        /**
         * @brief Vector \f$ \l \f$.
         */
        xt::xtensor<double, 1> m_l;
        /**
         * @brief Vector \f$ \e = \mathbf{d} - \mathbb{B} \u \f$.
         */
        xt::xtensor<double, 1> m_e; // vector c in 220517_PbDual_MiniForces.pdf
        /**
         * @brief Vector \f$ \u \f$.
         */
        xt::xtensor<double, 1> m_u;
        /**
         * @brief Vecotr \f$ \mathbb{B} \l \f$.
         */
        xt::xtensor<double, 1> m_bl;

        /**
         * @brief Matrix \f$ \A \f$.
         */
        sparse_matrix_t m_A;
        /**
         * @brief Structure specifying \f$ \A \f$ properties. 
         */
        struct matrix_descr m_descrA;
        /**
         * @brief Matrix \f$ \mathbb{B} \f$.
         */
        sparse_matrix_t m_B;
        /**
         * @brief Structure specifying \f$ \mathbb{B} \f$ properties. 
         */
        struct matrix_descr m_descrB;
        /**
         * @brief Matrix \f$ \mathbb{P}^{-1} \f$.
         */
        sparse_matrix_t m_inv_P;
        /**
         * @brief Structure specifying \f$ \mathbb{P}^{-1} \f$ properties. 
         */
        struct matrix_descr m_descr_inv_P;
        /**
         * @brief Value indicating whether the operation was successful or not, and why.
         */
        sparse_status_t m_status;
    };

    template<class problem_t, template <class> class gradient_t>
    template<std::size_t dim>
    OptimProjectedGradient<problem_t, gradient_t>::OptimProjectedGradient(std::size_t nparts,
                                                                          double dt,
                                                                          const scopi_container<dim>& particles,
                                                                          const OptimParams<OptimProjectedGradient<problem_t, gradient_t>>& optim_params,
                                                                          const ProblemParams<problem_t>& problem_params)
    : base_type(nparts, dt, 2*3*nparts, 0, optim_params, problem_params)
    , gradient_t<problem_t>(optim_params.max_iter, optim_params.rho, optim_params.tol_dg, optim_params.tol_l, optim_params.verbose)
    , m_u(xt::zeros<double>({6*nparts}))
    , m_bl(xt::zeros<double>({6*nparts}))
    {
        std::vector<MKL_INT> invP_csr_row;
        std::vector<MKL_INT> invP_csr_col;
        std::vector<double> invP_csr_val;
        invP_csr_col.reserve(6*nparts);
        invP_csr_row.reserve(6*nparts+1);
        invP_csr_val.reserve(6*nparts);

        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*i + d);
                invP_csr_col.push_back(3*i + d);
                invP_csr_val.push_back(1./particles.m()(active_offset + i));
            }
        }
        set_moment_matrix(nparts, invP_csr_row, invP_csr_col, invP_csr_val, particles);
        invP_csr_row.push_back(6*nparts);

        m_status = mkl_sparse_d_create_csr(&m_inv_P,
                                           SPARSE_INDEX_BASE_ZERO,
                                           6*nparts, // number of rows
                                           6*nparts, // number of cols
                                           invP_csr_row.data(),
                                           invP_csr_row.data()+1,
                                           invP_csr_col.data(),
                                           invP_csr_val.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_create_csr for matrix invP: " << m_status;

        m_descr_inv_P.type = SPARSE_MATRIX_TYPE_DIAGONAL;
        m_descr_inv_P.diag = SPARSE_DIAG_NON_UNIT;

        m_status = mkl_sparse_set_mv_hint(m_inv_P, SPARSE_OPERATION_NON_TRANSPOSE, m_descr_inv_P, 2 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix invP: " << m_status;
        m_status = mkl_sparse_optimize ( m_inv_P );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix invP: " << m_status;
    }

    template<class problem_t, template<class> class gradient_t>
    template <std::size_t dim>
    std::size_t OptimProjectedGradient<problem_t, gradient_t>::solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                                                               const std::vector<neighbor<dim>>& contacts,
                                                                                               const std::vector<neighbor<dim>>& contacts_worms)
    {
        tic();
        xt::noalias(m_l) = xt::zeros<double>({this->number_row_matrix(contacts, contacts_worms)});
        // u = P^{-1}*c = vap
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_inv_P, m_descr_inv_P, this->m_c.data(), 0., m_u.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for u = P^{-1}*c: " << m_status;
        double time_vector_operations = toc();

        create_matrix_B(particles, contacts, contacts_worms);
        tic();
        create_matrix_A();
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : projected gradient : A = B^T*M^-1*B = " << duration;

        // e = -B*u+distances
        tic();
        xt::noalias(m_e) = this->m_distances;
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_B, m_descrB, m_u.data(), 1., m_e.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for e = B*u+d: " << m_status;
        time_vector_operations += toc();

        tic();
        std::size_t nb_iter = this->projection(m_A, m_descrA, m_e, m_l);
        duration = toc();
        PLOG_INFO << "----> CPUTIME : projected gradient : projection = " << duration;

        // u = u - P^{-1}*B^T*l
        tic();
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE, 1., m_B, m_descrB, m_l.data(), 0., m_bl.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for bl = B^T*l: " << m_status;
        m_status = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, -1., m_inv_P, m_descr_inv_P, m_bl.data(), 1., m_u.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_mv for u = vap - P^{-1}*bl: " << m_status;

        m_status = mkl_sparse_destroy ( m_B );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix B: " << m_status;
        m_status = mkl_sparse_destroy ( m_A );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix A: " << m_status;
        time_vector_operations += toc();
        PLOG_INFO << "----> CPUTIME : projected gradient : vectors operations = " << time_vector_operations;

        return nb_iter;
    }

    template<class problem_t, template<class> class gradient_t>
    auto OptimProjectedGradient<problem_t, gradient_t>::uadapt_data()
    {
        return m_u.data();
    }

    template<class problem_t, template<class> class gradient_t>
    auto OptimProjectedGradient<problem_t, gradient_t>::wadapt_data()
    {
        return m_u.data() + 3*this->m_nparts;
    }

    template<class problem_t, template<class> class gradient_t>
    auto OptimProjectedGradient<problem_t, gradient_t>::lagrange_multiplier_data()
    {
        return m_l.data();
    }

    template<class problem_t, template<class> class gradient_t>
    double* OptimProjectedGradient<problem_t, gradient_t>::constraint_data()
    {
        return NULL;
    }

    template<class problem_t, template<class> class gradient_t>
    int OptimProjectedGradient<problem_t, gradient_t>::get_nb_active_contacts_impl() const
    {
        return xt::sum(xt::where(m_l > 0., xt::ones_like(m_l), xt::zeros_like(m_l)))();
    }

    template <class problem_t, template<class> class gradient_t>
    template <std::size_t dim>
    void OptimProjectedGradient<problem_t, gradient_t>::create_matrix_B(const scopi_container<dim>& particles,
                                                                        const std::vector<neighbor<dim>>& contacts,
                                                                        const std::vector<neighbor<dim>>& contacts_worms)
    {
        m_descrB.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t B_coo;

        tic();
        this->create_matrix_constraint_coo(particles, contacts, contacts_worms, 0UL);
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : projected gradient : create_matrix_B : create_matrix_constraint_coo = " << duration;

        tic();
        m_status =  mkl_sparse_d_create_coo(&B_coo,
                                           SPARSE_INDEX_BASE_ZERO,
                                           this->number_row_matrix(contacts, contacts_worms), // number of rows
                                           6*this->m_nparts, // number of cols
                                           this->m_A_values.size(), // number of non-zero elements
                                           this->m_A_rows.data(),
                                           this->m_A_cols.data(),
                                           this->m_A_values.data());
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_create_coo for matrix B: " << m_status;
        duration = toc();
        PLOG_INFO << "----> CPUTIME : projected gradient : create_matrix_B : mkl_sparse_d_create_coo = " << duration;

        tic();
        m_status = mkl_sparse_convert_csr(B_coo, SPARSE_OPERATION_NON_TRANSPOSE, &m_B);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_convert_csr for matrix B: " << m_status;
        m_status = mkl_sparse_destroy ( B_coo );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix B_coo: " << m_status;
        m_status = mkl_sparse_order(m_B);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_order for matrix B: " << m_status;
        duration = toc();
        PLOG_INFO << "----> CPUTIME : projected gradient : create_matrix_B : mkl_sparse_convert_csr = " << duration;

        tic();
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_order for matrix B: " << m_status;
        m_status = mkl_sparse_set_mv_hint(m_B, SPARSE_OPERATION_NON_TRANSPOSE, m_descrB, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix B SPARSE_OPERATION_NON_TRANSPOSE: " << m_status;
        m_status = mkl_sparse_set_mv_hint(m_B, SPARSE_OPERATION_TRANSPOSE, m_descrB, 1 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix B SPARSE_OPERATION_TRANSPOSE: " << m_status;
        m_status = mkl_sparse_optimize ( m_B );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix B: " << m_status;
        duration = toc();
        PLOG_INFO << "----> CPUTIME : projected gradient : create_matrix_B : mkl_sparse_optimize = " << duration;
    }

    template <class problem_t, template<class> class gradient_t>
    void OptimProjectedGradient<problem_t, gradient_t>::create_matrix_A()
    {
        m_descrA.type = SPARSE_MATRIX_TYPE_GENERAL;
        sparse_matrix_t tmp;
        // tmp = P^{-1}*B
        m_descr_inv_P.type = SPARSE_MATRIX_TYPE_GENERAL;
        m_status = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, m_descr_inv_P, m_inv_P,
                                  SPARSE_OPERATION_TRANSPOSE, m_descrB, m_B,
                                  SPARSE_STAGE_FULL_MULT, &tmp);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_sp2m for tmp = P^{-1}*B^T: " << m_status;
        m_descr_inv_P.type = SPARSE_MATRIX_TYPE_DIAGONAL;
        m_descr_inv_P.diag = SPARSE_DIAG_NON_UNIT;
        m_status = mkl_sparse_order(tmp);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_order for matrix tmp: " << m_status;
        // A = B^T*tmp
        m_status = mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, m_descrB, m_B,
                                  SPARSE_OPERATION_NON_TRANSPOSE, m_descrB, tmp,
                                  SPARSE_STAGE_FULL_MULT, &m_A);
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_sp2m for A = B*tmp: " << m_status;
        m_status = mkl_sparse_destroy ( tmp );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix tmp: " << m_status;
        m_status = mkl_sparse_set_mv_hint(m_A, SPARSE_OPERATION_NON_TRANSPOSE, m_descrA, 2 );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS && m_status != SPARSE_STATUS_NOT_SUPPORTED) << "Error in mkl_sparse_set_mv_hint for matrix A: " << m_status;
        m_status = mkl_sparse_optimize ( m_A );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_optimize for matrix A: " << m_status;
    }

    template <class problem_t, template<class> class gradient_t>
    void OptimProjectedGradient<problem_t, gradient_t>::set_moment_matrix(std::size_t nparts,
                                                                          std::vector<MKL_INT>& invP_csr_row,
                                                                          std::vector<MKL_INT>& invP_csr_col,
                                                                          std::vector<double>& invP_csr_val,
                                                                          const scopi_container<2>& particles)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 2; ++d)
            {
                invP_csr_row.push_back(3*nparts + 3*i + d);
                invP_csr_col.push_back(3*nparts + 3*i + d);
                invP_csr_val.push_back(0.);
            }
            invP_csr_row.push_back(3*nparts + 3*i + 2);
            invP_csr_col.push_back(3*nparts + 3*i + 2);
            invP_csr_val.push_back(1./particles.j()(active_offset + i));
        }
    }

    template <class problem_t, template<class> class gradient_t>
    void OptimProjectedGradient<problem_t, gradient_t>::set_moment_matrix(std::size_t nparts,
                                                                         std::vector<MKL_INT>& invP_csr_row,
                                                                         std::vector<MKL_INT>& invP_csr_col,
                                                                         std::vector<double>& invP_csr_val,
                                                                         const scopi_container<3>& particles)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*nparts + 3*i + d);
                invP_csr_col.push_back(3*nparts + 3*i + d);
                invP_csr_val.push_back(1./particles.j()(active_offset + i)(d));
            }
        }
    }

    template<class problem_t, template<class> class gradient_t>
    OptimParams<OptimProjectedGradient<problem_t, gradient_t>>::OptimParams(const OptimParams<OptimProjectedGradient<problem_t, gradient_t>>& params)
    : tol_dg(params.tol_dg)
    , tol_l(params.tol_l)
    , max_iter(params.max_iter)
    , rho(params.rho)
    , verbose(params.verbose)
    {}

    template<class problem_t, template<class> class gradient_t>
    OptimParams<OptimProjectedGradient<problem_t, gradient_t>>::OptimParams()
    : tol_dg(1e-9)
    , tol_l(1e-9)
    , max_iter(40000)
    , rho(2000.)
    , verbose(false)
    {}

}
#endif

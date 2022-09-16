#pragma once

#include <new>
#ifdef SCOPI_USE_MOSEK
#include "OptimBase.hpp"
#include "../problems/DryWithoutFriction.hpp"
#include "ConstraintMosek.hpp"

#include <memory>
#include <fusion.h>

namespace scopi{

    template<class problem_t>
    class OptimMosek;

    /**
     * @brief Parameters for \c OptimMosek<problem_t>
     *
     * Specialization of ProblemParams in params.hpp
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t>
    struct OptimParams<OptimMosek<problem_t>>
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
        OptimParams(const OptimParams<OptimMosek<problem_t>>& params);

        /**
         * @brief Whether to modify the default parameters for Mosek's solver.
         *
         * Default value: \c true.
         *
         * Mosek provides many parameters for its solver.
         * Non-regression tests were performed with some parameters modified.
         * This flag modifies these parameters so the tests would pass.
         * If \c false, use the default values given by Mosek.
         */
        bool change_default_tol_mosek;
    };

    /**
     * @brief Solve optimization problem using Mosek.
     *
     * See ProblemBase.hpp for the notations.
     * Instead of minimizing \f$ \frac{1}{2} \u \P \cdot \u + \u \cdot \c \f$, 
     * minimize \f$ \uMosek \cdot \cMosek \f$, with
     * \f$ \uMosek = (\sMosek, \u, \zMosek) \in \mathbb{R}^{1+6\N+6\N} \f$ and \f$ \cMosek = (1, \c, \underbrace{0}_{\mathbb{R}^{6\N}}) \in \mathbb{R}^{1+6\N+6\N} \f$.
     *
     * Without friction (\c DryWithoutFriction and \c ViscousWithoutFriction), the constraint is written as \f$ \BMosek \uMosek \le \d \f$, \f$ \AzMosek \uMosek = 0 \f$,and \f$ (1, \sMosek, \zMosek) \in Q_r^{2+6\N} \f$, with 
     * \f[
     *      \begin{aligned}
     *          \BMosek &= \left. (\underbrace{0}_{1} | \underbrace{\B}_{6\N} | \underbrace{0}_{6\N}) \right\} \Nc,\\
     *          \AzMosek &= \left. (\underbrace{0}_{1} | \underbrace{\sqrt{\P}}_{6\N} | \underbrace{-\Id}_{6\N}) \right\} 6\N.
     *      \end{aligned}
     * \f]
     * \f$ Q_r^n \f$ is the rotated quadratic cone, \f$ Q_r^n = \{ x \in \mathbb{R}^n, 2 x_1 x_2 \ge x_3^2 + \dots + x_n^2 \} \f$, see Mosek's documentation for more details.
     * Here, \f$ \Nc \f$ is the number of constraints (\f$ D > 0 \f$ and \f$ D < 0 \f$).
     * \f$ \Id \f$ is the identity matrix.
     *
     * With friction, the constraint is written as \f$ \d \B \u \in \left( Q^4 \right)^{\Nc} \f$,
     * with \f$ Q^n \f$ the quadratic cone, \f$ Q^n = \{ x \in \mathbb{R}^n, x_1 \ge \sqrt{x_2^2 + \dots + x_n^2 } \} \f$, see Mosek's documentation for more details.
     * Each component of \f$ \B \u \f$ is seen as \f$ (\d_{\ij} + \B \u_{\ij}, \T \u_{\ij}^1, \T \u_{\ij}^2, \T \u_{\ij}^3 ) \f$.
     * \note Similarly to the case without friction, one can try to introduce a new variable \f$ t_{\ij} = ||\T \u_{\ij} \f$, but this resulted in poor performances.
     * \todo The constraint should be written as \f$ \BMosek \uMosek \in Q \f$ with appropriate reshape.
     * Currently, only a part of the matrix is used.
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t = DryWithoutFriction>
    class OptimMosek: public OptimBase<OptimMosek<problem_t>, problem_t>
    {
    public:
        /**
         * @brief Alias for the problem.
         */
        using problem_type = problem_t; 
    private:
        /**
         * @brief Alias for the base class \c OptimBase
         */
        using base_type = OptimBase<OptimMosek<problem_t>, problem_t>;

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
        template <std::size_t dim>
        OptimMosek(std::size_t nparts,
                   double dt,
                   const scopi_container<dim>& particles,
                   const OptimParams<OptimMosek<problem_t>>& optim_params,
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
         * @return Number of iterations Mosek's solver needed to converge.
         */
        template <std::size_t dim>
        int solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                            const std::vector<neighbor<dim>>& contacts,
                                            const std::vector<neighbor<dim>>& contacts_worms);
        /**
         * @brief \f$ \u \in \mathbb{R}^{6\N} \f$ contains the velocities and the rotations of the particles, the function returns the velocities solution of the optimization problem..
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ 3 \N \f$ elements.
         */
        double* uadapt_data();
        /**
         * @brief \f$ \u \in \mathbb{R}^{6\N} \f$ contains the velocities and the rotations of the particles, the function returns the rotations solution of the optimization problem..
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ 3 \N \f$ elements.
         */
        double* wadapt_data();
        /**
         * @brief Returns \f$ \d + \B \u \f$, where \f$ \u \f$ is the solution of the optimization problem.
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ \Nc \f$ elements.
         */
        double* constraint_data();
        /**
         * @brief Returns the Lagrange multipliers (solution of the dual problem) when the optimization is solved.
         *
         * \pre \c solve_optimization_problem has to be called before this function.
         *
         * @return \f$ \Nc \f$ elements.
         */
        double* lagrange_multiplier_data();
        /**
         * @brief Number of Lagrange multipliers > 0 (active constraints).
         */
        int get_nb_active_contacts_impl() const;

    private:
        /**
         * @brief 2D implementation to set the moments of inertia in the matrix \f$ \AzMosek \f$.
         *
         * @param nparts [in] Number of particles.
         * @param Az_rows [out] Rows' indicies of \f$ \AzMosek \f$ in COO storage.
         * @param Az_cols [out] Columns' indices of \f$ \AzMosek \f$ in COO storage.
         * @param Az_values [out] Values of \f$ \AzMosek \f$ in COO storage.
         * @param particles [in] Array of particles (for the moments of interia).
         */
        void set_moment_mass_matrix(std::size_t nparts,
                                    std::vector<int>& Az_rows,
                                    std::vector<int>& Az_cols,
                                    std::vector<double>& Az_values,
                                    const scopi_container<2>& particles);
        /**
         * @brief 3D implementation to set the moments of inertia in the matrix \f$ \AzMosek \f$.
         *
         * @param nparts [in] Number of particles.
         * @param Az_rows [out] Rows' indicies of \f$ \AzMosek \f$ in COO storage.
         * @param Az_cols [out] Columns' indices of \f$ \AzMosek \f$ in COO storage.
         * @param Az_values [out] Values of \f$ \AzMosek \f$ in COO storage.
         * @param particles [in] Array of particles (for the moments of interia).
         */
        void set_moment_mass_matrix(std::size_t nparts,
                                    std::vector<int>& Az_rows,
                                    std::vector<int>& Az_cols,
                                    std::vector<double>& Az_values,
                                    const scopi_container<3>& particles);

        /**
         * @brief Mosek's data structure (pointer) for the matrix \f$ \AzMosek \f$.
         */
        mosek::fusion::Matrix::t m_Az;
        /**
         * @brief Mosek's data structure (pointer) for the matrix \f$ \B \f$.
         */
        mosek::fusion::Matrix::t m_A;
        /**
         * @brief Mosek's data structure (pointer) to the solution of the optimization problem.
         */
        std::shared_ptr<monty::ndarray<double,1>> m_Xlvl;
        /**
         * @brief The constraint depends on the problem, this class help to deal with this. 
         *
         * \todo Shouldn't OptimMosek inherits from this class?
         */
        ConstraintMosek<problem_t> m_constraint;
        /**
         * @brief Mosek's data structure for \f$ \d \f$.
         */
        std::shared_ptr<monty::ndarray<double, 1>> m_D_mosek;
        /**
         * @brief Mosek's data structure to compute the constraint after solving the optimization problem.
         */
        std::shared_ptr<monty::ndarray<double, 1>> m_result_gemv; 
    };

    template<class problem_t>
    template<std::size_t dim>
    int OptimMosek<problem_t>::solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                               const std::vector<neighbor<dim>>& contacts,
                                                               const std::vector<neighbor<dim>>& contacts_worms)
    {
        using namespace mosek::fusion;
        using namespace monty;

        tic();
        Model::t model = new Model("contact"); auto _M = finally([&]() { model->dispose(); });
        // variables
        Variable::t X = model->variable("X", 1 + 6*this->m_nparts + 6*this->m_nparts);

        // functional to minimize
        auto c_mosek = std::make_shared<ndarray<double, 1>>(this->m_c.data(), shape_t<1>({this->m_c.shape(0)}));
        model->objective("minvar", ObjectiveSense::Minimize, Expr::dot(c_mosek, X));

        // constraints
        m_D_mosek = std::make_shared<monty::ndarray<double, 1>>(this->m_distances.data(), monty::shape_t<1>(this->m_distances.shape(0)));

        // matrix
        this->create_matrix_constraint_coo(particles, contacts, contacts_worms, m_constraint.index_first_col_matrix());
        m_A = Matrix::sparse(this->number_row_matrix(contacts, contacts_worms), m_constraint.number_col_matrix(),
                             std::make_shared<ndarray<int, 1>>(this->m_A_rows.data(), shape_t<1>({this->m_A_rows.size()})),
                             std::make_shared<ndarray<int, 1>>(this->m_A_cols.data(), shape_t<1>({this->m_A_cols.size()})),
                             std::make_shared<ndarray<double, 1>>(this->m_A_values.data(), shape_t<1>({this->m_A_values.size()})));

        m_constraint.add_constraints(m_D_mosek, m_A, X, model, contacts);
        Constraint::t qc2 = model->constraint("qc2", Expr::mul(m_Az, X), Domain::equalsTo(0.));
        Constraint::t qc3 = model->constraint("qc3", Expr::vstack(1, X->index(0), X->slice(1 + 6*this->m_nparts, 1 + 6*this->m_nparts + 6*this->m_nparts)), Domain::inRotatedQCone());

        // int thread_qty = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 0);
        // model->setSolverParam("numThreads", thread_qty);
        // model->setSolverParam("intpntCoTolPfeas", 1e-11);
        // model->setSolverParam("intpntTolPfeas", 1.e-11);
        if (this->m_params.change_default_tol_mosek)
        {
            model->setSolverParam("intpntCoTolPfeas", 1e-11);
            model->setSolverParam("intpntCoTolRelGap", 1e-11);
        }

        // model->setSolverParam("intpntCoTolDfeas", 1e-6);
        model->setLogHandler([](const std::string & msg) {PLOG_VERBOSE << msg << std::flush; } );
        //solve
        model->solve();

        m_Xlvl = X->level();
        m_constraint.update_dual(this->number_row_matrix(contacts, contacts_worms), contacts.size());
        for (auto& x : *(m_constraint.m_dual))
        {
            x *= -1.;
        }
        auto duration3 = toc();
        PLOG_INFO << "----> CPUTIME : Mosek solve = " << duration3;

        /*
        auto u = std::make_shared<monty::ndarray<double, 1>>(m_Xlvl->raw()+1, shape_t<1>(m_A->numColumns()));
        auto y = std::make_shared<monty::ndarray<double, 1>>(D_mosek->raw(), shape_t<1>(m_A->numRows()));
        mosek::LinAlg::gemv(false, m_A->numRows(), m_A->numColumns(), -1., m_A->transpose()->getDataAsArray(), u, 1.,  y);
        */

        return model->getSolverIntInfo("intpntIter");
    }

    template<class problem_t>
    template <std::size_t dim>
    OptimMosek<problem_t>::OptimMosek(std::size_t nparts,
                                      double dt,
                                      const scopi_container<dim>& particles,
                                      const OptimParams<OptimMosek<problem_t>>& optim_params,
                                      const ProblemParams<problem_t>& problem_params)
    : base_type(nparts, dt, 1 + 2*3*nparts + 2*3*nparts, 1, optim_params, problem_params)
    , m_constraint(nparts)
    {
        using namespace mosek::fusion;
        using namespace monty;

        this->m_c(0) = 1;

        // mass matrix
        std::vector<int> Az_rows;
        std::vector<int> Az_cols;
        std::vector<double> Az_values;

        Az_rows.reserve(6*nparts*2);
        Az_cols.reserve(6*nparts*2);
        Az_values.reserve(6*nparts*2);

        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                Az_rows.push_back(3*i + d);
                Az_cols.push_back(1 + 3*i + d);
                Az_values.push_back(std::sqrt(particles.m()(active_offset + i)));
                Az_rows.push_back(3*i + d);
                Az_cols.push_back(1 + 6*nparts + 3*i + d);
                Az_values.push_back(-1.);
            }
        }

        set_moment_mass_matrix(nparts, Az_rows, Az_cols, Az_values, particles);

        m_Az = Matrix::sparse(6*nparts, 1 + 6*nparts + 6*nparts,
                              std::make_shared<ndarray<int, 1>>(Az_rows.data(), shape_t<1>(Az_rows.size())),
                              std::make_shared<ndarray<int, 1>>(Az_cols.data(), shape_t<1>(Az_cols.size())),
                              std::make_shared<ndarray<double, 1>>(Az_values.data(), shape_t<1>(Az_values.size())));
    }

    template<class problem_t>
    double* OptimMosek<problem_t>::uadapt_data()
    {
        return m_Xlvl->raw() + 1;
    }

    template<class problem_t>
    double* OptimMosek<problem_t>::wadapt_data()
    {
        return m_Xlvl->raw() + 1 + 3*this->m_nparts;
    }

    template<class problem_t>
    double* OptimMosek<problem_t>::lagrange_multiplier_data()
    {
        return m_constraint.m_dual->raw();
    }

    template<class problem_t>
    double* OptimMosek<problem_t>::constraint_data()
    {
        using namespace monty;
        auto u = std::make_shared<monty::ndarray<double, 1>>(m_Xlvl->raw()+1, shape_t<1>(m_A->numColumns()));
        m_result_gemv = std::make_shared<monty::ndarray<double, 1>>(m_D_mosek->raw(), shape_t<1>(m_A->numRows()));
        mosek::LinAlg::gemv(false, m_A->numRows(), m_A->numColumns(), -1., m_A->transpose()->getDataAsArray(), u, 1.,  m_result_gemv);
        return m_result_gemv->raw();
    }

    template<class problem_t>
    int OptimMosek<problem_t>::get_nb_active_contacts_impl() const
    {
        int nb_active_contacts = 0;
        for (auto x : *(m_constraint.m_dual))
        {
            if(std::abs(x) > 1e-3)
                nb_active_contacts++;
        }
        return nb_active_contacts;
    }

    template<class problem_t>
    void OptimMosek<problem_t>::set_moment_mass_matrix(std::size_t nparts,
                                                       std::vector<int>& Az_rows,
                                                       std::vector<int>& Az_cols,
                                                       std::vector<double>& Az_values,
                                                       const scopi_container<2>& particles)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            Az_rows.push_back(3*nparts + 3*i + 2);
            Az_cols.push_back(1 + 3*nparts + 3*i + 2);
            Az_values.push_back(std::sqrt(particles.j()(active_offset + i)));

            Az_rows.push_back(3*nparts + 3*i + 2);
            Az_cols.push_back( 1 + 6*nparts + 3*nparts + 3*i + 2);
            Az_values.push_back(-1);
        }
    }

    template<class problem_t>
    void OptimMosek<problem_t>::set_moment_mass_matrix(std::size_t nparts,
                                                       std::vector<int>& Az_rows,
                                                       std::vector<int>& Az_cols,
                                                       std::vector<double>& Az_values,
                                                       const scopi_container<3>& particles)
    {
        auto active_offset = particles.nb_inactive();
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                Az_rows.push_back(3*nparts + 3*i + d);
                Az_cols.push_back(1 + 3*nparts + 3*i + d);
                Az_values.push_back(std::sqrt(particles.j()(active_offset + i)[d]));

                Az_rows.push_back(3*nparts + 3*i + d);
                Az_cols.push_back( 1 + 6*nparts + 3*nparts + 3*i + d);
                Az_values.push_back(-1);
            }
        }
    }

    template<class problem_t>
    OptimParams<OptimMosek<problem_t>>::OptimParams(const OptimParams<OptimMosek<problem_t>>& params)
    : change_default_tol_mosek(params.change_default_tol_mosek)
    {}

    template<class problem_t>
    OptimParams<OptimMosek<problem_t>>::OptimParams()
    : change_default_tol_mosek(true)
    {}
}
#endif

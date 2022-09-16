#pragma once

#include "OptimBase.hpp"

#include <cstddef>
#include <omp.h>
#include "tbb/tbb.h"

#include <xtensor/xadapt.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xnoalias.hpp>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

#include "../problems/DryWithoutFriction.hpp"

namespace scopi{
    /**
     * @brief Shared parameters for different implementation of Uzawa algorithm.
     */
    struct OptimParamsUzawaBase
    {
        /**
         * @brief Default constructor.
         */
        OptimParamsUzawaBase();
        /**
         * @brief Copy constructor.
         *
         * @param params Parameters to by copied.
         */
        OptimParamsUzawaBase(const OptimParamsUzawaBase& params);

        /**
         * @brief Tolerance.
         *
         * Default value is \f$ 10^{-9} \f$.
         * \note \c tol > 0
         */
        double tol;
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
         *
         * Reasonable performances are obtained with \c rho = \f$ \frac{0.2}{\Delta t^2} \f$.
         */
        double rho;
    };

    /**
     * @brief Uzawa algorithm to solve the optimization problem.
     *
     * See ProblemBase for the notations.
     * The algortihm requires matrix-vector products.
     * Several methods are implemented.
     *
     * The algorithm is:
     *  - \f$ \indexUzawa = 0 \f$;
     *  - \f$ \l^{\indexUzawa} = 0 \f$;
     *  - \f$ cmax = - \infty \f$;
     *  - While (\f$ cmax < tol \f$ and \f$ \indexUzawa < max\_iter \f$)
     *      - \f$ \mathbf{u}^{\indexUzawa+1} = \mathbb{P}^{-1} \left( \mathbf{c} - B^T \l^{\indexUzawa} \right) \f$;
     *      - \f$ \r^{\indexUzawa+1} = \mathbb{B} \mathbf{u}^{\indexUzawa+1} - \mathbf{d} \f$;
     *      - \f$ \l^{\indexUzawa+1}_{ij} = \max \left( \l_{ij}^{\indexUzawa} - \rho \r_{ij}^{\indexUzawa+1}, 0 \right) \f$;
     *      - \f$ cmax = \min_{ij} \left( \r_{ij}^{\indexUzawa+1} \right) \f$;
     *      - \f$ \indexUzawa++\f$.
     *
     * @tparam Derived Class that implements matrix-vector products.
     * @tparam problem_t Problem to be solved.
     */
    template<class Derived, class problem_t = DryWithoutFriction>
    class OptimUzawaBase: public OptimBase<Derived, problem_t>
    {
    private:
        /**
         * @brief Alias for the problem.
         */
        using problem_type = problem_t; 
        /**
         * @brief Alias for the base class \c OptimBase.
         */
        using base_type = OptimBase<Derived, problem_t>;

    protected:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         * @param dt [in] Time step.
         * @param optim_params [in] Parameters.
         * @param problem_params [in] Parameters for the problem.
         */
        OptimUzawaBase(std::size_t nparts,
                       double dt,
                       const OptimParams<Derived>& optim_params,
                       const ProblemParams<problem_t>& problem_params);

    public:
        /**
         * @brief Solve the optimization problem.
         *
         * Implements Uzawa algorithm, without knowing how the matrix-vector product is implemented.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         *
         * @return Number of iterations Uzawa algorithm needed to converge.
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
        auto uadapt_data();
        /**
         * @brief \f$ \mathbf{u} \in \mathbb{R}^{6N} \f$ contains the velocities and the rotations of the particles, the function returns the rotations solution of the optimization problem..
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
         * @brief Returns \f$ \mathbf{d} + \mathbb{B} \mathbf{u} \f$, where \f$ \mathbf{u} \f$ is the solution of the optimization problem.
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
         * @brief Computes \f$ \mathbb{P}^{-1} \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         */
        template <std::size_t dim>
        void gemv_inv_P(const scopi_container<dim>& particles);

        /**
         * @brief Computes \f$ \r = \r - \mathbb{B} \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim>
        void gemv_A(const scopi_container<dim>& particles,
                    const std::vector<neighbor<dim>>& contacts);

        /**
         * @brief Computes \f$ \mathbf{u} = \mathbb{B}^T \l + \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param contacts [in] Array of contacts.
         */
        template <std::size_t dim>
        void gemv_transpose_A(const scopi_container<dim>& particles,
                              const std::vector<neighbor<dim>>& contacts);

        /**
         * @brief Initialize the matrices for matrix-vector products with stored matrix.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for positions).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         */
        template <std::size_t dim>
        void init_uzawa(const scopi_container<dim>& particles,
                        const std::vector<neighbor<dim>>& contacts, 
                        const std::vector<neighbor<dim>>& contacts_worms);
        /**
         * @brief Free the memory allocated for the matrices.
         */
        void finalize_uzawa();

    protected:
        /**
         * @brief Vector \f$ \mathbf{u} \f$.
         */
        xt::xtensor<double, 1> m_U;
        /**
         * @brief Vector \f$ \l \f$.
         */
        xt::xtensor<double, 1> m_L;
        /**
         * @brief Vector \f$ \l \f$.
         */
        xt::xtensor<double, 1> m_R;

    private:
        /**
         * @brief Instead of imposing \f$ D > 0 \f$ between the particles, impose \f$ D > dmin \f$.
         */
        const double m_dmin;
    };

    template<class Derived, class problem_t>
    OptimUzawaBase<Derived, problem_t>::OptimUzawaBase(std::size_t nparts,
                                                       double dt,
                                                       const OptimParams<Derived>& optim_params,
                                                       const ProblemParams<problem_t>& problem_params)
    : base_type(nparts, dt, 2*3*nparts, 0, optim_params, problem_params)
    , m_U(xt::zeros<double>({6*nparts}))
    , m_dmin(0.)
    {}

    template<class Derived, class problem_t>
    template <std::size_t dim>
    int OptimUzawaBase<Derived, problem_t>::solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                                            const std::vector<neighbor<dim>>& contacts,
                                                                            const std::vector<neighbor<dim>>& contacts_worms)
    {
        tic();
        init_uzawa(particles, contacts, contacts_worms);
        auto duration = toc();
        m_L = xt::zeros<double>({this->number_row_matrix(contacts, contacts_worms)});
        m_R = xt::zeros<double>({this->number_row_matrix(contacts, contacts_worms)});
        PLOG_INFO << "----> CPUTIME : Uzawa matrix = " << duration;

        double time_assign_u = 0.;
        double time_gemv_transpose_A = 0.;
        double time_gemv_inv_P = 0.;
        double time_assign_r = 0.;
        double time_gemv_A = 0.;
        double time_assign_l = 0.;
        double time_compute_cmax = 0.;
        double time_solve = 0.;

        std::size_t cc = 0;
        double cmax = -1000.0;
        while ( (cmax<=-this->m_params.tol) && (cc <= this->m_params.max_iter) )
        {
            tic();
            xt::noalias(m_U) = this->m_c;
            auto duration = toc();
            time_assign_u += duration;
            time_solve += duration;

            tic();
            gemv_transpose_A(particles, contacts); // U = A^T * L + U
            duration = toc();
            time_gemv_transpose_A += duration;
            time_solve += duration;

            tic();
            gemv_inv_P(particles);  // U = - P^-1 * U
            duration = toc();
            time_gemv_inv_P += duration;
            time_solve += duration;

            tic();
            xt::noalias(m_R) = this->m_distances - m_dmin;
            duration = toc();
            time_assign_r += duration;
            time_solve += duration;

            tic();
            gemv_A(particles, contacts); // R = - A * U + R
            duration = toc();
            time_gemv_A += duration;
            time_solve += duration;

            tic();
            xt::noalias(m_L) = xt::maximum( m_L-this->m_params.rho*m_R, 0);
            duration = toc();
            time_assign_l += duration;
            time_solve += duration;

            tic();
            cmax = double((xt::amin(m_R))(0));
            duration = toc();
            time_compute_cmax += duration;
            time_solve += duration;
            cc += 1;

            PLOG_VERBOSE << "-- C++ -- Projection : minimal constraint : " << cmax;
        }

        PLOG_ERROR_IF(cc >= this->m_params.max_iter) << "Uzawa does not converge";

        PLOG_INFO << "----> CPUTIME : solve (total) = " << time_solve;
        PLOG_INFO << "----> CPUTIME : solve (U = c) = " << time_assign_u;
        PLOG_INFO << "----> CPUTIME : solve (U = A^T*L+U) = " << time_gemv_transpose_A;
        PLOG_INFO << "----> CPUTIME : solve (U = -P^-1*U) = " << time_gemv_inv_P;
        PLOG_INFO << "----> CPUTIME : solve (R = d) = " << time_assign_r;
        PLOG_INFO << "----> CPUTIME : solve (R = -A*U+R) = " << time_gemv_A;
        PLOG_INFO << "----> CPUTIME : solve (L = max(L-rho*R, 0)) = " << time_assign_l;
        PLOG_INFO << "----> CPUTIME : solve (cmax = min(R)) = " << time_compute_cmax;

        finalize_uzawa();

        return cc;
    }

    template<class Derived, class problem_t>
    auto OptimUzawaBase<Derived, problem_t>::uadapt_data()
    {
        return m_U.data();
    }

    template<class Derived, class problem_t>
    auto OptimUzawaBase<Derived, problem_t>::wadapt_data()
    {
        return m_U.data() + 3*this->m_nparts;
    }

    template<class Derived, class problem_t>
    auto OptimUzawaBase<Derived, problem_t>::lagrange_multiplier_data()
    {
        return m_L.data();
    }

    template<class Derived, class problem_t>
    double* OptimUzawaBase<Derived, problem_t>::constraint_data()
    {
        return NULL;
    }

    template<class Derived, class problem_t>
    int OptimUzawaBase<Derived, problem_t>::get_nb_active_contacts_impl() const
    {
        return xt::sum(xt::where(m_L > 0., xt::ones_like(m_L), xt::zeros_like(m_L)))();
    }

    template<class Derived, class problem_t>
    template <std::size_t dim>
    void OptimUzawaBase<Derived, problem_t>::gemv_inv_P(const scopi_container<dim>& particles)
    {
        static_cast<Derived&>(*this).gemv_inv_P_impl(particles);
    }

    template<class Derived, class problem_t>
    template <std::size_t dim>
    void OptimUzawaBase<Derived, problem_t>::gemv_A(const scopi_container<dim>& particles,
                                                    const std::vector<neighbor<dim>>& contacts)
    {
        static_cast<Derived&>(*this).gemv_A_impl(particles, contacts);
    }

    template<class Derived, class problem_t>
    template <std::size_t dim>
    void OptimUzawaBase<Derived, problem_t>::gemv_transpose_A(const scopi_container<dim>& particles,
                                                              const std::vector<neighbor<dim>>& contacts)
    {
        static_cast<Derived&>(*this).gemv_transpose_A_impl(particles, contacts);
    }

    template<class Derived, class problem_t>
    template <std::size_t dim>
    void OptimUzawaBase<Derived, problem_t>::init_uzawa(const scopi_container<dim>& particles,
                                                        const std::vector<neighbor<dim>>& contacts,
                                                        const std::vector<neighbor<dim>>& contacts_worms)
    {
        static_cast<Derived&>(*this).init_uzawa_impl(particles, contacts, contacts_worms);
    }

    template<class Derived, class problem_t>
    void OptimUzawaBase<Derived, problem_t>::finalize_uzawa()
    {
        static_cast<Derived&>(*this).finalize_uzawa_impl();
    }

    OptimParamsUzawaBase::OptimParamsUzawaBase(const OptimParamsUzawaBase& params)
    : tol(params.tol)
    , max_iter(params.max_iter)
    , rho(params.rho)
    {}

    OptimParamsUzawaBase::OptimParamsUzawaBase()
    : tol(1e-9)
    , max_iter(40000)
    , rho(2000.)
    {}

}

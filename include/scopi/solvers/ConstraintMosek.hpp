#pragma once

#ifdef SCOPI_USE_MOSEK
#include <fusion.h>

#include "../problems/DryWithFriction.hpp"
#include "../problems/DryWithFrictionFixedPoint.hpp"
#include "../problems/DryWithoutFriction.hpp"
#include "../problems/ViscousWithFriction.hpp"
#include "../problems/ViscousWithoutFriction.hpp"
#include "OptimBase.hpp"

namespace scopi
{
    template <class problem_t, class constraint_t, std::size_t dim>
    void add_constraints_impl(std::shared_ptr<monty::ndarray<double, 1>> D,
                              mosek::fusion::Matrix::t A,
                              mosek::fusion::Variable::t X,
                              mosek::fusion::Model::t model,
                              const std::vector<neighbor<dim>>&,
                              constraint_t& qc,
                              const problem_t& problem)
    {
        using namespace mosek::fusion;
        qc = model->constraint("qc1", Expr::mul(A, X->slice(1, 1 + 6 * problem.size())), Domain::lessThan(D));
    }

    template <class problem_t, class constraint_t, std::size_t dim>
    void update_dual_impl(const problem_t&,
                          const std::vector<neighbor<dim>>&,
                          std::shared_ptr<monty::ndarray<double, 1>>& dual,
                          const constraint_t& qc)
    {
        dual = qc->dual();
    }

    //
    // With friction
    //
    /////////////////////////////////////////////////////////////////////////////////////
    template <class constraint_t, std::size_t dim>
    void add_constraints_impl(std::shared_ptr<monty::ndarray<double, 1>> D,
                              mosek::fusion::Matrix::t A,
                              mosek::fusion::Variable::t X,
                              mosek::fusion::Model::t model,
                              const std::vector<neighbor<dim>>& contacts,
                              constraint_t& qc,
                              const DryWithFriction& problem)
    {
        using namespace mosek::fusion;
        qc = model->constraint("qc1",
                               Expr::reshape(Expr::sub(D, Expr::mul(A, X->slice(1, 1 + 6 * problem.size()))), contacts.size(), 4),
                               Domain::inQCone());
    }

    template <class constraint_t, std::size_t dim>
    void update_dual_impl(const DryWithFriction&,
                          const std::vector<neighbor<dim>>& contacts,
                          std::shared_ptr<monty::ndarray<double, 1>>& dual,
                          const constraint_t& qc)
    {
        dual = std::make_shared<monty::ndarray<double, 1>>(qc->dual()->raw(), monty::shape_t<1>(contacts.size()));
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            dual->raw()[i] = -qc->dual()->raw()[4 * i];
        }
    }

    //
    // With friction fixed point
    //
    /////////////////////////////////////////////////////////////////////////////////////
    template <class constraint_t, std::size_t dim>
    void add_constraints_impl(std::shared_ptr<monty::ndarray<double, 1>> D,
                              mosek::fusion::Matrix::t A,
                              mosek::fusion::Variable::t X,
                              mosek::fusion::Model::t model,
                              const std::vector<neighbor<dim>>& contacts,
                              constraint_t& qc,
                              const DryWithFrictionFixedPoint& problem)
    {
        using namespace mosek::fusion;
        qc = model->constraint("qc1",
                               Expr::reshape(Expr::sub(D, Expr::mul(A, X->slice(1, 1 + 6 * problem.size()))), contacts.size(), 4),
                               Domain::inQCone());
    }

    template <class constraint_t, std::size_t dim>
    void update_dual_impl(const DryWithFrictionFixedPoint&,
                          const std::vector<neighbor<dim>>& contacts,
                          std::shared_ptr<monty::ndarray<double, 1>>& dual,
                          const constraint_t& qc)
    {
        using namespace mosek::fusion;
        using namespace monty;
        dual = std::make_shared<monty::ndarray<double, 1>>(qc->dual()->raw(), shape_t<1>(contacts.size()));
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            dual->raw()[i] = -qc->dual()->raw()[4 * i];
        }
    }

    //
    // Viscous with friction
    //
    /////////////////////////////////////////////////////////////////////////////////////
    template <class constraint_t, std::size_t dim>
    void add_constraints_impl(std::shared_ptr<monty::ndarray<double, 1>> D,
                              mosek::fusion::Matrix::t A,
                              mosek::fusion::Variable::t X,
                              mosek::fusion::Model::t model,
                              const std::vector<neighbor<dim>>& contacts,
                              constraint_t& qc,
                              const ViscousWithFriction<dim>& problem)
    {
        using namespace mosek::fusion;
        using namespace monty;

        auto nb_gamma_min = problem.get_nb_gamma_min();
        auto nb_gamma_neg = problem.get_nb_gamma_neg();

        auto D_restricted_1 = std::make_shared<ndarray<double, 1>>(D->raw(), shape_t<1>(contacts.size() - nb_gamma_min + nb_gamma_neg));
        qc[0]               = model->constraint("qc1",
                                  Expr::mul(A, X->slice(1, 1 + 6 * problem.size()))->slice(0, contacts.size() - nb_gamma_min + nb_gamma_neg),
                                  Domain::lessThan(D_restricted_1));

        auto D_restricted_4 = std::make_shared<ndarray<double, 1>>(D->raw() + (contacts.size() - nb_gamma_min + nb_gamma_neg),
                                                                   shape_t<1>(4 * nb_gamma_min));

        if (nb_gamma_min != 0)
        {
            qc[1] = model->constraint("qc4",
                                      Expr::reshape(Expr::sub(D_restricted_4,
                                                              (Expr::mul(A, X->slice(1, 1 + 6 * problem.size())))
                                                                  ->slice(contacts.size() - nb_gamma_min + nb_gamma_neg,
                                                                          contacts.size() - nb_gamma_min + nb_gamma_neg + 4 * nb_gamma_min)),
                                                    nb_gamma_min,
                                                    4),
                                      Domain::inQCone());
        }
    }

    template <class constraint_t, std::size_t dim>
    void update_dual_impl(const ViscousWithFriction<dim>& problem,
                          const std::vector<neighbor<dim>>& contacts,
                          std::shared_ptr<monty::ndarray<double, 1>>& dual,
                          const constraint_t& qc)
    {
        using namespace mosek::fusion;
        using namespace monty;

        dual = std::make_shared<monty::ndarray<double, 1>>(shape_t<1>(problem.number_row_matrix(contacts)));

        for (std::size_t i = 0; i < qc[0]->dual()->size(); ++i)
        {
            dual->raw()[i] = qc[0]->dual()->raw()[i];
        }

        auto nb_gamma_min    = problem.get_nb_gamma_min();
        std::size_t qc1_size = qc[0]->dual()->size();
        if (nb_gamma_min != 0)
        {
            for (std::size_t i = 0; i < 4 * nb_gamma_min; ++i)
            {
                dual->raw()[qc1_size + i] = qc[1]->dual()->raw()[i];
            }
        }
    }

    namespace detail
    {
        template <class T>
        struct constraint
        {
            using type = mosek::fusion::Constraint::t;
        };

        template <std::size_t dim>
        struct constraint<ViscousWithFriction<dim>>
        {
            using type = std::array<mosek::fusion::Constraint::t, 2>;
        };

        template <std::size_t dim>
        struct constraint<ViscousWithoutFriction<dim>>
        {
            using type = std::array<mosek::fusion::Constraint::t, 2>;
        };

        template <class T>
        using constraint_t = typename constraint<T>::type;
    }

    /**
     * @brief Helper to set the constraint in OptimMosek.
     *
     * The constraint depends on the problem, template specializations of this
     * class help manage the dependance on the problem.
     *
     * @tparam problem_t Problem to be solved.
     */
    template <class problem_t>
    class ConstraintMosek
    {
      public:

        ConstraintMosek(const problem_t& problem);

        /**
         * @brief Number of columns in the matrix (\f$ 1 + 6N + 6N \f$).
         */
        std::size_t number_col_matrix() const;

        /**
         * @brief Add the constraint \f$ \mathbf{d} + \mathbb{B} \mathbf{u} \ge
         * 0 \f$ in Mosek's solver.
         *
         * @tparam dim Dimension (2 or 3).
         * @param D [in] Array \f$ \mathbf{d} \f$.
         * @param A [in] Matrix \f$ \tilde{\mathbb{B}} \f$.
         * @param X [in] Unknown \f$ \mathbf{u} \f$.
         * @param model [in] Mosek's solver.
         * @param contacts [in] Array of contatcs (for compatibility with other
         * problems).
         */
        template <std::size_t dim>
        void add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                             mosek::fusion::Matrix::t A,
                             mosek::fusion::Variable::t X,
                             mosek::fusion::Model::t model,
                             const std::vector<neighbor<dim>>& contacts);
        /**
         * @brief Get the solution of the dual problem.
         *
         * Call \pre <tt> model->solve() </tt> before this function.
         *
         * @param nb_row_matrix [in] Number of row of the matrix \f$
         * \tilde{\mathbb{B}} \f$ (for compatibility with other problems).
         * @param nb_contacts [in] Number of contacts (for compatibility with
         * other problems).
         */
        template <std::size_t dim>
        void update_dual(const std::vector<neighbor<dim>>& contacts);

        /**
         * @brief Mosek's data structure for the solution of the dual problem.
         */
        std::shared_ptr<monty::ndarray<double, 1>> m_dual;

      private:

        /**
         * @brief Mosek's data structure for the constraint.
         */
        detail::constraint_t<problem_t> m_qc;
        const problem_t& problem;
    };

    template <class problem_t>
    ConstraintMosek<problem_t>::ConstraintMosek(const problem_t& problem)
        : problem(problem)
    {
    }

    template <class problem_t>
    std::size_t ConstraintMosek<problem_t>::number_col_matrix() const
    {
        return 6 * problem.size();
    }

    template <class problem_t>
    template <std::size_t dim>
    void ConstraintMosek<problem_t>::update_dual(const std::vector<neighbor<dim>>& contacts)
    {
        update_dual_impl(problem, contacts, m_dual, m_qc);
    }

    template <class problem_t>
    template <std::size_t dim>
    void ConstraintMosek<problem_t>::add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                     mosek::fusion::Matrix::t A,
                                                     mosek::fusion::Variable::t X,
                                                     mosek::fusion::Model::t model,
                                                     const std::vector<neighbor<dim>>& contacts)
    {
        add_constraints_impl(D, A, X, model, contacts, m_qc, problem);
    }
} // namespace scopi
#endif

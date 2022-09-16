#pragma once

#ifdef SCOPI_USE_MOSEK
#include <fusion.h>

#include "../problems/DryWithoutFriction.hpp"
#include "../problems/DryWithFriction.hpp"
#include "../problems/DryWithFrictionFixedPoint.hpp"
#include "../problems/ViscousWithoutFriction.hpp"
#include "../problems/ViscousWithFriction.hpp"

namespace scopi
{
    /**
     * @brief Helper to set the constraint in OptimMosek.
     *
     * The constraint depends on the problem, template specializations of this class help manage the dependance on the problem.
     *
     * @tparam problem_t Problem to be solved.
     */
    template<class problem_t>
    class ConstraintMosek
    {
    };

    /**
     * @brief Specialization of \c ConstraintMosek for \c DryWithoutFriction.
     */
    template<>
    class ConstraintMosek<DryWithoutFriction>
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         */
        ConstraintMosek(std::size_t nparts);
        /**
         * @brief Number of columns in the matrix (\f$ 1 + 6N + 6N \f$).
         */
        std::size_t number_col_matrix() const;
        /**
         * @brief A matrix larger than \f$ \mathbb{B} \f$ is built (see OptimMosek), the function returns the index of the first column of \f$ \mathbb{B} \f$ in \f$ \tilde{\mathbb{B}} \f$.
         *
         * @return 1.
         */
        std::size_t index_first_col_matrix() const;

        /**
         * @brief Add the constraint \f$ \mathbf{d} + \mathbb{B} \u \ge 0 \f$ in Mosek's solver.
         *
         * @tparam dim Dimension (2 or 3).
         * @param D [in] Array \f$ \mathbf{d} \f$.
         * @param A [in] Matrix \f$ \tilde{\mathbb{B}} \f$.
         * @param X [in] Unknown \f$ \u \f$.
         * @param model [in] Mosek's solver.
         * @param contacts [in] Array of contatcs (for compatibility with other problems).
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
         * @param nb_row_matrix [in] Number of row of the matrix \f$ \tilde{\mathbb{B}} \f$ (for compatibility with other problems).
         * @param nb_contacts [in] Number of contacts (for compatibility with other problems).
         */
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts);

        /**
         * @brief Mosek's data structure for the solution of the dual problem.
         */
        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        /**
         * @brief Number of particles.
         */
        std::size_t m_nparticles;
        /**
         * @brief Mosek's data structure for the constraint.
         */
        mosek::fusion::Constraint::t m_qc1;
    };

    template <std::size_t dim>
    void ConstraintMosek<DryWithoutFriction>::add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                             mosek::fusion::Matrix::t A,
                                                             mosek::fusion::Variable::t X,
                                                             mosek::fusion::Model::t model,
                                                             const std::vector<neighbor<dim>>&)
    {
        using namespace mosek::fusion;
        m_qc1 =  model->constraint("qc1", Expr::mul(A, X), Domain::lessThan(D));
    }






    /**
     * @brief Specialization of \c ConstraintMosek for \c DryWithFriction.
     */
    template<>
    class ConstraintMosek<DryWithFriction>
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         */
        ConstraintMosek(std::size_t nparts);
        /**
         * @brief Number of columns in the matrix (\f$ 6 N \f$).
         */
        std::size_t number_col_matrix() const;
        /**
         * @brief The matrix \f$ \tilde{\mathbb{B}} \f$ contains the matrices \f$ \mathbb{B} \f$ and \f$ \mathbb{T} \f$ (see ProblemBase for the notations), but it is not a larger matrix (see <tt> ConstraintMosek<DryWithFriction> </tt> for the difference).  
         *
         * @return 0.
         */
        std::size_t index_first_col_matrix() const;
        /**
         * @brief Add the constraint \f$ \mathbf{d}_{ij} + \mathbb{B} \u_{ij} \ge ||\mathbb{T} \u_{ij}|| \f$ in Mosek's solver.
         *
         * @tparam dim Dimension (2 or 3).
         * @param D [in] Array \f$ \mathbf{d} \f$.
         * @param A [in] Matrix \f$ \tilde{\mathbb{B}} \f$.
         * @param X [in] Unknown \f$ \u \f$.
         * @param model [in] Mosek's solver.
         * @param contacts [in] Array of contatcs.
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
         * @param nb_row_matrix [in] Number of row of the matrix \f$ \tilde{\mathbb{B}} \f$ (for compatibility with other problems).
         * @param nb_contacts [in] Number of contacts (for compatibility with other problems).
         */
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts);

        /**
         * @brief Mosek's data structure for the solution of the dual problem.
         */
        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        /**
         * @brief Number of particles.
         */
        std::size_t m_nparticles;
        /**
         * @brief Mosek's data structure for the constraint.
         */
        mosek::fusion::Constraint::t m_qc1;
    };

    template <std::size_t dim>
    void ConstraintMosek<DryWithFriction>::add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                           mosek::fusion::Matrix::t A,
                                                           mosek::fusion::Variable::t X,
                                                           mosek::fusion::Model::t model,
                                                           const std::vector<neighbor<dim>>& contacts)
    {
        using namespace mosek::fusion;
        m_qc1 = model->constraint("qc1"
                , Expr::reshape(Expr::sub(D, Expr::mul(A, X->slice(1, 1 + 6*this->m_nparticles))), contacts.size(), 4)
                , Domain::inQCone());
    }






    /**
     * @brief Specialization of \c ConstraintMosek for \c DryWithFrictionFixedPoint.
     */
    template<>
    class ConstraintMosek<DryWithFrictionFixedPoint>
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         */
        ConstraintMosek(std::size_t nparts);
        /**
         * @brief Number of columns in the matrix (\f$ 6 N \f$).
         */
        std::size_t number_col_matrix() const;
        /**
         * @brief The matrix \f$ \tilde{\mathbb{B}} \f$ contains the matrices \f$ \mathbb{B} \f$ and \f$ \mathbb{T} \f$ (see ProblemBase for the notations), but it is not a larger matrix (see <tt> ConstraintMosek<DryWithFriction> </tt> for the difference).  
         *
         * @return 0.
         */
        std::size_t index_first_col_matrix() const;
        /**
         * @brief Add the constraint \f$ \mathbf{d}_{ij} + \mathbb{B} \u_{ij} \ge \mathbb{T} \u_{ij}|| \f$ in Mosek's solver.
         *
         * @tparam dim Dimension (2 or 3).
         * @param D [in] Array \f$ \mathbf{d} \f$.
         * @param A [in] Matrix \f$ \tilde{\mathbb{B}} \f$.
         * @param X [in] Unknown \f$ \u \f$.
         * @param model [in] Mosek's solver.
         * @param contacts [in] Array of contatcs.
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
         * @param nb_row_matrix [in] Number of row of the matrix \f$ \tilde{\mathbb{B}} \f$ (for compatibility with other problems).
         * @param nb_contacts [in] Number of contacts (for compatibility with other problems).
         */
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts);

        /**
         * @brief Mosek's data structure for the solution of the dual problem.
         */
        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        /**
         * @brief Number of particles.
         */
        std::size_t m_nparticles;
        /**
         * @brief Mosek's data structure for the constraint.
         */
        mosek::fusion::Constraint::t m_qc1;
    };

    template <std::size_t dim>
    void ConstraintMosek<DryWithFrictionFixedPoint>::add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                                     mosek::fusion::Matrix::t A,
                                                                     mosek::fusion::Variable::t X,
                                                                     mosek::fusion::Model::t model,
                                                                     const std::vector<neighbor<dim>>& contacts)
    {
        using namespace mosek::fusion;
        m_qc1 = model->constraint("qc1"
                , Expr::reshape(Expr::sub(D, Expr::mul(A, X->slice(1, 1 + 6*this->m_nparticles))), contacts.size(), 4)
                , Domain::inQCone());
    }






    /**
     * @brief Specialization of \c ConstraintMosek for \c ViscousWithoutFriction<dim>.
     *
     * @tparam dim Dimension(2 or 3).
     */
    template<std::size_t dim>
    class ConstraintMosek<ViscousWithoutFriction<dim>>
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         */
        ConstraintMosek(std::size_t nparts);
        /**
         * @brief Number of columns in the matrix (\f$ 1 + 6N + 6N \f$).
         */
        std::size_t number_col_matrix() const;
        /**
         * @brief A matrix larger than \f$ \mathbb{B} \f$ is built (see OptimMosek), the function returns the index of the first column of \f$ \mathbb{B} \f$ in \f$ \tilde{\mathbb{B}} \f$.
         *
         * @return 1.
         */
        std::size_t index_first_col_matrix() const;

        /**
         * @brief Add the constraint \f$ \mathbf{d} + \mathbb{B} \u \ge 0 \f$ in Mosek's solver.
         *
         * @tparam dim Dimension (2 or 3).
         * @param D [in] Array \f$ \mathbf{d} \f$.
         * @param A [in] Matrix \f$ \tilde{\mathbb{B}} \f$.
         * @param X [in] Unknown \f$ \u \f$.
         * @param model [in] Mosek's solver.
         * @param contacts [in] Array of contatcs (for compatibility with other problems).
         */
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
         * @param nb_row_matrix [in] Number of row of the matrix \f$ \tilde{\mathbb{B}} \f$ (for compatibility with other problems).
         * @param nb_contacts [in] Number of contacts (for compatibility with other problems).
         */
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts);

        /**
         * @brief Mosek's data structure for the solution of the dual problem.
         */
        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        /**
         * @brief Number of particles.
         */
        std::size_t m_nparticles;
        /**
         * @brief Mosek's data structure for the constraint.
         */
        mosek::fusion::Constraint::t m_qc1;
    };

    template <std::size_t dim>
    void ConstraintMosek<ViscousWithoutFriction<dim>>::add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                                       mosek::fusion::Matrix::t A,
                                                                       mosek::fusion::Variable::t X,
                                                                       mosek::fusion::Model::t model,
                                                                       const std::vector<neighbor<dim>>&)

    {
        using namespace mosek::fusion;
        using namespace monty;

        m_qc1 = model->constraint("qc1", Expr::mul(A, X), Domain::lessThan(D));
    }

    template <std::size_t dim>
    ConstraintMosek<ViscousWithoutFriction<dim>>::ConstraintMosek(std::size_t nparticles)
    : m_nparticles(nparticles)
    {}

    template <std::size_t dim>
    std::size_t ConstraintMosek<ViscousWithoutFriction<dim>>::index_first_col_matrix() const
    {
        return 1;
    }

    template <std::size_t dim>
    std::size_t ConstraintMosek<ViscousWithoutFriction<dim>>::number_col_matrix() const
    {
        return 1 + 6*m_nparticles + 6*m_nparticles;
    }

    template <std::size_t dim>
    void ConstraintMosek<ViscousWithoutFriction<dim>>::update_dual(std::size_t,
                                                                   std::size_t)
    {
        m_dual = m_qc1->dual();
    }






    /**
     * @brief Specialization of \c ConstraintMosek for \c ViscousWithFriction<dim>.
     *
     * @tparam dim Dimension(2 or 3).
     */
    template<std::size_t dim>
    class ConstraintMosek<ViscousWithFriction<dim>> : protected ViscousWithFriction<dim>
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         */
        ConstraintMosek(std::size_t nparts);
        /**
         * @brief Number of columns in the matrix (\f$ 6 N \f$).
         */
        std::size_t number_col_matrix() const;
        /**
         * @brief The matrix \f$ \tilde{\mathbb{B}} \f$ contains the matrices \f$ \mathbb{B} \f$ and \f$ \mathbb{T} \f$ (see ProblemBase for the notations), but it is not a larger matrix (see <tt> ConstraintMosek<DryWithFriction> </tt> for the difference).  
         *
         * @return 0.
         */
        std::size_t index_first_col_matrix() const;

        /**
         * @brief Add the constraints \f$ \mathbf{d} + \mathbb{B} \u \ge 0 \f$ and \f$ \mathbf{d}_{ij} + \mathbb{B} \u_{ij} \ge ||\mathbb{T} \u_{ij}|| \f$ in Mosek's solver.
         *
         * @tparam dim Dimension (2 or 3).
         * @param D [in] Array \f$ \mathbf{d} \f$.
         * @param A [in] Matrix \f$ \tilde{\mathbb{B}} \f$.
         * @param X [in] Unknown \f$ \u \f$.
         * @param model [in] Mosek's solver.
         * @param contacts [in] Array of contatcs.
         */
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
         * @param nb_row_matrix [in] Number of row of the matrix \f$ \tilde{\mathbb{B}} \f$.
         * @param nb_contacts [in] Number of contacts.
         */
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts);

        /**
         * @brief Mosek's data structure for the solution of the dual problem.
         */
        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        /**
         * @brief Number of particles.
         */
        std::size_t m_nparticles;
        /**
         * @brief Mosek's data structure for the constraint \f$ \mathbf{d} + \mathbb{B} \u \ge 0 \f$.
         */
        mosek::fusion::Constraint::t m_qc1;
        /**
         * @brief Mosek's data structure for the constraint \f$ \mathbf{d}_{ij} + \mathbb{B} \u_{ij} \ge ||\mathbb{T} \u_{ij}|| \f$.
         */
        mosek::fusion::Constraint::t m_qc4;
    };

    template <std::size_t dim>
    void ConstraintMosek<ViscousWithFriction<dim>>::add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                                    mosek::fusion::Matrix::t A,
                                                                    mosek::fusion::Variable::t X,
                                                                    mosek::fusion::Model::t model,
                                                                    const std::vector<neighbor<dim>>& contacts)
    {
        using namespace mosek::fusion;
        using namespace monty;

        auto D_restricted_1 = std::make_shared<ndarray<double, 1>>(D->raw(), shape_t<1>({contacts.size() - this->get_nb_gamma_min() + this->get_nb_gamma_neg()}));
        m_qc1 = model->constraint("qc1", Expr::mul(A, X->slice(1, 1 + 6*this->m_nparticles))->slice(0, contacts.size() - this->get_nb_gamma_min() + this->get_nb_gamma_neg()), Domain::lessThan(D_restricted_1));

        auto D_restricted_4 = std::make_shared<ndarray<double, 1>>(D->raw()+(contacts.size() - this->get_nb_gamma_min() + this->get_nb_gamma_neg()), shape_t<1>(4*this->get_nb_gamma_min()));
        m_qc4 = model->constraint("qc4", 
                Expr::reshape(
                    Expr::sub(D_restricted_4, (Expr::mul(A, X->slice(1, 1 + 6*this->m_nparticles)))->slice(contacts.size() - this->get_nb_gamma_min() + this->get_nb_gamma_neg(), contacts.size() - this->get_nb_gamma_min() + this->get_nb_gamma_neg() + 4*this->get_nb_gamma_min()) ),
                    this->get_nb_gamma_min(), 4),
                Domain::inQCone());
    }

    template <std::size_t dim>
    ConstraintMosek<ViscousWithFriction<dim>>::ConstraintMosek(std::size_t nparticles)
    : ViscousWithFriction<dim>(nparticles)
    , m_nparticles(nparticles)
    {}

    template <std::size_t dim>
    std::size_t ConstraintMosek<ViscousWithFriction<dim>>::index_first_col_matrix() const
    {
        return 0;
    }

    template <std::size_t dim>
    std::size_t ConstraintMosek<ViscousWithFriction<dim>>::number_col_matrix() const
    {
        return 6*m_nparticles;
    }

    template <std::size_t dim>
    void ConstraintMosek<ViscousWithFriction<dim>>::update_dual(std::size_t nb_row_matrix,
                                                                std::size_t nb_contacts)
    {
        using namespace mosek::fusion;
        using namespace monty;
        m_dual = std::make_shared<monty::ndarray<double, 1>>(m_qc1->dual()->raw(), shape_t<1>(nb_row_matrix));
        for (std::size_t i = 0; i < 4*this->get_nb_gamma_min(); ++i)
        {
            m_dual->raw()[nb_contacts - this->get_nb_gamma_min() + this->get_nb_gamma_neg() + i] = -m_qc4->dual()->raw()[i];
        }
    }
}
#endif

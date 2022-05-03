#pragma once

#ifdef SCOPI_USE_MOSEK
#include <fusion.h>

#include "../problems/MatrixOptimSolver.hpp"
#include "../problems/MatrixOptimSolverFriction.hpp"
#include "../problems/MatrixOptimSolverViscosity.hpp"

namespace scopi
{
    template<class model_t>
    class ConstraintMosek
    {
    };

    template<>
    class ConstraintMosek<MatrixOptimSolver>
    {
    protected:
        ConstraintMosek(std::size_t nparts);
        std::size_t number_col_matrix() const;
        std::size_t index_first_col_matrix() const;

        template <std::size_t dim>
        void add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                             mosek::fusion::Matrix::t A,
                             mosek::fusion::Variable::t X,
                             mosek::fusion::Model::t model,
                             const std::vector<neighbor<dim>>& contacts,
                             std::size_t nb_gamma_neg,
                             std::size_t nb_gamma_min);
        void update_dual(std::size_t nb_row_matrix,
                        std::size_t nb_contacts,
                        std::size_t nb_gamma_neg,
                        std::size_t nb_gamma_min);

        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        std::size_t m_nparticles;
        mosek::fusion::Constraint::t m_qc1;
    };

    template <std::size_t dim>
    void ConstraintMosek<MatrixOptimSolver>::add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                             mosek::fusion::Matrix::t A,
                                                             mosek::fusion::Variable::t X,
                                                             mosek::fusion::Model::t model,
                                                             const std::vector<neighbor<dim>>&,
                                                             std::size_t,
                                                             std::size_t)
    {
        using namespace mosek::fusion;
        m_qc1 =  model->constraint("qc1", Expr::mul(A, X), Domain::lessThan(D));
    }






    template<>
    class ConstraintMosek<MatrixOptimSolverFriction>
    {
    protected:
        ConstraintMosek(std::size_t nparts);
        std::size_t number_col_matrix() const;
        std::size_t index_first_col_matrix() const;
        template <std::size_t dim>
        void add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                             mosek::fusion::Matrix::t A,
                             mosek::fusion::Variable::t X,
                             mosek::fusion::Model::t model,
                             const std::vector<neighbor<dim>>& contacts,
                             std::size_t nb_gamma_neg,
                             std::size_t nb_gamma_min);
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts,
                         std::size_t nb_gamma_neg,
                         std::size_t nb_gamma_min);

        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        std::size_t m_nparticles;
        mosek::fusion::Constraint::t m_qc1;
    };

    template <std::size_t dim>
    void ConstraintMosek<MatrixOptimSolverFriction>::add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                                     mosek::fusion::Matrix::t A,
                                                                     mosek::fusion::Variable::t X,
                                                                     mosek::fusion::Model::t model,
                                                                     const std::vector<neighbor<dim>>& contacts,
                                                                     std::size_t,
                                                                     std::size_t)
    {
        using namespace mosek::fusion;
        m_qc1 = model->constraint("qc1"
                , Expr::reshape(Expr::sub(D, Expr::mul(A, X->slice(1, 1 + 6*this->m_nparticles))), contacts.size(), 4)
                , Domain::inQCone());
    }





    template<std::size_t dim>
    class ConstraintMosek<MatrixOptimSolverViscosity<dim>>
    {
    protected:
        ConstraintMosek(std::size_t nparts);
        std::size_t number_col_matrix() const;
        std::size_t index_first_col_matrix() const;

        void add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                             mosek::fusion::Matrix::t A,
                             mosek::fusion::Variable::t X,
                             mosek::fusion::Model::t model,
                             const std::vector<neighbor<dim>>& contacts,
                             std::size_t nb_gamma_neg,
                             std::size_t nb_gamma_min);
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts,
                         std::size_t nb_gamma_neg,
                         std::size_t nb_gamma_min);

        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        std::size_t m_nparticles;
        mosek::fusion::Constraint::t m_qc1;
        mosek::fusion::Constraint::t m_qc4;
    };

    template <std::size_t dim>
    void ConstraintMosek<MatrixOptimSolverViscosity<dim>>::add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                                           mosek::fusion::Matrix::t A,
                                                                           mosek::fusion::Variable::t X,
                                                                           mosek::fusion::Model::t model,
                                                                           const std::vector<neighbor<dim>>& contacts,
                                                                           std::size_t nb_gamma_neg,
                                                                           std::size_t nb_gamma_min)
    {
        using namespace mosek::fusion;
        using namespace monty;
        auto D_restricted = std::make_shared<ndarray<double, 1>>(D->raw(), shape_t<1>({contacts.size() - nb_gamma_min + nb_gamma_neg}));
        m_qc1 = model->constraint("qc1", Expr::mul(A, X->slice(1, 1 + 6*this->m_nparticles))->slice(0, contacts.size() - nb_gamma_min + nb_gamma_neg), Domain::lessThan(D_restricted));
        m_qc4 = model->constraint("qc4", 
                Expr::reshape(
                    Expr::sub(D_restricted, (Expr::mul(A, X->slice(1, 1 + 6*this->m_nparticles)))->slice(contacts.size() - nb_gamma_min + nb_gamma_neg, contacts.size() - nb_gamma_min + nb_gamma_neg + 2*4*nb_gamma_min) ),
                    2*nb_gamma_min, 4),
                Domain::inQCone());
    }

    template <std::size_t dim>
    ConstraintMosek<MatrixOptimSolverViscosity<dim>>::ConstraintMosek(std::size_t nparticles)
    : m_nparticles(nparticles)
    {}

    template <std::size_t dim>
    std::size_t ConstraintMosek<MatrixOptimSolverViscosity<dim>>::index_first_col_matrix() const
    {
        return 0;
    }

    template <std::size_t dim>
    std::size_t ConstraintMosek<MatrixOptimSolverViscosity<dim>>::number_col_matrix() const
    {
        return 6*m_nparticles;
    }

    template <std::size_t dim>
    void ConstraintMosek<MatrixOptimSolverViscosity<dim>>::update_dual(std::size_t nb_row_matrix,
                                                                       std::size_t nb_contacts,
                                                                       std::size_t nb_gamma_neg,
                                                                       std::size_t nb_gamma_min)
    {
        using namespace mosek::fusion;
        using namespace monty;
        m_dual = std::make_shared<monty::ndarray<double, 1>>(m_qc1->dual()->raw(), shape_t<1>(nb_row_matrix));
        for (std::size_t i = 0; i < 2*4*nb_gamma_min; ++i)
        {
            m_dual->raw()[nb_contacts - nb_gamma_min + nb_gamma_neg + i] = -m_qc4->dual()->raw()[i];
        }
    }

}
#endif

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
        mosek::fusion::Constraint::t constraint(std::shared_ptr<monty::ndarray<double, 1>> D,
                                       mosek::fusion::Matrix::t A,
                                       mosek::fusion::Variable::t X,
                                       mosek::fusion::Model::t model,
                                       const std::vector<neighbor<dim>>& contacts) const;

    private:
        std::size_t m_nparticles;
    };

    template <std::size_t dim>
    mosek::fusion::Constraint::t ConstraintMosek<MatrixOptimSolver>::constraint(std::shared_ptr<monty::ndarray<double, 1>> D,
                                   mosek::fusion::Matrix::t A,
                                   mosek::fusion::Variable::t X,
                                   mosek::fusion::Model::t model,
                                   const std::vector<neighbor<dim>>&) const
    {
        using namespace mosek::fusion;
        return model->constraint("qc1", Expr::mul(A, X), Domain::lessThan(D));
    }




    template<>
    class ConstraintMosek<MatrixOptimSolverFriction>
    {
    protected:
        ConstraintMosek(std::size_t nparts);
        std::size_t number_col_matrix() const;
        std::size_t index_first_col_matrix() const;
        template <std::size_t dim>
        mosek::fusion::Constraint::t constraint(std::shared_ptr<monty::ndarray<double, 1>> D,
                                       mosek::fusion::Matrix::t A,
                                       mosek::fusion::Variable::t X,
                                       mosek::fusion::Model::t model,
                                       const std::vector<neighbor<dim>>& contacts) const;

    private:
        std::size_t m_nparticles;
    };

    template <std::size_t dim>
    mosek::fusion::Constraint::t ConstraintMosek<MatrixOptimSolverFriction>::constraint(std::shared_ptr<monty::ndarray<double, 1>> D,
                                   mosek::fusion::Matrix::t A,
                                   mosek::fusion::Variable::t X,
                                   mosek::fusion::Model::t model,
                                   const std::vector<neighbor<dim>>& contacts) const
    {
        using namespace mosek::fusion;
        return model->constraint("qc1"
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

        mosek::fusion::Constraint::t constraint(std::shared_ptr<monty::ndarray<double, 1>> D,
                                       mosek::fusion::Matrix::t A,
                                       mosek::fusion::Variable::t X,
                                       mosek::fusion::Model::t model,
                                       const std::vector<neighbor<dim>>& contacts,
                                       std::size_t nb_gamma_neg,
                                       std::size_t nb_gamma_min) const;

        mosek::fusion::Constraint::t extraConstraint(std::shared_ptr<monty::ndarray<double, 1>> D,
                                                     mosek::fusion::Matrix::t A,
                                                     mosek::fusion::Variable::t X,
                                                     mosek::fusion::Model::t model,
                                                     const std::vector<neighbor<dim>>& contacts,
                                                     std::size_t nb_gamma_neg,
                                                     std::size_t nb_gamma_min) const;

    private:
        std::size_t m_nparticles;
    };

    template <std::size_t dim>
    mosek::fusion::Constraint::t ConstraintMosek<MatrixOptimSolverViscosity<dim>>::constraint(std::shared_ptr<monty::ndarray<double, 1>> D,
                                   mosek::fusion::Matrix::t A,
                                   mosek::fusion::Variable::t X,
                                   mosek::fusion::Model::t model,
                                   const std::vector<neighbor<dim>>& contacts,
                                   std::size_t nb_gamma_neg,
                                   std::size_t nb_gamma_min) const
    {
        using namespace mosek::fusion;
        using namespace monty;
        auto D_restricted = std::make_shared<ndarray<double, 1>>(D->raw(), shape_t<1>({contacts.size() - nb_gamma_min + nb_gamma_neg}));
        return model->constraint("qc1", Expr::mul(A, X->slice(1, 1 + 6*this->m_nparticles))->slice(0, contacts.size() - nb_gamma_min + nb_gamma_neg), Domain::lessThan(D_restricted));
    }

    template <std::size_t dim>
    mosek::fusion::Constraint::t ConstraintMosek<MatrixOptimSolverViscosity<dim>>::extraConstraint(std::shared_ptr<monty::ndarray<double, 1>> D,
                                   mosek::fusion::Matrix::t A,
                                   mosek::fusion::Variable::t X,
                                   mosek::fusion::Model::t model,
                                   const std::vector<neighbor<dim>>& contacts,
                                   std::size_t nb_gamma_neg,
                                   std::size_t nb_gamma_min) const
    {
        using namespace mosek::fusion;
        using namespace monty;
        auto D_restricted = std::make_shared<ndarray<double, 1>>(D->raw()+(contacts.size() - nb_gamma_min + nb_gamma_neg), shape_t<1>(4*nb_gamma_min));
        return model->constraint("qc4", 
                Expr::reshape(
                    Expr::sub(D_restricted, (Expr::mul(A, X->slice(1, 1 + 6*this->m_nparticles)))->slice(contacts.size() - nb_gamma_min + nb_gamma_neg, contacts.size() - nb_gamma_min + nb_gamma_neg + 4*nb_gamma_min) ),
                    nb_gamma_min, 4),
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
}
#endif

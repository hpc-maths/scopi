#pragma once

#ifdef SCOPI_USE_MOSEK
#include <fusion.h>

#include "MatrixOptimSolver.hpp"
#include "MatrixOptimSolverFriction.hpp"

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
        std::shared_ptr<monty::ndarray<double, 1>> distances_to_vector(xt::xtensor<double, 1> distances) const;
        template <std::size_t dim>
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts) const;
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
    std::size_t ConstraintMosek<MatrixOptimSolver>::number_row_matrix(const std::vector<neighbor<dim>>& contacts) const
    {
        return contacts.size();
    }

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
        std::shared_ptr<monty::ndarray<double, 1>> distances_to_vector(xt::xtensor<double, 1> distances) const;
        template <std::size_t dim>
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts) const;
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
    std::size_t ConstraintMosek<MatrixOptimSolverFriction>::number_row_matrix(const std::vector<neighbor<dim>>& contacts) const
    {
        return 4*contacts.size();
    }

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

}
#endif

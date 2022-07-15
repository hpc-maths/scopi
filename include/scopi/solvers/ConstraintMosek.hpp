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
    template<class problem_t>
    class ConstraintMosek
    {
    };

    // DryWithoutFriction
    template<>
    class ConstraintMosek<DryWithoutFriction>
    {
    public:
        ConstraintMosek(std::size_t nparts);
        std::size_t number_col_matrix() const;
        std::size_t index_first_col_matrix() const;

        template <std::size_t dim>
        void add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                             mosek::fusion::Matrix::t A,
                             mosek::fusion::Variable::t X,
                             mosek::fusion::Model::t model,
                             const std::vector<neighbor<dim>>& contacts);
        void update_dual(std::size_t nb_row_matrix,
                        std::size_t nb_contacts);

        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        std::size_t m_nparticles;
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






    // DryWithFriction
    template<>
    class ConstraintMosek<DryWithFriction>
    {
    public:
        ConstraintMosek(std::size_t nparts);
        std::size_t number_col_matrix() const;
        std::size_t index_first_col_matrix() const;
        template <std::size_t dim>
        void add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                             mosek::fusion::Matrix::t A,
                             mosek::fusion::Variable::t X,
                             mosek::fusion::Model::t model,
                             const std::vector<neighbor<dim>>& contacts);
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts);

        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        std::size_t m_nparticles;
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






    // DryWithFrictionFixedPoint
    template<>
    class ConstraintMosek<DryWithFrictionFixedPoint>
    {
    public:
        ConstraintMosek(std::size_t nparts);
        std::size_t number_col_matrix() const;
        std::size_t index_first_col_matrix() const;
        template <std::size_t dim>
        void add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                             mosek::fusion::Matrix::t A,
                             mosek::fusion::Variable::t X,
                             mosek::fusion::Model::t model,
                             const std::vector<neighbor<dim>>& contacts);
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts);

        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        std::size_t m_nparticles;
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






    // ViscousWithoutFriction
    template<std::size_t dim>
    class ConstraintMosek<ViscousWithoutFriction<dim>>
    {
    public:
        ConstraintMosek(std::size_t nparts);
        std::size_t number_col_matrix() const;
        std::size_t index_first_col_matrix() const;

        void add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                             mosek::fusion::Matrix::t A,
                             mosek::fusion::Variable::t X,
                             mosek::fusion::Model::t model,
                             const std::vector<neighbor<dim>>& contacts);
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts);

        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        std::size_t m_nparticles;
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






    // ViscousWithFriction
    template<std::size_t dim>
    class ConstraintMosek<ViscousWithFriction<dim>> : protected ViscousWithFriction<dim>
    {
    public:
        ConstraintMosek(std::size_t nparts);
        std::size_t number_col_matrix() const;
        std::size_t index_first_col_matrix() const;

        void add_constraints(std::shared_ptr<monty::ndarray<double, 1>> D,
                             mosek::fusion::Matrix::t A,
                             mosek::fusion::Variable::t X,
                             mosek::fusion::Model::t model,
                             const std::vector<neighbor<dim>>& contacts);
        void update_dual(std::size_t nb_row_matrix,
                         std::size_t nb_contacts);

        std::shared_ptr<monty::ndarray<double,1>> m_dual;

    private:
        std::size_t m_nparticles;
        mosek::fusion::Constraint::t m_qc1;
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

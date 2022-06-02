#pragma once

#ifdef SCOPI_USE_MOSEK
#include "OptimBase.hpp"
#include "../problems/DryWithoutFriction.hpp"
#include "ConstraintMosek.hpp"
#include "../params/OptimParams.hpp"

#include <memory>
#include <fusion.h>

namespace scopi{

    template<class problem_t>
    class OptimMosek;

    template<>
    class OptimParams<OptimMosek>
    {};

    template<class problem_t = DryWithoutFriction>
    class OptimMosek: public OptimBase<OptimMosek<problem_t>, problem_t>
                    , public ConstraintMosek<problem_t>
    {
    public:
        using base_type = OptimBase<OptimMosek, problem_t>;

        template <std::size_t dim>
        OptimMosek(std::size_t nparts, double dt, const scopi_container<dim>& particles, OptimParams<OptimMosek>& params);

        template <std::size_t dim>
        int solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                            const std::vector<neighbor<dim>>& contacts);

        double* uadapt_data();
        double* wadapt_data();
        double* lagrange_multiplier_data();
        int get_nb_active_contacts_impl() const;

    private:

        void set_moment_mass_matrix(std::size_t nparts,
                                    std::vector<int>& Az_rows,
                                    std::vector<int>& Az_cols,
                                    std::vector<double>& Az_values,
                                    const scopi_container<2>& particles);
        void set_moment_mass_matrix(std::size_t nparts,
                                    std::vector<int>& Az_rows,
                                    std::vector<int>& Az_cols,
                                    std::vector<double>& Az_values,
                                    const scopi_container<3>& particles);

        mosek::fusion::Matrix::t m_Az;
        mosek::fusion::Matrix::t m_A;
        std::shared_ptr<monty::ndarray<double,1>> m_Xlvl;

        OptimParams<OptimMosek> m_params;
    };

    template<class problem_t>
    template<std::size_t dim>
    int OptimMosek<problem_t>::solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                    const std::vector<neighbor<dim>>& contacts)
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
        auto D_mosek = std::make_shared<monty::ndarray<double, 1>>(this->m_distances.data(), monty::shape_t<1>(this->m_distances.shape(0)));

        // matrix
        this->create_matrix_constraint_coo(particles, contacts, this->index_first_col_matrix());
        m_A = Matrix::sparse(this->number_row_matrix(contacts), this->number_col_matrix(),
                             std::make_shared<ndarray<int, 1>>(this->m_A_rows.data(), shape_t<1>({this->m_A_rows.size()})),
                             std::make_shared<ndarray<int, 1>>(this->m_A_cols.data(), shape_t<1>({this->m_A_cols.size()})),
                             std::make_shared<ndarray<double, 1>>(this->m_A_values.data(), shape_t<1>({this->m_A_values.size()})));

        this->add_constraints(D_mosek, m_A, X, model, contacts, this->get_nb_gamma_neg(), this->get_nb_gamma_min());
        Constraint::t qc2 = model->constraint("qc2", Expr::mul(m_Az, X), Domain::equalsTo(0.));
        Constraint::t qc3 = model->constraint("qc3", Expr::vstack(1, X->index(0), X->slice(1 + 6*this->m_nparts, 1 + 6*this->m_nparts + 6*this->m_nparts)), Domain::inRotatedQCone());

        // int thread_qty = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 0);
        // model->setSolverParam("numThreads", thread_qty);
        // model->setSolverParam("intpntCoTolPfeas", 1e-11);
        // model->setSolverParam("intpntTolPfeas", 1.e-11);
        model->setSolverParam("intpntCoTolPfeas", 1e-11);
        model->setSolverParam("intpntCoTolRelGap", 1e-11);

        // model->setSolverParam("intpntCoTolDfeas", 1e-6);
        model->setLogHandler([](const std::string & msg) {PLOG_VERBOSE << msg << std::flush; } );
        //solve
        model->solve();

        m_Xlvl = X->level();
        this->update_dual(this->number_row_matrix(contacts), contacts.size(), this->get_nb_gamma_neg(), this->get_nb_gamma_min());
        for (auto& x : *this->m_dual)
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
    OptimMosek<problem_t>::OptimMosek(std::size_t nparts, double dt, const scopi_container<dim>& particles, OptimParams<OptimMosek>& params)
    : base_type(nparts, dt, 1 + 2*3*nparts + 2*3*nparts, 1)
    , ConstraintMosek<problem_t>(nparts)
    , m_params(params)
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
    double* OptimMosek<problem_t>::lagrange_multiplier_data()
    {
        return this->m_dual->raw();
    }

    template<class problem_t>
    double* OptimMosek<problem_t>::wadapt_data()
    {
        return m_Xlvl->raw() + 1 + 3*this->m_nparts;
    }

    template<class problem_t>
    int OptimMosek<problem_t>::get_nb_active_contacts_impl() const
    {
        int nb_active_contacts = 0;
        for (auto x : *(this->m_dual))
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
                Az_values.push_back(std::sqrt(particles.j()(active_offset + i)(d)));

                Az_rows.push_back(3*nparts + 3*i + d);
                Az_cols.push_back( 1 + 6*nparts + 3*nparts + 3*i + d);
                Az_values.push_back(-1);
            }
        }
    }
}
#endif

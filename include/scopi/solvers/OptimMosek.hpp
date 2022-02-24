#pragma once

#ifdef SCOPI_USE_MOSEK
#include "OptimBase.hpp"
#include "MatrixOptimSolver.hpp"

#include <memory>
#include <fusion.h>

namespace scopi{
    using namespace mosek::fusion;
    using namespace monty;

    class OptimMosek: public OptimBase<OptimMosek>
                    , public MatrixOptimSolver
    {
    public:
        using base_type = OptimBase<OptimMosek>;

        OptimMosek(std::size_t nparts, double dt, double tol = 1e-8);

        template <std::size_t dim>
        int solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                            const std::vector<neighbor<dim>>& contacts);

        double* uadapt_data();
        double* wadapt_data();
        int get_nb_active_contacts_impl() const;

    private:
        Matrix::t m_Az;
        Matrix::t m_A;
        std::shared_ptr<ndarray<double,1>> m_Xlvl;
        std::shared_ptr<ndarray<double,1>> m_dual;
    };

    template<std::size_t dim>
    int OptimMosek::solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                                    const std::vector<neighbor<dim>>& contacts)
    {
        tic();
        Model::t model = new Model("contact"); auto _M = finally([&]() { model->dispose(); });
        // variables
        Variable::t X = model->variable("X", 1 + 6*this->m_nparts + 6*this->m_nparts);

        // functional to minimize
        auto c_mosek = std::make_shared<ndarray<double, 1>>(this->m_c.data(), shape_t<1>({this->m_c.shape(0)}));
        model->objective("minvar", ObjectiveSense::Minimize, Expr::dot(c_mosek, X));

        // constraints
        auto D_mosek = std::make_shared<ndarray<double, 1>>(this->m_distances.data(), shape_t<1>({this->m_distances.shape(0)}));
        auto duration1 = toc();

        // matrix
        tic();
        this->create_matrix_constraint_coo(particles, contacts, 1);
        m_A = Matrix::sparse(contacts.size(), 1 + 6*this->m_nparts + 6*this->m_nparts,
                             std::make_shared<ndarray<int, 1>>(this->m_A_rows.data(), shape_t<1>({this->m_A_rows.size()})),
                             std::make_shared<ndarray<int, 1>>(this->m_A_cols.data(), shape_t<1>({this->m_A_cols.size()})),
                             std::make_shared<ndarray<double, 1>>(this->m_A_values.data(), shape_t<1>({this->m_A_values.size()})));
        auto duration2 = toc();
        PLOG_INFO << "----> CPUTIME : Mosek matrix = " << duration2;

        tic();
        Constraint::t qc1 = model->constraint("qc1", Expr::mul(m_A, X), Domain::lessThan(D_mosek));
        Constraint::t qc2 = model->constraint("qc2", Expr::mul(m_Az, X), Domain::equalsTo(0.));
        Constraint::t qc3 = model->constraint("qc3", Expr::vstack(1, X->index(0), X->slice(1 + 6*this->m_nparts, 1 + 6*this->m_nparts + 6*this->m_nparts)), Domain::inRotatedQCone());
        // int thread_qty = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 0);
        // model->setSolverParam("numThreads", thread_qty);
        model->setSolverParam("intpntCoTolPfeas", 1e-11);
        model->setSolverParam("intpntTolPfeas", 1.e-11);

        // model->setSolverParam("intpntCoTolDfeas", 1e-6);
        // model->setLogHandler([](const std::string & msg) { std::cout << msg << std::flush; } );
        //solve
        model->solve();

        m_Xlvl = X->level();
        m_dual = qc1->dual();
        auto duration3 = toc();
        PLOG_INFO << "----> CPUTIME : Mosek solve = " << duration1 + duration3;

        return model->getSolverIntInfo("intpntIter");
    }
}
#endif

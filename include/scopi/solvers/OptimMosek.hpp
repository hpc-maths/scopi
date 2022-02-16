#pragma once

#ifdef SCOPI_USE_MOSEK
#include "OptimBase.hpp"
#include "MatrixOptimSolver.hpp"

#include <memory>
#include <fusion.h>

namespace scopi{
    template<class model_t = MatrixOptimSolver>
    class OptimMosek: public OptimBase<OptimMosek<model_t>>
                    , public model_t
    {
    public:
        using base_type = OptimBase<OptimMosek>;

        OptimMosek(std::size_t nparts, double dt, double mu = 0., double tol = 1e-8);

        template <std::size_t dim>
        int solve_optimization_problem_impl(const scopi_container<dim>& particles,
                                            const std::vector<neighbor<dim>>& contacts);

        double* uadapt_data();
        double* wadapt_data();
        int get_nb_active_contacts_impl() const;

    private:

        mosek::fusion::Matrix::t m_Az;
        mosek::fusion::Matrix::t m_A;
        std::shared_ptr<monty::ndarray<double,1>> m_Xlvl;
        std::shared_ptr<monty::ndarray<double,1>> m_dual;
    };

    template<class model_t>
    template<std::size_t dim>
    int OptimMosek<model_t>::solve_optimization_problem_impl(const scopi_container<dim>& particles,
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
        auto D_mosek = this->distances_to_mosek_vector(this->m_distances);
        auto duration1 = toc();

        // matrix
        this->create_matrix_constraint_coo(particles, contacts, this->matrix_first_col_index_mosek());
        m_A = Matrix::sparse(this->number_row_matrix_mosek(contacts), this->number_col_matrix_mosek(),
                             std::make_shared<ndarray<int, 1>>(this->m_A_rows.data(), shape_t<1>({this->m_A_rows.size()})),
                             std::make_shared<ndarray<int, 1>>(this->m_A_cols.data(), shape_t<1>({this->m_A_cols.size()})),
                             std::make_shared<ndarray<double, 1>>(this->m_A_values.data(), shape_t<1>({this->m_A_values.size()})));

        Constraint::t qc1 = this->constraint_mosek(D_mosek, m_A, X, model, contacts);
        Constraint::t qc2 = model->constraint("qc2", Expr::mul(m_Az, X), Domain::equalsTo(0.));
        Constraint::t qc3 = model->constraint("qc3", Expr::vstack(1, X->index(0), X->slice(1 + 6*this->m_nparts, 1 + 6*this->m_nparts + 6*this->m_nparts)), Domain::inRotatedQCone());

        // int thread_qty = std::max(atoi(std::getenv("OMP_NUM_THREADS")), 0);
        // model->setSolverParam("numThreads", thread_qty);
        // model->setSolverParam("intpntCoTolPfeas", 1e-11);
        // model->setSolverParam("intpntTolPfeas", 1.e-11);

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

    template<class model_t>
    OptimMosek<model_t>::OptimMosek(std::size_t nparts, double dt,  double mu, double)
    : base_type(nparts, dt, 1 + 2*3*nparts + 2*3*nparts, 1)
    , model_t(nparts, dt, mu)
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

        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 2; ++d)
            {
                Az_rows.push_back(3*i + d);
                Az_cols.push_back(1 + 3*i + d);
                Az_values.push_back(std::sqrt(this->m_mass)); // TODO: add mass into particles
                Az_rows.push_back(3*i + d);
                Az_cols.push_back(1 + 6*nparts + 3*i + d);
                Az_values.push_back(-1.);
            }

            Az_rows.push_back(3*nparts + 3*i + 2);
            Az_cols.push_back(1 + 3*nparts + 3*i + 2);
            Az_values.push_back(std::sqrt(this->m_moment));

            Az_rows.push_back(3*nparts + 3*i + 2);
            Az_cols.push_back( 1 + 6*nparts + 3*nparts + 3*i + 2);
            Az_values.push_back(-1);
        }

        m_Az = Matrix::sparse(6*nparts, 1 + 6*nparts + 6*nparts,
                              std::make_shared<ndarray<int, 1>>(Az_rows.data(), shape_t<1>({Az_rows.size()})),
                              std::make_shared<ndarray<int, 1>>(Az_cols.data(), shape_t<1>({Az_cols.size()})),
                              std::make_shared<ndarray<double, 1>>(Az_values.data(), shape_t<1>({Az_values.size()})));
    }

    template<class model_t>
    double* OptimMosek<model_t>::uadapt_data()
    {
        return m_Xlvl->raw() + 1;
    }

    template<class model_t>
    double* OptimMosek<model_t>::wadapt_data()
    {
        return m_Xlvl->raw() + 1 + 3*this->m_nparts;
    }

    template<class model_t>
    int OptimMosek<model_t>::get_nb_active_contacts_impl() const
    {
        int nb_active_contacts = 0;
        for (auto x : *m_dual)
        {
            if(std::abs(x) > 1e-3)
                nb_active_contacts++;
        }
        return nb_active_contacts;
    }
}
#endif

#pragma once

#ifdef SCOPI_USE_MOSEK
#include "OptimBase.hpp"
#include "MatrixOptimSolver.hpp"

#include <memory>
#include <fusion.h>

namespace scopi{
    using namespace mosek::fusion;
    using namespace monty;

    template <std::size_t dim>
    class OptimMosek: public OptimBase<OptimMosek<dim>, dim>
                    , public MatrixOptimSolver<OptimMosek<dim>, dim>
    {
    public:
        OptimMosek(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, double tol = 1e-8);
        void create_matrix_constraint_impl(const std::vector<neighbor<dim>>& contacts);
        void create_matrix_mass_impl();
        int solve_optimization_problem_impl(const std::vector<neighbor<dim>>& contacts);
        auto get_uadapt_impl();
        auto get_wadapt_impl();
        void setup_impl(const std::vector<neighbor<dim>>& contacts);
        void tear_down_impl();
        int get_nb_active_contacts_impl();

    private:
        using base_type = OptimBase<OptimMosek<dim>, dim>;
        Matrix::t m_Az;
        Matrix::t m_A;
        std::shared_ptr<ndarray<double,1>> m_Xlvl;
        std::shared_ptr<ndarray<double,1>> m_dual;
    };

    template<std::size_t dim>
    OptimMosek<dim>::OptimMosek(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, double)
    : OptimBase<OptimMosek<dim>, dim>(particles, dt, Nactive, active_ptr, 1 + 2*3*Nactive + 2*3*Nactive, 1)
    , MatrixOptimSolver<OptimMosek<dim>, dim>(particles, dt, Nactive, active_ptr)
    {
        this->m_c(0) = 1;
    }

    template<std::size_t dim>
    void OptimMosek<dim>::setup_impl(const std::vector<neighbor<dim>>& contacts)
    {
        // constraint matrix
        std::vector<int> A_rows;
        std::vector<int> A_cols;
        std::vector<double> A_values;

        this->create_matrix_constraint_coo(contacts, A_rows, A_cols, A_values, 1);

        m_A = Matrix::sparse(contacts.size(), 1 + 6*base_type::m_Nactive + 6*base_type::m_Nactive,
                             std::make_shared<ndarray<int, 1>>(A_rows.data(), shape_t<1>({A_rows.size()})),
                             std::make_shared<ndarray<int, 1>>(A_cols.data(), shape_t<1>({A_cols.size()})),
                             std::make_shared<ndarray<double, 1>>(A_values.data(), shape_t<1>({A_values.size()})));

        // mass matrix
        std::vector<int> Az_rows;
        std::vector<int> Az_cols;
        std::vector<double> Az_values;

        Az_rows.reserve(6*base_type::m_Nactive*2);
        Az_cols.reserve(6*base_type::m_Nactive*2);
        Az_values.reserve(6*base_type::m_Nactive*2);

        for (std::size_t i = 0; i < base_type::m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < 2; ++d)
            {
                Az_rows.push_back(3*i + d);
                Az_cols.push_back(1 + 3*i + d);
                Az_values.push_back(std::sqrt(this->m_mass)); // TODO: add mass into particles
                Az_rows.push_back(3*i + d);
                Az_cols.push_back(1 + 6*base_type::m_Nactive + 3*i + d);
                Az_values.push_back(-1.);
            }

            Az_rows.push_back(3*base_type::m_Nactive + 3*i + 2);
            Az_cols.push_back(1 + 3*base_type::m_Nactive + 3*i + 2);
            Az_values.push_back(std::sqrt(this->m_moment));

            Az_rows.push_back(3*base_type::m_Nactive + 3*i + 2);
            Az_cols.push_back( 1 + 6*base_type::m_Nactive + 3*base_type::m_Nactive + 3*i + 2);
            Az_values.push_back(-1);
        }

        m_Az = Matrix::sparse(6*base_type::m_Nactive, 1 + 6*base_type::m_Nactive + 6*base_type::m_Nactive,
                              std::make_shared<ndarray<int, 1>>(Az_rows.data(), shape_t<1>({Az_rows.size()})),
                              std::make_shared<ndarray<int, 1>>(Az_cols.data(), shape_t<1>({Az_cols.size()})),
                              std::make_shared<ndarray<double, 1>>(Az_values.data(), shape_t<1>({Az_values.size()})));
    }

    template<std::size_t dim>
    int OptimMosek<dim>::solve_optimization_problem_impl(const std::vector<neighbor<dim>>&)
    {
        Model::t model = new Model("contact"); auto _M = finally([&]() { model->dispose(); });
        // variables
        Variable::t X = model->variable("X", 1 + 6*base_type::m_Nactive + 6*base_type::m_Nactive);

        // functional to minimize
        auto c_mosek = std::make_shared<ndarray<double, 1>>(this->m_c.data(), shape_t<1>({this->m_c.shape(0)}));
        model->objective("minvar", ObjectiveSense::Minimize, Expr::dot(c_mosek, X));

        // constraints
        auto D_mosek = std::make_shared<ndarray<double, 1>>(this->m_distances.data(), shape_t<1>({this->m_distances.shape(0)}));

        Constraint::t qc1 = model->constraint("qc1", Expr::mul(m_A, X), Domain::lessThan(D_mosek));
        Constraint::t qc2 = model->constraint("qc2", Expr::mul(m_Az, X), Domain::equalsTo(0.));
        Constraint::t qc3 = model->constraint("qc3", Expr::vstack(1, X->index(0), X->slice(1 + 6*base_type::m_Nactive, 1 + 6*base_type::m_Nactive + 6*base_type::m_Nactive)), Domain::inRotatedQCone());
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

        return model->getSolverIntInfo("intpntIter");
    }

    template<std::size_t dim>
    auto OptimMosek<dim>::get_uadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_Xlvl->raw()+1), {base_type::m_Nactive, 3UL});
    }

    template<std::size_t dim>
    auto OptimMosek<dim>::get_wadapt_impl()
    {
        return xt::adapt(reinterpret_cast<double*>(m_Xlvl->raw()+1+3*base_type::m_Nactive), {base_type::m_Nactive, 3UL});
    }

    template<std::size_t dim>
    int OptimMosek<dim>::get_nb_active_contacts_impl()
    {
        int nb_active_contacts = 0;
        for (auto x : *m_dual)
        {
            if(std::abs(x) > 1e-3)
                nb_active_contacts++;
        }
        return nb_active_contacts;
    }

    template<std::size_t dim>
    void OptimMosek<dim>::tear_down_impl()
    {}

}
#endif

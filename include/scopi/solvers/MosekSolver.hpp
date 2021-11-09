#pragma once

#include "OptimizationSolver.hpp"
#include <fusion.h>

namespace scopi{
    using namespace mosek::fusion;
    using namespace monty;

    template<std::size_t dim>
        class MosekSolver : public OptimizationSolver<dim>
        {
            public:
                MosekSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
                void createMatrixConstraint(std::vector<scopi::neighbor<dim>>& contacts);
                void createMatrixMass();
                int solveOptimizationProbelm(std::vector<scopi::neighbor<dim>>& contacts);
                auto getUadapt();
                auto getWadapt();
                void allocateMemory(std::size_t nc);
                void freeMemory();

            private:
                Matrix::t _Az;
                Matrix::t _A;
                shared_ptr<ndarray<double,1>> _Xlvl;
        };

    template<std::size_t dim>
        MosekSolver<dim>::MosekSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr) : 
            OptimizationSolver<dim>(particles, dt, Nactive, active_ptr, 1 + 2*3*Nactive + 2*3*Nactive, 1)
    {
        this->_c(0) = 1;
    }


    template<std::size_t dim>
        void MosekSolver<dim>::createMatrixConstraint(std::vector<scopi::neighbor<dim>>& contacts)
        {
            // Preallocate
            std::vector<int> A_rows;
            std::vector<int> A_cols;
            std::vector<double> A_values;

            OptimizationSolver<dim>::createMatrixConstraint(contacts, A_rows, A_cols, A_values, 1);

            _A = Matrix::sparse(contacts.size(), 1 + 6*this->_Nactive + 6*this->_Nactive,
                    std::make_shared<ndarray<int, 1>>(A_rows.data(), shape_t<1>({A_rows.size()})),
                    std::make_shared<ndarray<int, 1>>(A_cols.data(), shape_t<1>({A_cols.size()})),
                    std::make_shared<ndarray<double, 1>>(A_values.data(), shape_t<1>({A_values.size()})));
        }

    template<std::size_t dim>
        void MosekSolver<dim>::createMatrixMass()
        {
            std::vector<int> Az_rows;
            std::vector<int> Az_cols;
            std::vector<double> Az_values;

            Az_rows.reserve(6*this->_Nactive*2);
            Az_cols.reserve(6*this->_Nactive*2);
            Az_values.reserve(6*this->_Nactive*2);

            for (std::size_t i=0; i<this->_Nactive; ++i)
            {
                for (std::size_t d=0; d<2; ++d)
                {
                    Az_rows.push_back(3*i + d);
                    Az_cols.push_back(1 + 3*i + d);
                    Az_values.push_back(std::sqrt(this->_mass)); // TODO: add mass into particles
                    // }
                    // for (std::size_t d=0; d<3; ++d)
                    // {
                    Az_rows.push_back(3*i + d);
                    Az_cols.push_back(1 + 6*this->_Nactive + 3*i + d);
                    Az_values.push_back(-1.);
            }

            // for (std::size_t d=0; d<3; ++d)
            // {
            //     Az_rows.push_back(3*_Nactive + 3*i + d);
            //     Az_cols.push_back(1 + 3*_Nactive + 3*i + d);
            //     Az_values.push_back(std::sqrt(moment));
            // }
            Az_rows.push_back(3*this->_Nactive + 3*i + 2);
            Az_cols.push_back(1 + 3*this->_Nactive + 3*i + 2);
            Az_values.push_back(std::sqrt(this->_moment));

            // for (std::size_t d=0; d<3; ++d)
            // {
            //     Az_rows.push_back(3*_Nactive + 3*i + d);
            //     Az_cols.push_back( 1 + 6*_Nactive + 3*_Nactive + 3*i + d);
            //     Az_values.push_back(-1);
            // }
            Az_rows.push_back(3*this->_Nactive + 3*i + 2);
            Az_cols.push_back( 1 + 6*this->_Nactive + 3*this->_Nactive + 3*i + 2);
            Az_values.push_back(-1);
            }

            _Az = Matrix::sparse(6*this->_Nactive, 1 + 6*this->_Nactive + 6*this->_Nactive,
                    std::make_shared<ndarray<int, 1>>(Az_rows.data(), shape_t<1>({Az_rows.size()})),
                    std::make_shared<ndarray<int, 1>>(Az_cols.data(), shape_t<1>({Az_cols.size()})),
                    std::make_shared<ndarray<double, 1>>(Az_values.data(), shape_t<1>({Az_values.size()})));
        }

    template<std::size_t dim>
        int MosekSolver<dim>::solveOptimizationProbelm(std::vector<scopi::neighbor<dim>>& contacts)
        {
            Model::t model = new Model("contact"); auto _M = finally([&]() { model->dispose(); });
            // variables
            Variable::t X = model->variable("X", 1 + 6*this->_Nactive + 6*this->_Nactive);

            // functional to minimize
            auto c_mosek = std::make_shared<ndarray<double, 1>>(this->_c.data(), shape_t<1>({this->_c.shape(0)}));
            model->objective("minvar", ObjectiveSense::Minimize, Expr::dot(c_mosek, X));

            // constraints
            auto D_mosek = std::make_shared<ndarray<double, 1>>(this->_distances.data(), shape_t<1>({this->_distances.shape(0)}));

            Constraint::t qc1 = model->constraint("qc1", Expr::mul(_A, X), Domain::lessThan(D_mosek));
            Constraint::t qc2 = model->constraint("qc2", Expr::mul(_Az, X), Domain::equalsTo(0.));
            Constraint::t qc3 = model->constraint("qc3", Expr::vstack(1, X->index(0), X->slice(1 + 6*this->_Nactive, 1 + 6*this->_Nactive + 6*this->_Nactive)), Domain::inRotatedQCone());
            // model->setSolverParam("intpntCoTolPfeas", 1e-10);
            // model->setSolverParam("intpntTolPfeas", 1.e-10);

            // model->setSolverParam("intpntCoTolDfeas", 1e-6);
            // model->setLogHandler([](const std::string & msg) { std::cout << msg << std::flush; } );
            //solve
            model->solve();

            _Xlvl = X->level();

            int nbIter = model->getSolverIntInfo("intpntIter");

            auto dual = *(qc1->dual());
            int nbActiveContatcs = 0;
            for(auto x : dual) 
            {
                if(std::abs(x) > 1e-3)
                    nbActiveContatcs++;
            }
            std::cout << "Contacts: " << contacts.size() << "  active contacts " << nbActiveContatcs << std::endl;

            return nbIter;
        }

    template<std::size_t dim>
        auto MosekSolver<dim>::getUadapt()
        {
            return xt::adapt(reinterpret_cast<double*>(_Xlvl->raw()+1), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        auto MosekSolver<dim>::getWadapt()
        {
            return xt::adapt(reinterpret_cast<double*>(_Xlvl->raw()+1+3*this->_Nactive), {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        void MosekSolver<dim>::allocateMemory(std::size_t nc)
        {
            std::ignore = nc;
        }

    template<std::size_t dim>
        void MosekSolver<dim>::freeMemory()
        {
        }
}

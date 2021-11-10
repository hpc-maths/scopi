#pragma once

namespace scopi{
    template<std::size_t dim>
        class UzawaSolver : public OptimizationSolver<dim>
    {
        public:
            UzawaSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr);
            void createMatrixConstraint(std::vector<scopi::neighbor<dim>>& contacts);
            void createMatrixMass();
            int solveOptimizationProbelm(std::vector<scopi::neighbor<dim>>& contacts);
            auto getUadapt();
            auto getWadapt();
            void allocateMemory(std::size_t nc);
            void freeMemory();
    };

    template<std::size_t dim>
        UzawaSolver<dim>::UzawaSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr) : 
            OptimizationSolver<dim>(particles, dt, Nactive, active_ptr, 2*3*Nactive + 2*3*Nactive, 0)
    {
    }

    template<std::size_t dim>
        void UzawaSolver<dim>::createMatrixConstraint(std::vector<scopi::neighbor<dim>>& contacts)
        {
        }

    template<std::size_t dim>
        void UzawaSolver<dim>::createMatrixMass()
        {
        }

    template<std::size_t dim>
        int UzawaSolver<dim>::solveOptimizationProbelm(std::vector<scopi::neighbor<dim>>& contacts)
        {
            return 0;
        }

    template<std::size_t dim>
        auto UzawaSolver<dim>::getUadapt()
        {
            std::vector<double> v(this->_Nactive * 3UL, 0.);
            return xt::adapt(v, {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        auto UzawaSolver<dim>::getWadapt()
        {
            std::vector<double> v(this->_Nactive * 3UL, 0.);
            return xt::adapt(v, {this->_Nactive, 3UL});
        }

    template<std::size_t dim>
        void UzawaSolver<dim>::allocateMemory(std::size_t nc)
        {
            std::ignore = nc;
        }

    template<std::size_t dim>
        void UzawaSolver<dim>::freeMemory()
        {
        }
}

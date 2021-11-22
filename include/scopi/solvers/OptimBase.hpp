#pragma once
#include "../crtp.hpp"
#include "../container.hpp"

namespace scopi{
    template <class D, std::size_t dim>
        class OptimBase: public crtp_base<D>
    {
        public:
            void run(const std::vector<scopi::neighbor<dim>>& contacts, const std::size_t nite);
            void freeMemory();
            auto getUadapt();
            auto getWadapt();

        protected:
            OptimBase(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, std::size_t cSize, std::size_t cDec);
            void createMatrixConstraintCoo(const std::vector<scopi::neighbor<dim>>& contacts, std::vector<int>& A_rows, std::vector<int>& A_cols, std::vector<double>& A_values, std::size_t firstCol);
            void cooToCsr(std::vector<int> coo_rows, std::vector<int> coo_cols, std::vector<double> coo_vals, std::vector<int>& csr_rows, std::vector<int>& csr_cols, std::vector<double>& csr_vals);

            scopi::scopi_container<dim>& _particles;
            double _dt;
            std::size_t _Nactive;
            std::size_t _active_ptr;
            double _mass = 1.;
            double _moment = .1;
            xt::xtensor<double, 1> _c;
            std::size_t _cDec;
            xt::xtensor<double, 1> _distances;

        private:
            void createVectorDistances(const std::vector<scopi::neighbor<dim>>& contacts);
            void createVectorC();

            void createMatrixConstraint(const std::vector<scopi::neighbor<dim>>& contacts);
            void createMatrixMass();
            int solveOptimizationProblem(const std::vector<scopi::neighbor<dim>>& contacts);
            void allocateMemory(const std::size_t nc);
            int getNbActiveContacts();
    };

    template<class D, std::size_t dim>
        void OptimBase<D, dim>::run(const std::vector<scopi::neighbor<dim>>& contacts, const std::size_t nite)
        {
            // create mass and inertia matrices
            tic();
            allocateMemory(contacts.size());
            createMatrixConstraint(contacts);
            createMatrixMass();
            createVectorC();
            createVectorDistances(contacts);
            auto duration4 = toc();
            std::cout << "----> CPUTIME : matrices = " << duration4 << std::endl;

            // Solve optimization problem
            std::cout << "----> Create optimization problem " << nite << std::endl;
            tic();
            auto nbIter = solveOptimizationProblem(contacts);
            auto duration5 = toc();
            std::cout << "----> CPUTIME : solve = " << duration5 << std::endl;
            std::cout << "iterations : " << nbIter << std::endl;
            std::cout << "Contacts: " << contacts.size() << "  active contacts " << getNbActiveContacts() << std::endl;
        }


    template<class D, std::size_t dim>
        OptimBase<D, dim>::OptimBase(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, std::size_t cSize, std::size_t cDec) : 
            _particles(particles),
            _dt(dt),
            _Nactive(Nactive),
            _active_ptr(active_ptr),
            _c(xt::zeros<double>({cSize})),
            _cDec(cDec)
            {
            }

    template<class D, std::size_t dim>
        void OptimBase<D, dim>::createVectorDistances(const std::vector<scopi::neighbor<dim>>& contacts)
        {
            // fill vector with distances
            _distances = xt::zeros<double>({contacts.size()});
            for(std::size_t i=0; i<contacts.size(); ++i)
            {
                _distances[i] = contacts[i].dij;
            }
            // std::cout << "distances " << distances << std::endl;
        }

    template<class D, std::size_t dim>
        void OptimBase<D, dim>::createVectorC()
        {
            std::size_t Mdec = _cDec;
            std::size_t Jdec = Mdec + 3*_Nactive;
            for (std::size_t i=0; i<_Nactive; ++i)
            {
                for (std::size_t d=0; d<dim; ++d)
                {
                    _c(Mdec + 3*i + d) = -_mass*_particles.vd()(_active_ptr + i)[d]; // TODO: add mass into particles
                }
                _c(Jdec + 3*i + 2) = -_moment*_particles.desired_omega()(_active_ptr + i);
            }
        }

    template<class D, std::size_t dim>
        void OptimBase<D, dim>::createMatrixConstraintCoo(const std::vector<scopi::neighbor<dim>>& contacts, std::vector<int>& A_rows, std::vector<int>& A_cols, std::vector<double>& A_values, std::size_t firstCol)
        {
            std::size_t u_size = 3*contacts.size()*2;
            std::size_t w_size = 3*contacts.size()*2;
            A_rows.reserve(u_size + w_size);
            A_cols.reserve(u_size + w_size);
            A_values.reserve(u_size + w_size);

            std::size_t ic = 0;
            for (auto &c: contacts)
            {

                for (std::size_t d=0; d<3; ++d)
                {
                    if (c.i >= _active_ptr)
                    {
                        A_rows.push_back(ic);
                        A_cols.push_back(firstCol + (c.i - _active_ptr)*3 + d);
                        A_values.push_back(-_dt*c.nij[d]);
                    }
                    if (c.j >= _active_ptr)
                    {
                        A_rows.push_back(ic);
                        A_cols.push_back(firstCol + (c.j - _active_ptr)*3 + d);
                        A_values.push_back(_dt*c.nij[d]);
                    }
                }

                auto r_i = c.pi - _particles.pos()(c.i);
                auto r_j = c.pj - _particles.pos()(c.j);

                xt::xtensor_fixed<double, xt::xshape<3, 3>> ri_cross, rj_cross;

                if (dim == 2)
                {
                    ri_cross = {{      0,      0, r_i(1)},
                        {      0,      0, -r_i(0)},
                        {-r_i(1), r_i(0),       0}};

                    rj_cross = {{      0,      0,  r_j(1)},
                        {      0,      0, -r_j(0)},
                        {-r_j(1), r_j(0),       0}};
                }
                else
                {
                    ri_cross = {{      0, -r_i(2),  r_i(1)},
                        { r_i(2),       0, -r_i(0)},
                        {-r_i(1),  r_i(0),       0}};

                    rj_cross = {{      0, -r_j(2),  r_j(1)},
                        { r_j(2),       0, -r_j(0)},
                        {-r_j(1),  r_j(0),       0}};
                }

                auto Ri = scopi::rotation_matrix<3>(_particles.q()(c.i));
                auto Rj = scopi::rotation_matrix<3>(_particles.q()(c.j));

                if (c.i >= _active_ptr)
                {
                    std::size_t ind_part = c.i - _active_ptr;
                    auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                    for (std::size_t ip=0; ip<3; ++ip)
                    {
                        A_rows.push_back(ic);
                        A_cols.push_back(firstCol + 3*_Nactive + 3*ind_part + ip);
                        A_values.push_back(_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                }

                if (c.j >= _active_ptr)
                {
                    std::size_t ind_part = c.j - _active_ptr;
                    auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                    for (std::size_t ip=0; ip<3; ++ip)
                    {
                        A_rows.push_back(ic);
                        A_cols.push_back(firstCol + 3*_Nactive + 3*ind_part + ip);
                        A_values.push_back(-_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                }

                ++ic;
            }
        }

    template<class D, std::size_t dim>
        void OptimBase<D, dim>::cooToCsr(std::vector<int> coo_rows, std::vector<int> coo_cols, std::vector<double> coo_vals, std::vector<int>& csr_rows, std::vector<int>& csr_cols, std::vector<double>& csr_vals)
        {
            // https://www-users.cse.umn.edu/~saad/software/SPARSKIT/
            std::size_t nrow = 6*this->_Nactive;
            std::size_t nnz = coo_vals.size();
            csr_rows.resize(nrow+1);
            std::fill(csr_rows.begin(), csr_rows.end(), 0);
            csr_cols.resize(nnz);
            csr_vals.resize(nnz);

            // determine row-lengths.
            for(std::size_t k = 0; k < nnz; ++k)
            {
                csr_rows[coo_rows[k]]++;
            }

            // starting position of each row..
            {
                int k = 0;
                for(std::size_t j = 0; j < nrow+1; ++j)
                {
                    int k0 = csr_rows[j];
                    csr_rows[j] = k;
                    k += k0;
                }
            }

            // go through the structure  once more. Fill in output matrix.
            for(std::size_t k = 0; k < nnz; ++k)
            {
                int i = coo_rows[k];
                int j = coo_cols[k];
                double x = coo_vals[k];
                int iad = csr_rows[i];
                csr_vals[iad] = x;
                csr_cols[iad] = j;
                csr_rows[i] = iad+1;
            }

            // shift back iao
            for(std::size_t j = nrow; j >= 1; --j)
            {
                csr_rows[j] = csr_rows[j-1];
            }
            csr_rows[0] = 0;
        }

    template<class D, std::size_t dim>
        void OptimBase<D, dim>::createMatrixConstraint(const std::vector<scopi::neighbor<dim>>& contacts)
        {
            this->derived_cast().createMatrixConstraint_impl(contacts);
        }
    template<class D, std::size_t dim>
        void OptimBase<D, dim>::createMatrixMass()
        {
            this->derived_cast().createMatrixMass_impl();
        }

    template<class D, std::size_t dim>
        int OptimBase<D, dim>::solveOptimizationProblem(const std::vector<scopi::neighbor<dim>>& contacts)
        {
            return this->derived_cast().solveOptimizationProblem_impl(contacts);
        }

    template<class D, std::size_t dim>
        auto OptimBase<D, dim>::getUadapt()
        {
            return this->derived_cast().getUadapt_impl();
        }

    template<class D, std::size_t dim>
        auto OptimBase<D, dim>::getWadapt()
        {
            return this->derived_cast().getWadapt_impl();
        }

    template<class D, std::size_t dim>
        void OptimBase<D, dim>::allocateMemory(const std::size_t nc)
        {
            this->derived_cast().allocateMemory_impl(nc);
        }

    template<class D, std::size_t dim>
        void OptimBase<D, dim>::freeMemory()
        {
            this->derived_cast().freeMemory_impl();
        }

    template<class D, std::size_t dim>
        int OptimBase<D, dim>::getNbActiveContacts()
        {
            return this->derived_cast().getNbActiveContacts_impl();
        }
}


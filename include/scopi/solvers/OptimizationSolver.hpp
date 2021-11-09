#pragma once

namespace scopi{
    template<std::size_t dim>
        class OptimizationSolver
        {
            public:
                void createVectorDistances(std::vector<scopi::neighbor<dim>>& contacts);

            protected:
                OptimizationSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, std::size_t cSize);
                xt::xtensor<double, 1> createVectorC();
                void createMatrixConstraint(std::vector<scopi::neighbor<dim>>& contacts, std::vector<int>& A_rows, std::vector<int>& A_cols, std::vector<double>& A_values, std::size_t firstCol);

                scopi::scopi_container<dim>& _particles;
                double _dt;
                std::size_t _Nactive;
                std::size_t _active_ptr;
                double _mass = 1.;
                double _moment = .1;
                xt::xtensor<double, 1> _c;
                xt::xtensor<double, 1> _distances;
        };

    template<std::size_t dim>
        OptimizationSolver<dim>::OptimizationSolver(scopi::scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, std::size_t cSize) : 
            _particles(particles),
            _dt(dt),
            _Nactive(Nactive),
            _active_ptr(active_ptr),
            _c(xt::zeros<double>({cSize}))
    {
    }

    template<std::size_t dim>
        void OptimizationSolver<dim>::createVectorDistances(std::vector<scopi::neighbor<dim>>& contacts)
        {
            // fill vector with distances
            _distances = xt::zeros<double>({contacts.size()});
            for(std::size_t i=0; i<contacts.size(); ++i)
            {
                _distances[i] = contacts[i].dij;
            }
            // std::cout << "distances " << distances << std::endl;
        }


    template<std::size_t dim>
        xt::xtensor<double, 1> OptimizationSolver<dim>::createVectorC()
        {
            xt::xtensor<double, 1> c = xt::zeros<double>({2*3*_Nactive});
            std::size_t Mdec = 0;
            std::size_t Jdec = Mdec + 3*_Nactive;
            for (std::size_t i=0; i<_Nactive; ++i)
            {
                for (std::size_t d=0; d<dim; ++d)
                {
                    c(Mdec + 3*i + d) = -_mass*_particles.vd()(_active_ptr + i)[d]; // TODO: add mass into particles
                }
                c(Jdec + 3*i + 2) = -_moment*_particles.desired_omega()(_active_ptr + i);
            }
            return c;
        }

    template<std::size_t dim>
        void OptimizationSolver<dim>::createMatrixConstraint(std::vector<scopi::neighbor<dim>>& contacts, std::vector<int>& A_rows, std::vector<int>& A_cols, std::vector<double>& A_values, std::size_t firstCol)
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

}


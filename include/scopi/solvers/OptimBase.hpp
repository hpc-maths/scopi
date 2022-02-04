#pragma once
#include "../crtp.hpp"
#include "../container.hpp"
#include "../objects/neighbor.hpp"

namespace scopi{
    template <class D, std::size_t dim>
    class OptimBase: public crtp_base<D>
    {
    public:
        void run(const std::vector<neighbor<dim>>& contacts, const std::size_t nite);
        void free_memory();
        auto get_uadapt();
        auto get_wadapt();

    protected:
        OptimBase(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, std::size_t cSize, std::size_t c_dec);
        void create_matrix_constraint_coo(const std::vector<neighbor<dim>>& contacts, std::vector<int>& A_rows, std::vector<int>& A_cols, std::vector<double>& A_values, std::size_t firstCol);

        scopi_container<dim>& m_particles;
        double m_dt;
        std::size_t m_Nactive;
        std::size_t m_active_ptr;
        double m_mass;
        double m_moment;
        xt::xtensor<double, 1> m_c;
        std::size_t m_c_dec;
        xt::xtensor<double, 1> m_distances;

    private:
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts);
        void create_vector_c();

        void create_matrix_constraint(const std::vector<neighbor<dim>>& contacts);
        void create_matrix_mass();
        int solve_optimization_problem(const std::vector<neighbor<dim>>& contacts);
        void allocate_memory(const std::size_t nc);
        int get_nb_active_contacts();
    };

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::run(const std::vector<neighbor<dim>>& contacts, const std::size_t)
    {
        tic();
        create_vector_c();
        create_vector_distances(contacts);
        allocate_memory(contacts.size());
        create_matrix_constraint(contacts);
        create_matrix_mass();
        auto duration4 = toc();
        // std::cout << "----> CPUTIME : matrices = " << duration4 << std::endl;

        // Solve optimization problem
        // std::cout << "----> Create optimization problem " << nite << std::endl;
        // tic();
        auto nbIter = solve_optimization_problem(contacts);
        // auto duration5 = toc();
        // std::cout << "----> CPUTIME : solve = " << duration5 << std::endl;
        // std::cout << "iterations : " << nbIter << std::endl;
        // std::cout << "Contacts: " << contacts.size() << "  active contacts " << getNbActiveContacts() << std::endl;
    }


    template<class D, std::size_t dim>
    OptimBase<D, dim>::OptimBase(scopi_container<dim>& particles, double dt, std::size_t Nactive, std::size_t active_ptr, std::size_t cSize, std::size_t c_dec)
    : m_particles(particles)
    , m_dt(dt)
    , m_Nactive(Nactive)
    , m_active_ptr(active_ptr)
    , m_mass(1.)
    , m_moment(0.1)
    , m_c(xt::zeros<double>({cSize}))
    , m_c_dec(c_dec)
    {}

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::create_vector_distances(const std::vector<neighbor<dim>>& contacts)
    {
        m_distances = xt::zeros<double>({contacts.size()});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            m_distances[i] = contacts[i].dij;
        }
        // std::cout << "distances " << distances << std::endl;
    }

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::create_vector_c()
    {
        std::size_t mass_dec = m_c_dec;
        std::size_t moment_dec = mass_dec + 3*m_Nactive;
        for (std::size_t i = 0; i < m_Nactive; ++i)
        {
            for (std::size_t d = 0; d < dim; ++d)
            {
                m_c(mass_dec + 3*i + d) = -m_mass*m_particles.vd()(m_active_ptr + i)[d]; // TODO: add mass into particles
            }
            m_c(moment_dec + 3*i + 2) = -m_moment*m_particles.desired_omega()(m_active_ptr + i);
        }
    }

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::create_matrix_constraint_coo(const std::vector<neighbor<dim>>& contacts, std::vector<int>& A_rows, std::vector<int>& A_cols, std::vector<double>& A_values, std::size_t firstCol)
    {
        std::size_t u_size = 3*contacts.size()*2;
        std::size_t w_size = 3*contacts.size()*2;
        A_rows.reserve(u_size + w_size);
        A_cols.reserve(u_size + w_size);
        A_values.reserve(u_size + w_size);

        std::size_t ic = 0;
        for (auto &c: contacts)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                if (c.i >= m_active_ptr)
                {
                    A_rows.push_back(ic);
                    A_cols.push_back(firstCol + (c.i - m_active_ptr)*3 + d);
                    A_values.push_back(-m_dt*c.nij[d]);
                }
                if (c.j >= m_active_ptr)
                {
                    A_rows.push_back(ic);
                    A_cols.push_back(firstCol + (c.j - m_active_ptr)*3 + d);
                    A_values.push_back(m_dt*c.nij[d]);
                }
            }

            auto r_i = c.pi - m_particles.pos()(c.i);
            auto r_j = c.pj - m_particles.pos()(c.j);

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

            auto Ri = rotation_matrix<3>(m_particles.q()(c.i));
            auto Rj = rotation_matrix<3>(m_particles.q()(c.j));

            if (c.i >= m_active_ptr)
            {
                std::size_t ind_part = c.i - m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    A_rows.push_back(ic);
                    A_cols.push_back(firstCol + 3*m_Nactive + 3*ind_part + ip);
                    A_values.push_back(m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }

            if (c.j >= m_active_ptr)
            {
                std::size_t ind_part = c.j - m_active_ptr;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    A_rows.push_back(ic);
                    A_cols.push_back(firstCol + 3*m_Nactive + 3*ind_part + ip);
                    A_values.push_back(-m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }

            ++ic;
        }
    }

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::create_matrix_constraint(const std::vector<neighbor<dim>>& contacts)
    {
        this->derived_cast().create_matrix_constraint_impl(contacts);
    }

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::create_matrix_mass()
    {
        this->derived_cast().create_matrix_mass_impl();
    }

    template<class D, std::size_t dim>
    int OptimBase<D, dim>::solve_optimization_problem(const std::vector<neighbor<dim>>& contacts)
    {
        return this->derived_cast().solve_optimization_problem_impl(contacts);
    }

    template<class D, std::size_t dim>
    auto OptimBase<D, dim>::get_uadapt()
    {
        return this->derived_cast().get_uadapt_impl();
    }

    template<class D, std::size_t dim>
    auto OptimBase<D, dim>::get_wadapt()
    {
        return this->derived_cast().get_wadapt_impl();
    }

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::allocate_memory(const std::size_t nc)
    {
        this->derived_cast().allocate_memory_impl(nc);
    }

    template<class D, std::size_t dim>
    void OptimBase<D, dim>::free_memory()
    {
        this->derived_cast().free_memory_impl();
    }

    template<class D, std::size_t dim>
    int OptimBase<D, dim>::get_nb_active_contacts()
    {
        return this->derived_cast().get_nb_active_contacts_impl();
    }
}


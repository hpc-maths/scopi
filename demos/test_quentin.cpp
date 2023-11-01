#include <cstdlib>
#include <scopi/container.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <vector>
#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>

#include <scopi/contact/contact_brute_force.hpp>
#include <scopi/matrix/velocities.hpp>

namespace scopi
{

    template <class Contacts, class Particles>
    class Gradient
    {
      public:

        static constexpr std::size_t dim = Particles::dim;

        Gradient(double dt, const Contacts& contacts, const Particles& particles)
            : m_dt(dt)
            , m_contacts(contacts)
            , m_particles(particles)
            , m_A(contacts, particles)
            , m_At(contacts, particles)
            , m_work(xt::zeros<double>({3 * contacts.size()}))
        {
            static constexpr std::size_t dim = Particles::dim;
            xt::xtensor<double, 1> U         = xt::zeros<double>({6 * particles.nb_active()});
            m_invM                           = xt::zeros<double>({6 * particles.nb_active()});
            xt::xtensor<double, 1> normal    = xt::zeros<double>({3 * contacts.size()});

            D d(contacts);

            std::size_t offset = 3 * particles.nb_active();
            for (std::size_t i = 0; i < particles.nb_active(); ++i)
            {
                xt::view(U, xt::range(3 * i, 3 * i + dim))      = particles.vd()[particles.nb_inactive() + i];
                xt::view(m_invM, xt::range(3 * i, 3 * i + dim)) = 1. / particles.m()[particles.nb_inactive() + i];
                U[offset + 3 * i + 2]                           = particles.desired_omega()[particles.nb_inactive() + i];
                m_invM[offset + 3 * i + 2]                      = particles.j()[particles.nb_inactive() + i];
            }
            for (std::size_t i = 0; i < contacts.size(); ++i)
            {
                xt::view(normal, xt::range(3 * i, 3 * i + dim)) = contacts[i].nij;
            }
            m_C = d.mat_mult(normal) + dt * m_A.mat_mult(U);
            std::cout << "m_C " << d.mat_mult(normal) << " " << dt * m_A.mat_mult(U) << " " << xt::linalg::dot(m_C, normal) << std::endl;
        }

        auto& operator()(const xt::xtensor<double, 1>& lambda)
        {
            xt::noalias(m_work) = m_dt * m_dt * m_A.mat_mult(m_invM * m_At.mat_mult(lambda)) + m_C;
            return m_work;
        }

        const auto& contacts() const
        {
            return m_contacts;
        }

        auto velocities(const xt::xtensor<double, 1>& lambda)
        {
            return xt::eval(m_invM * m_At.mat_mult(lambda));
        }

        std::size_t size() const
        {
            return m_contacts.size();
        }

      private:

        double m_dt;
        const Contacts& m_contacts;
        const Particles& m_particles;
        A<Contacts, Particles> m_A;
        AT<Contacts, Particles> m_At;
        xt::xtensor<double, 1> m_invM;
        xt::xtensor<double, 1> m_C;
        xt::xtensor<double, 1> m_work;
    };

    template <class Contacts, class D>
    class LagrangeMultiplierBase : public crtp_base<D>
    {
      public:

        auto get()
        {
            return this->derived_cast().get();
        }

        void set(const xt::xtensor<double, 1>& lambda)
        {
            this->derived_cast().set(lambda);
        }

        auto& projection()
        {
            return this->derived_cast().projection();
        }

      protected:

        LagrangeMultiplierBase(const Contacts& contacts)
            : m_contacts(contacts)
        {
        }

        const Contacts& m_contacts;
    };

    template <std::size_t dim, class Type, class Contacts>
    class LagrangeMultiplier;

    //  : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<Type, Contacts>>
    // {
    //   public:

    //     static constexpr std::size_t dim = Contacts::dim;

    //     using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<Type, Contacts>>;

    //     LagrangeMultiplier(const Contacts& contacts)
    //         : base(contacts)
    //         , m_work(xt::zeros<double>({3 * contacts.size()}))
    //     {
    //     }

    //   private:

    //     xt::xtensor<double, 1> m_work;
    // };

    struct WithoutFriction
    {
    };

    struct Viscous
    {
    };

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, WithoutFriction, Contacts>
        : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, WithoutFriction, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, WithoutFriction, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts)
            : base(contacts)
            , m_lambda(xt::zeros<double>({contacts.size()}))
            , m_work(xt::zeros<double>({3 * contacts.size()}))
        {
        }

        auto& get()
        {
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                xt::view(this->m_work, xt::range(3 * i, 3 * i + dim)) = m_lambda[i] * this->m_contacts[i].nij;
            }

            return this->m_work;
        }

        auto& get(const xt::xtensor<double, 1>& lambda)
        {
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                xt::view(this->m_work, xt::range(3 * i, 3 * i + dim)) = lambda[i] * this->m_contacts[i].nij;
            }

            return this->m_work;
        }

        void set(const xt::xtensor<double, 1>& lambda)
        {
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                m_lambda[i] = xt::linalg::dot(xt::view(lambda, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
            }
        }

        const auto& lambda() const
        {
            return m_lambda;
        }

        auto& lambda()
        {
            return m_lambda;
        }

        auto& projection()
        {
            m_lambda = xt::maximum(m_lambda, 0.);
            return m_lambda;
        }

      private:

        xt::xtensor<double, 1> m_lambda;
        xt::xtensor<double, 1> m_work;
    };

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, Viscous, Contacts> : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Viscous, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Viscous, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts)
            : base(contacts)
            , m_lambda(xt::zeros<double>({contacts.size()}))
            , m_work(xt::zeros<double>({3 * contacts.size()}))
            , m_gamma(xt::zeros<double>({contacts.size()}))
            , m_gamma_tol(1e-6)
            , m_gamma_min(-2)
        {
        }

        auto& get()
        {
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                if (m_gamma[i] < -m_gamma_tol)
                {
                    xt::view(this->m_work, xt::range(3 * i, 3 * i + dim)) = (m_lambda[i] - m_lambda[i + m_contacts.size()])
                                                                          * this->m_contacts[i].nij;
                }
                else
                {
                    xt::view(this->m_work, xt::range(3 * i, 3 * i + dim)) = m_lambda[i] * this->m_contacts[i].nij;
                }
            }

            return this->m_work;
        }

        auto& get(const xt::xtensor<double, 1>& lambda)
        {
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                xt::view(this->m_work, xt::range(3 * i, 3 * i + dim)) = lambda[i] * this->m_contacts[i].nij;
            }

            return this->m_work;
        }

        void set(const xt::xtensor<double, 1>& lambda)
        {
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                auto lambda_tmp = xt::linalg::dot(xt::view(lambda, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
                m_gamma[i]      = std::min(std::max(m_gamma[i] - dt * lambda, m_gamma_min), 0);
                if (m_gamma[i] < -m_gamma_tol)
                {
                }
                else
                {
                    m_lambda[i] = lambda_tmp;
                }
                m_lambda_plus[i] = xt::linalg::dot(xt::view(lambda, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
            }
        }

        const auto& lambda() const

        {
            return m_lambda;
        }

        auto& lambda()
        {
            return m_lambda;
        }

        auto& projection()
        {
            m_lambda = xt::maximum(m_lambda, 0.);
            return m_lambda;
        }

      private:

        xt::xtensor<double, 1> m_lambda;
        xt::xtensor<double, 1> m_work;
        xt::xtensor<double, 1> m_gamma;
        double m_gamma_tol;
        double m_gamma_min;
    };

    template <std::size_t dim, class Type, class Contacts>
    auto make_lagrange_multplier(const Contacts& contacts)
    {
        return LagrangeMultiplier<dim, Type, std::decay_t<Contacts>>(contacts);
    }

    template <class Gradient, class Lambda>
    auto pgd(Gradient& gradient, Lambda& lambda, double alpha, double tolerance = 1e-6, std::size_t max_ite = 10000)
    {
        std::size_t ite = 0;

        double residual = 1;

        xt::xtensor<double, 1> lambda_n = lambda.lambda();

        double r0 = xt::linalg::norm(lambda_n);
        if (r0 == 0.)
        {
            r0 = 1.;
        }

        while (residual > tolerance && ite < max_ite)
        {
            auto& la = lambda.get();

            auto& dG = gradient(la);
            la -= alpha * dG;

            lambda.set(la);
            auto& lambda_np1 = lambda.projection();

            residual = xt::linalg::norm(lambda_np1 - lambda_n) / r0;
            lambda_n = lambda_np1;
            std::cout << fmt::format("ite = {}, |DG| = {}, |lambda| = {}, residual = {}", ite, xt::linalg::norm(dG), lambda_n[0], residual)
                      << std::endl;
            ite++;
        }
        return lambda.lambda();
    }

    template <class Gradient, class Lambda>
    auto apgd(Gradient& gradient, Lambda& lambda, double alpha, double tolerance = 1e-6, std::size_t max_ite = 10000)
    {
        std::size_t ite = 0;

        xt::xtensor<double, 1> theta_n   = xt::ones<double>({gradient.size()});
        xt::xtensor<double, 1> theta_np1 = xt::ones<double>({gradient.size()});

        xt::xtensor<double, 1> y_n   = xt::zeros<double>({gradient.size()});
        xt::xtensor<double, 1> y_np1 = xt::zeros<double>({gradient.size()});

        double residual = 1;

        xt::xtensor<double, 1> lambda_n = lambda.lambda();

        double r0 = xt::linalg::norm(lambda_n);
        if (r0 == 0.)
        {
            r0 = 1.;
        }

        while (residual > tolerance && ite < max_ite)
        {
            auto& la = lambda.get(y_n);

            auto& dG = gradient(la);

            // if (dynamic_descent)
            // {
            //     double lipsch = 1./alpha;
            //     auto& la = lambda.get(y_n);

            //     while()

            // }
            la -= alpha * dG;

            double norm_dG = xt::linalg::norm(dG);
            double norm_la = xt::linalg::norm(la);
            std::cout << fmt::format("ite = {}, |DG| = {}, |lambda| = {}, residual = {}", ite, norm_dG, norm_la, residual) << std::endl;
            if (norm_dG < tolerance || norm_la < tolerance)
            {
                break;
            }

            lambda.set(la);
            auto& lambda_np1 = lambda.projection();

            theta_np1 = 0.5 * (theta_n * xt::sqrt(4 + theta_n * theta_n) - theta_n * theta_n);
            auto beta = theta_n * (1 - theta_n) / (theta_n * theta_n + theta_np1);
            y_np1     = lambda_np1 + beta * (lambda_np1 - lambda_n);

            residual = xt::linalg::norm(lambda_np1 - lambda_n) / r0;
            lambda_n = lambda_np1;
            std::swap(theta_n, theta_np1);
            std::swap(y_n, y_np1);

            ite++;
        }
        return lambda.lambda();
    }

    template <std::size_t dim>
    class NewOptimSolver
    {
      public:

        NewOptimSolver(std::size_t, double dt, const scopi_container<dim>&)
            : m_dt(dt)
        {
        }

        void extra_steps_before_solve(const std::vector<neighbor<dim>>&)
        {
            m_should_solve = true;
        }

        void extra_steps_after_solve(const std::vector<neighbor<dim>>&)
        {
            m_should_solve = false;
        }

        void run(const scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts, std::size_t)
        {
            std::size_t active_offset = particles.nb_inactive();
            m_u.resize({particles.nb_active(), 3});
            m_omega.resize({particles.nb_active(), 3});
            m_omega.fill(0);

            if (contacts.size() != 0)
            {
                auto gradient = Gradient(m_dt, contacts, particles);
                auto lambda   = make_lagrange_multplier<dim, WithoutFriction>(contacts);
                // pgd(gradient, lambda, 0.05, 1e-6, 100);
                auto result = apgd(gradient, lambda, 0.05, 1e-6, 10000);

                auto velocities = gradient.velocities(lambda.get(result));
                // std::cout << velocities << std::endl;
                for (std::size_t i = 0; i < particles.nb_active(); ++i)
                {
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        m_u(i, d) = particles.vd()(i + active_offset)(d) + m_dt * velocities(i * 3 + d);
                    }
                    m_omega(i, 2) = particles.desired_omega()(i + active_offset) + m_dt * velocities((i + particles.nb_active()) * 3 + 2);
                }
            }
            else
            {
                for (std::size_t i = 0; i < particles.nb_active(); ++i)
                {
                    for (std::size_t d = 0; d < dim; ++d)
                    {
                        m_u(i, d) = particles.vd()(i + active_offset)(d);
                    }
                    m_omega(i, 2) = particles.desired_omega()(i + active_offset);
                }
            }
        }

        const auto& get_uadapt() const
        {
            return m_u;
        }

        const auto& get_wadapt() const
        {
            return m_omega;
        }

        bool should_solve()
        {
            return m_should_solve;
        }

      private:

        double m_dt;
        bool m_should_solve;
        xt::xtensor<double, 2> m_u;
        xt::xtensor<double, 2> m_omega;
    };

    // template <class Type, class Particles>
    // void solver(double dt, double Tf, Particles& p)
    // {
    //     double dt = 0.05;
    //     double t  = 0;

    //     contact_brute_force cont;
    //     while (t != Tf)
    //     {
    //         t += dt;
    //         if (t > Tf)
    //         {
    //             dt += Tf - t;
    //             t = Tf;
    //         }
    //         auto contacts = cont.run(particles, 0);
    //         auto gradient = Gradient(dt, contacts, particles);
    //         auto lambda   = make_lagrange_multplier<dim, scopi::WithoutFriction>(contacts);

    //         scopi::apgd(gradient, lambda, 0.05, 1e-6, 100);
    //     }
    // }

} // namespace scopi

int main()

{
    std::setprecision(12);
    xt::print_options::set_precision(12);
    // plog::init(plog::info, "quentin.log");

    constexpr std::size_t dim = 2;

    scopi::scopi_container<dim> particles;

    // SPHERE - PLAN CASE
    double PI = xt::numeric_constants<double>::PI;
    scopi::plan<dim> plan(
        {
            {0., 0.}
    },
        PI / 4.);
    scopi::sphere<dim> sphere(
        {
            {0, 1.5}
    },
        0.5);
    particles.push_back(plan, scopi::property<dim>().deactivate());
    particles.push_back(sphere,
                        scopi::property<dim>().mass(1).desired_velocity({
                            {0, -1}
    }));
    double Tf = 2.5;

    // SPHERE - SPHERE CASE

    // scopi::sphere<dim> s1(
    //     {
    //         {1.5, 1.6}
    // },
    //     0.5);

    // scopi::sphere<dim> s2(
    //     {
    //         {3.5, 1.4}
    // },
    //     0.5);
    // particles.push_back(s1,
    //                     scopi::property<dim>().mass(1).desired_velocity({
    //                         {1, 0}
    // }));
    // particles.push_back(s2,
    //                     scopi::property<dim>().mass(1).desired_velocity({
    //                         {-1, 0}
    // }));

    double dt = 0.05;

    using optim_solver = scopi::NewOptimSolver<dim>;
    scopi::ScopiSolver<dim, optim_solver> solver(particles, dt);
    solver.run(Tf / dt);
    return 0;
}

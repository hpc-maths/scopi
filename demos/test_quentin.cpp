#include <cstdlib>
#include <scopi/container.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/property.hpp>
#include <scopi/solver.hpp>
#include <scopi/vap/vap_fpd.hpp>
#include <vector>
#include <xtensor/xmath.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xnorm.hpp>

#include <scopi/contact/contact_brute_force.hpp>
#include <scopi/matrix/velocities.hpp>

namespace scopi
{

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

    struct Friction
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
        {
        }

        auto global2local(const xt::xtensor<double, 1>& x) const
        {
            assert(x.size() == 3 * this->m_contacts.size());
            xt::xtensor<double, 1> out = xt::empty<double>({this->m_contacts.size()});
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                out[i] = xt::linalg::dot(xt::view(x, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
            }
            return out;
        }

        auto local2global(const xt::xtensor<double, 1>& x) const
        {
            assert(x.size() == this->m_contacts.size());
            xt::xtensor<double, 1> out = xt::empty<double>({3 * this->m_contacts.size()});
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                xt::view(out, xt::range(3 * i, 3 * i + dim)) = x[i] * this->m_contacts[i].nij;
            }
            return out;
        }

        std::size_t size() const
        {
            return this->m_contacts.size();
        }

        void projection(xt::xtensor<double, 1>& lambda) const
        {
            lambda = xt::maximum(lambda, 0.);
        }
    };

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, Viscous, Contacts> : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Viscous, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Viscous, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts, const xt::xtensor<double, 1>& gamma, double tol)
            : base(contacts)
            , m_gamma(gamma)
            , m_tol(tol)
        {
        }

        auto global2local(const xt::xtensor<double, 1>& x) const
        {
            // std::cout << "in global2local " << m_gamma << std::endl;
            assert(x.size() == 3 * this->m_contacts.size());
            xt::xtensor<double, 1> out = xt::empty<double>({size()});
            std::size_t next_gamma_neg = this->m_contacts.size();

            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                if (m_gamma[i] < -m_tol)
                {
                    out[i]                = xt::linalg::dot(xt::view(x, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
                    out[next_gamma_neg++] = -xt::linalg::dot(xt::view(x, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
                }
                else
                {
                    out[i] = xt::linalg::dot(xt::view(x, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
                }
            }
            return out;
        }

        auto local2global(const xt::xtensor<double, 1>& x) const
        {
            // std::cout << "in local2global " << m_gamma << std::endl;

            assert(x.size() == size());
            xt::xtensor<double, 1> out = xt::empty<double>({3 * this->m_contacts.size()});
            std::size_t next_gamma_neg = this->m_contacts.size();
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                if (m_gamma[i] < -m_tol)
                {
                    xt::view(out, xt::range(3 * i, 3 * i + dim)) = (x[i] - x[next_gamma_neg++]) * this->m_contacts[i].nij;
                }
                else
                {
                    xt::view(out, xt::range(3 * i, 3 * i + dim)) = x[i] * this->m_contacts[i].nij;
                }
            }
            return out;
        }

        std::size_t size() const
        {
            std::size_t s = 0;
            for (auto& g : m_gamma)
            {
                if (g < -m_tol)
                {
                    s += 2;
                }
                else
                {
                    s++;
                }
            }
            return s;
        }

        void projection(xt::xtensor<double, 1>& lambda) const
        {
            assert(lambda.size() == size());
            lambda = xt::maximum(lambda, 0.);
        }

      private:

        xt::xtensor<double, 1> m_gamma;
        const double m_tol;
    };

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, Friction, Contacts> : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Friction, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Friction, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts, double mu)
            : base(contacts)
            , m_mu(mu)
        {
        }

        auto global2local(const xt::xtensor<double, 1>& x) const
        {
            return x;
        }

        auto local2global(const xt::xtensor<double, 1>& x) const
        {
            return x;
        }

        std::size_t size() const
        {
            return 3 * this->m_contacts.size();
        }

        void projection(xt::xtensor<double, 1>& lambda) const
        {
            assert(lambda.size() == size());
            for (std::size_t i = 0, row = 0; i < this->m_contacts.size(); ++i, row += 3)
            {
                auto lambda_i = xt::view(lambda, xt::range(row, row + dim));
                auto lambda_n = xt::linalg::dot(lambda_i, this->m_contacts[i].nij)[0];
                auto lambda_t = xt::eval(lambda_i - lambda_n * this->m_contacts[i].nij);
                auto norm     = xt::norm_l2(lambda_t)[0];
                if (norm > m_mu * lambda_n)
                {
                    if (lambda_n <= -m_mu * norm)
                    {
                        lambda_i = 0;
                    }
                    else
                    {
                        auto new_norm     = (m_mu * m_mu) * (norm + lambda_n / m_mu) / (m_mu * m_mu + 1);
                        auto new_lambda_n = new_norm / m_mu;
                        lambda_t /= norm;
                        lambda_i = new_norm * lambda_t + new_lambda_n * this->m_contacts[i].nij;
                    }
                }
            }
        }

      private:

        const double m_mu;
    };

    template <class Problem, class Contacts, class Particles>
    class Gradient
    {
      public:

        static constexpr std::size_t dim = Particles::dim;

        Gradient(double dt,
                 const Contacts& contacts,
                 const Particles& particles,
                 const LagrangeMultiplier<Particles::dim, Problem, Contacts>& lagrange)
            : m_dt(dt)
            , m_lagrange(lagrange)
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
                m_invM[offset + 3 * i + 2]                      = 1. / particles.j()[particles.nb_inactive() + i];
            }
            for (std::size_t i = 0; i < contacts.size(); ++i)
            {
                xt::view(normal, xt::range(3 * i, 3 * i + dim)) = contacts[i].nij;
            }
            m_C = d.mat_mult(normal) + dt * m_A.mat_mult(U);
            // std::cout << "C " << m_C << std::endl;
        }

        auto operator()(const xt::xtensor<double, 1>& lambda) const
        {
            // std::cout << "C_vector " << m_C << std::endl;
            // std::cout << "C " << m_lagrange.global2local(m_C) << std::endl;
            return m_lagrange.global2local(m_dt * m_dt * m_A.mat_mult(m_invM * m_At.mat_mult(m_lagrange.local2global(lambda))) + m_C);
        }

        double min_f(const xt::xtensor<double, 1>& lambda) const
        {
            auto lambda_global = m_lagrange.local2global(lambda);
            return xt::linalg::dot(lambda_global, 0.5 * m_A.mat_mult(lambda_global) + m_C)[0];
        }

        const auto& contacts() const
        {
            return m_contacts;
        }

        auto velocities(const xt::xtensor<double, 1>& lambda)
        {
            return xt::eval(m_invM * m_At.mat_mult(m_lagrange.local2global(lambda)));
        }

        void projection(xt::xtensor<double, 1>& lambda) const
        {
            m_lagrange.projection(lambda);
        }

        std::size_t size() const
        {
            return m_lagrange.size();
        }

      private:

        double m_dt;
        const LagrangeMultiplier<Particles::dim, Problem, Contacts>& m_lagrange;
        const Contacts& m_contacts;
        const Particles& m_particles;
        const A<Contacts, Particles> m_A;
        const AT<Contacts, Particles> m_At;
        xt::xtensor<double, 1> m_invM;
        xt::xtensor<double, 1> m_C;
        xt::xtensor<double, 1> m_work;
    };

    template <class Problem, class Contacts, class Particles, class Lagrange>
    auto make_gradient(double dt, const Contacts& contacts, const Particles& particles, const Lagrange& lagrange)
    {
        return Gradient<Problem, Contacts, Particles>(dt, contacts, particles, lagrange);
    }

    // template <std::size_t dim_, class Contacts>
    // class LagrangeMultiplier<dim_, Viscous, Contacts> : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Viscous,
    // Contacts>>
    // {
    //   public:

    //     static constexpr std::size_t dim = dim_;

    //     using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Viscous, Contacts>>;

    //     LagrangeMultiplier(const Contacts& contacts)
    //         : base(contacts)
    //         , m_lambda(xt::zeros<double>({contacts.size()}))
    //         , m_work(xt::zeros<double>({3 * contacts.size()}))
    //         , m_gamma(xt::zeros<double>({contacts.size()}))
    //         , m_gamma_tol(1e-6)
    //         , m_gamma_min(-2)
    //     {
    //     }

    //     auto& get()
    //     {
    //         for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
    //         {
    //             if (m_gamma[i] < -m_gamma_tol)
    //             {
    //                 xt::view(this->m_work, xt::range(3 * i, 3 * i + dim)) = (m_lambda[i] - m_lambda[i + m_contacts.size()])
    //                                                                       * this->m_contacts[i].nij;
    //             }
    //             else
    //             {
    //                 xt::view(this->m_work, xt::range(3 * i, 3 * i + dim)) = m_lambda[i] * this->m_contacts[i].nij;
    //             }
    //         }

    //         return this->m_work;
    //     }

    //     auto& get(const xt::xtensor<double, 1>& lambda)
    //     {
    //         for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
    //         {
    //             xt::view(this->m_work, xt::range(3 * i, 3 * i + dim)) = lambda[i] * this->m_contacts[i].nij;
    //         }

    //         return this->m_work;
    //     }

    //     void set(const xt::xtensor<double, 1>& lambda)
    //     {
    //         for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
    //         {
    //             auto lambda_tmp = xt::linalg::dot(xt::view(lambda, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
    //             m_gamma[i]      = std::min(std::max(m_gamma[i] - dt * lambda, m_gamma_min), 0);
    //             if (m_gamma[i] < -m_gamma_tol)
    //             {
    //             }
    //             else
    //             {
    //                 m_lambda[i] = lambda_tmp;
    //             }
    //             m_lambda_plus[i] = xt::linalg::dot(xt::view(lambda, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
    //         }
    //     }

    //     const auto& lambda() const

    //     {
    //         return m_lambda;
    //     }

    //     auto& lambda()
    //     {
    //         return m_lambda;
    //     }

    //     std::size_t size()
    //     {
    //         return m_lambda.size();
    //     }

    //     void projection(xt::xtensor<double, 1>& lambda)
    //     {
    //         lambda = xt::maximum(lambda, 0.);
    //     }

    //   private:

    //     xt::xtensor<double, 1> m_lambda;
    //     xt::xtensor<double, 1> m_work;
    //     xt::xtensor<double, 1> m_gamma;
    //     double m_gamma_tol;
    //     double m_gamma_min;
    // };

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
            // std::cout << fmt::format("ite = {}, |DG| = {}, |lambda| = {}, residual = {}", ite, xt::linalg::norm(dG), lambda_n[0],
            // residual)
            //   << std::endl;
            ite++;
        }
        return lambda.lambda();
    }

    template <class Gradient>
    auto apgd(Gradient& gradient, double alpha, double tolerance = 1e-6, std::size_t max_ite = 10000, bool dynamic_descent = false)
    {
        std::size_t ite = 0;

        xt::xtensor<double, 1> lambda_n   = xt::zeros<double>({gradient.size()});
        xt::xtensor<double, 1> lambda_np1 = xt::zeros<double>({gradient.size()});

        xt::xtensor<double, 1> theta_n   = xt::ones<double>({gradient.size()});
        xt::xtensor<double, 1> theta_np1 = xt::ones<double>({gradient.size()});

        xt::xtensor<double, 1> y_n   = xt::zeros<double>({gradient.size()});
        xt::xtensor<double, 1> y_np1 = xt::zeros<double>({gradient.size()});

        double residual = 1;

        double r0 = xt::linalg::norm(lambda_n);
        if (r0 == 0.)
        {
            r0 = 1.;
        }

        double lipsch = 1. / alpha;

        while (residual > tolerance && ite < max_ite)
        {
            xt::xtensor<double, 1> dG = gradient(y_n);
            lambda_np1                = y_n - alpha * dG;
            // std::cout << "lambda avant proj: " << lambda_np1 << std::endl;

            gradient.projection(lambda_np1);
            // double norm_dG = xt::linalg::norm(dG);
            // double norm_la = xt::linalg::norm(lambda_np1);
            double norm_dG = xt::norm_linf(dG)[0];
            double norm_la = xt::norm_linf(lambda_np1)[0];
            // std::cout << "lambda: " << lambda_np1 << std::endl;
            // std::cout << "dG: " << dG << std::endl;
            // std::cout << fmt::format("ite = {}, |DG| = {}, |lambda| = {}, residual = {}", ite, norm_dG, norm_la, residual) << std::endl;
            if (norm_dG < tolerance || norm_la < tolerance)
            {
                std::swap(lambda_n, lambda_np1);
                break;
            }

            if (dynamic_descent)
            {
                while (gradient.min_f(lambda_np1)
                       >= gradient.min_f(y_n) + xt::linalg::dot(dG, lambda_np1 - y_n)[0] + 0.5 * lipsch * xt::norm_l2(lambda_np1 - y_n)[0])
                {
                    lipsch *= 2;
                    alpha      = 1. / lipsch;
                    lambda_np1 = y_n - alpha * dG;
                    gradient.projection(lambda_np1);
                }
            }
            theta_np1 = 0.5 * (theta_n * xt::sqrt(4 + theta_n * theta_n) - theta_n * theta_n);
            auto beta = theta_n * (1 - theta_n) / (theta_n * theta_n + theta_np1);
            y_np1     = lambda_np1 + beta * (lambda_np1 - lambda_n);

            residual = xt::linalg::norm(lambda_np1 - lambda_n) / r0;
            // std::cout << "residual = " << residual << std::endl;

            if (dynamic_descent)
            {
                if (xt::linalg::dot(dG, lambda_np1 - lambda_n)[0] > 0)
                {
                    y_np1 = lambda_np1;
                    theta_np1.fill(1.);
                }
                lipsch *= 0.97;
                alpha = 1. / lipsch;
            }
            std::swap(lambda_n, lambda_np1);
            std::swap(theta_n, theta_np1);
            std::swap(y_n, y_np1);
            ite++;
        }
        std::cout << fmt::format("apgd converged in {} iterations.", ite) << std::endl;
        return lambda_n;
    }

    template <std::size_t dim_, class Problem, class D>
    class NewOptimSolverBase : public crtp_base<D>
    {
      public:

        using problem_t                  = Problem;
        static constexpr std::size_t dim = dim_;

        NewOptimSolverBase(double dt)
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

            // std::cout << "V " << particles.vd() << std::endl;
            // std::cout << "pos " << particles.pos() << std::endl;
            if (contacts.size() != 0)
            {
                auto lagrange = this->derived_cast().make_lagrange(contacts);
                auto gradient = make_gradient<problem_t>(m_dt, contacts, particles, lagrange);

                m_lambda = apgd(gradient, 0.05, 1e-6, 10000);

                auto velocities = gradient.velocities(m_lambda);
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

        const auto& lagrange_multiplier() const
        {
            return m_lambda;
        }

        bool should_solve()
        {
            return m_should_solve;
        }

      protected:

        double m_dt;

      private:

        bool m_should_solve;
        xt::xtensor<double, 2> m_u;
        xt::xtensor<double, 2> m_omega;
        xt::xtensor<double, 1> m_lambda;
    };

    struct pairhash
    {
      public:

        template <typename T, typename U>
        std::size_t operator()(const std::pair<T, U>& x) const
        {
            return std::hash<T>()(x.first) ^ std::hash<U>()(x.second);
        }
    };

    // Friction
    template <std::size_t dim_, class Problem>
    class NewOptimSolver : public NewOptimSolverBase<dim_, Problem, NewOptimSolver<dim_, Problem>>
    {
      public:

        using base_t                     = NewOptimSolverBase<dim_, Problem, NewOptimSolver<dim_, Problem>>;
        static constexpr std::size_t dim = dim_;
        using problem_t                  = Problem;

        NewOptimSolver(std::size_t, double dt, const scopi_container<dim>&)
            : base_t(dt)
            , m_mu(.1)
        {
        }

        template <class Contacts>
        auto make_lagrange(const Contacts& contacts) const
        {
            return LagrangeMultiplier<dim, Problem, std::decay_t<Contacts>>(contacts, m_mu);
        }

      private:

        double m_mu;
    };

    // // Viscous
    // template <std::size_t dim_, class Problem>
    // class NewOptimSolver : public NewOptimSolverBase<dim_, Problem, NewOptimSolver<dim_, Problem>>
    // {
    //   public:

    //     using base_t                     = NewOptimSolverBase<dim_, Problem, NewOptimSolver<dim_, Problem>>;
    //     static constexpr std::size_t dim = dim_;
    //     using problem_t                  = Problem;

    //     NewOptimSolver(std::size_t, double dt, const scopi_container<dim>&)
    //         : base_t(dt)
    //         , m_gamma_min(-2.)
    //         , m_gamma_tol(1e-6)
    //     {
    //     }

    //     void run(const scopi_container<dim>& particles, const std::vector<neighbor<dim>>& contacts, std::size_t ite)
    //     {
    //         base_t::run(particles, contacts, ite);

    //         auto lagrange = make_lagrange(contacts);
    //         auto lambda   = lagrange.local2global(this->lagrange_multiplier());

    //         std::cout << "ite: " << ite << " gamma = ";
    //         std::size_t row = 0;
    //         for (std::size_t i = 0; i < contacts.size(); ++i)
    //         {
    //             m_gamma[{contacts[i].i, contacts[i].j}].second = std::min(
    //                 std::max(m_gamma[{contacts[i].i, contacts[i].j}].second
    //                              - this->m_dt * xt::linalg::dot(contacts[i].nij, xt::view(lambda, xt::range(row, row + dim)))[0],
    //                          m_gamma_min),
    //                 0.);
    //             std::cout << m_gamma[{contacts[i].i, contacts[i].j}].second << " ";
    //             row += 3;
    //         }

    //         std::cout << " lambda = " << this->lagrange_multiplier() << std::endl;
    //     }

    //     template <class Contacts>
    //     auto make_lagrange(const Contacts& contacts)
    //     {
    //         xt::xtensor<double, 1> gamma = xt::zeros<double>({contacts.size()});

    //         for (auto it = m_gamma.begin(); it != m_gamma.end(); ++it)
    //         {
    //             it->second.first = false;
    //         }

    //         std::size_t next = 0;
    //         for (std::size_t i = 0; i < contacts.size(); ++i)
    //         {
    //             if (auto search = m_gamma.find({contacts[i].i, contacts[i].j}); search == m_gamma.end())
    //             {
    //                 m_gamma[{contacts[i].i, contacts[i].j}] = {true, 0.};
    //                 gamma[next]                             = 0;
    //             }
    //             else
    //             {
    //                 m_gamma[{contacts[i].i, contacts[i].j}].first = true;
    //                 gamma[next]                                   = m_gamma[{contacts[i].i, contacts[i].j}].second;
    //             }
    //             ++next;
    //         }

    //         for (auto it = m_gamma.begin(); it != m_gamma.end();)
    //         {
    //             if (it->second.first == false)
    //             {
    //                 m_gamma.erase(it);
    //             }
    //             else
    //             {
    //                 ++it;
    //             }
    //         }

    //         return LagrangeMultiplier<dim, Problem, std::decay_t<Contacts>>(contacts, gamma, m_gamma_tol);
    //     }

    //   private:

    //     double m_gamma_min;
    //     double m_gamma_tol;
    //     std::unordered_map<std::pair<std::size_t, std::size_t>, std::pair<bool, double>, pairhash> m_gamma;
    // };

    // WithoutFriction
    // template <std::size_t dim_, class Problem>
    // class NewOptimSolver : public NewOptimSolverBase<dim_, Problem, NewOptimSolver<dim_, Problem>>
    // {
    //   public:

    //     using base_t                     = NewOptimSolverBase<dim_, Problem, NewOptimSolver<dim_, Problem>>;
    //     static constexpr std::size_t dim = dim_;
    //     using problem_t                  = Problem;

    //     NewOptimSolver(std::size_t, double dt, const scopi_container<dim>&)
    //         : base_t(dt)
    //     {
    //     }

    //     template <class Contacts>
    //     auto make_lagrange(const Contacts& contacts) const
    //     {
    //         return LagrangeMultiplier<dim, Problem, std::decay_t<Contacts>>(contacts);
    //     }
    // };

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
    std::setprecision(15);
    xt::print_options::set_precision(15);
    // plog::init(plog::info, "quentin.log");

    constexpr std::size_t dim = 2;

    scopi::scopi_container<dim> particles;

    // SPHERE - PLAN CASE
    double rr = 2;
    double dd = 1;
    double PI = xt::numeric_constants<double>::PI;
    scopi::plan<dim> plan(
        {
            {0., 0.}
    },
        PI / 2 - PI / 6);
    scopi::sphere<dim> sphere(
        {
            {0, (rr + dd) / std::cos(PI / 6)}
    },
        rr);
    particles.push_back(plan, scopi::property<dim>().deactivate());
    particles.push_back(sphere,
                        scopi::property<dim>()
                            .mass(1)
                            .moment_inertia(.5 * rr * rr)
                            .desired_velocity({
                                {0, 0}
    })
                            .force({{0, -1}}));
    double Tf = 10;

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

    double dt = 0.1;

    // using optim_solver = scopi::NewOptimSolver<dim, scopi::WithoutFriction>;
    // using optim_solver = scopi::NewOptimSolver<dim, scopi::Viscous>;
    using optim_solver = scopi::NewOptimSolver<dim, scopi::Friction>;
    using contact_t    = scopi::contact_kdtree;
    using vap_t        = scopi::vap_fpd;
    scopi::ScopiSolver<dim, optim_solver, contact_t, vap_t> solver(particles, dt);
    solver.run(Tf / dt);
    // solver.run(15);
    return 0;
}

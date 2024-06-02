#pragma once

#include <xtensor/xnorm.hpp>
#include <xtensor/xtensor.hpp>

#include "../contact/property.hpp"
#include "../crtp.hpp"

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

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, NoFriction, Contacts>
        : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, NoFriction, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, NoFriction, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts, double)
            : base(contacts)
        {
            m_S_Vector    = xt::zeros<double>({size()});
            m_local_work  = xt::zeros<double>({size()});
            m_global_work = xt::zeros<double>({3 * contacts.size()});
        }

        const auto& global2local(const xt::xtensor<double, 1>& x) const
        {
            assert(x.size() == 3 * this->m_contacts.size());
            // xt::xtensor<double, 1> out = xt::empty<double>({this->m_contacts.size()});
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                // m_local_work[i] = xt::linalg::dot(xt::view(x, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];

                m_local_work[i] = 0;
                for (std::size_t d = 0; d < dim; ++d)
                {
                    m_local_work[i] += x[3 * i + d] * this->m_contacts[i].nij[d];
                }
            }
            return m_local_work;
        }

        auto local2global(const xt::xtensor<double, 1>& x) const
        {
            assert(x.size() == this->m_contacts.size());
            xt::xtensor<double, 1> out = xt::empty<double>({3 * this->m_contacts.size()});
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                for (std::size_t d = 0; d < dim; ++d)
                {
                    out(3 * i + d) = x[i] * this->m_contacts[i].nij(d);
                }
                // xt::view(out, xt::range(3 * i, 3 * i + dim)) = x[i] * this->m_contacts[i].nij;
            }
            return out;
        }

        std::size_t size() const
        {
            return this->m_contacts.size();
        }

        const auto& S_Vector() const
        {
            return m_S_Vector;
        }

        void projection(xt::xtensor<double, 1>& lambda) const
        {
            lambda = xt::maximum(lambda, 0.);
        }

      private:

        mutable xt::xtensor<double, 1> m_local_work;
        mutable xt::xtensor<double, 1> m_global_work;
        xt::xtensor<double, 1> m_S_Vector;
    };

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, Viscous, Contacts> : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Viscous, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Viscous, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts, double)
            : base(contacts)
        {
            m_size = 0;
            for (std::size_t i = 0; i < contacts.size(); ++i)
            {
                if (this->m_contacts[i].property.gamma < -this->m_contacts[i].property.gamma_tol)
                {
                    m_size += 2;
                }
                else
                {
                    ++m_size;
                }
            }
            m_S_Vector = xt::zeros<double>({m_size});
        }

        auto global2local(const xt::xtensor<double, 1>& x) const
        {
            // std::cout << "in global2local " << m_gamma << std::endl;
            assert(x.size() == 3 * this->m_contacts.size());
            xt::xtensor<double, 1> out = xt::empty<double>({size()});
            std::size_t next_gamma_neg = this->m_contacts.size();

            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                if (this->m_contacts[i].property.gamma < -this->m_contacts[i].property.gamma_tol)
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
                if (this->m_contacts[i].property.gamma < -this->m_contacts[i].property.gamma_tol)
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
            return m_size;
        }

        const auto& S_Vector() const
        {
            return m_S_Vector;
        }

        void projection(xt::xtensor<double, 1>& lambda) const
        {
            assert(lambda.size() == m_size);
            lambda = xt::maximum(lambda, 0.);
        }

      private:

        std::size_t m_size;
        xt::xtensor<double, 1> m_S_Vector;
    };

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, Friction, Contacts> : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Friction, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Friction, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts, double)
            : base(contacts)
        {
            m_S_Vector = xt::zeros<double>({size()});
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

        const auto& S_Vector() const
        {
            return m_S_Vector;
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
                double mu     = this->m_contacts[i].property.mu;
                if (norm > mu * lambda_n)
                {
                    if (lambda_n <= -mu * norm)
                    {
                        lambda_i = 0;
                    }
                    else
                    {
                        auto new_norm     = (mu * mu) * (norm + lambda_n / mu) / (mu * mu + 1);
                        auto new_lambda_n = new_norm / mu;
                        lambda_t /= norm;
                        lambda_i = new_norm * lambda_t + new_lambda_n * this->m_contacts[i].nij;
                    }
                }
            }
        }

      private:

        xt::xtensor<double, 1> m_S_Vector;
    };

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, FrictionFixedPoint, Contacts>
        : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, FrictionFixedPoint, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, FrictionFixedPoint, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts, [[maybe_unused]] double dt)
            : base(contacts)
        {
            m_S_Vector = xt::zeros<double>({size()});
            for (std::size_t i = 0, row = 0; i < this->m_contacts.size(); ++i, row += 3)
            {
                auto S_i = xt::view(m_S_Vector, xt::range(row, row + dim));
                S_i      = this->m_contacts[i].sij * this->m_contacts[i].nij;
            }
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

        const auto& S_Vector() const
        {
            return m_S_Vector;
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
                double mu     = this->m_contacts[i].property.mu;
                if (norm > mu * lambda_n)
                {
                    if (lambda_n <= -mu * norm)
                    {
                        lambda_i = 0;
                    }
                    else
                    {
                        auto new_norm     = (mu * mu) * (norm + lambda_n / mu) / (mu * mu + 1);
                        auto new_lambda_n = new_norm / mu;
                        lambda_t /= norm;
                        lambda_i = new_norm * lambda_t + new_lambda_n * this->m_contacts[i].nij;
                    }
                }
            }
        }

      private:

        xt::xtensor<double, 1> m_S_Vector;
    };

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, ViscousFriction, Contacts>
        : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, ViscousFriction, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, ViscousFriction, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts, double dt)
            : base(contacts)
        {
            m_size = 0;
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                if (this->m_contacts[i].property.gamma < -this->m_contacts[i].property.gamma_tol)
                {
                    if (this->m_contacts[i].property.gamma != this->m_contacts[i].property.gamma_min)
                    {
                        m_size += 2;
                    }
                    else
                    {
                        m_size += 4;
                    }
                }
                else
                {
                    ++m_size;
                }
            }
            m_S_Vector      = xt::zeros<double>({size()});
            std::size_t row = 0;
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                if (this->m_contacts[i].property.gamma < -this->m_contacts[i].property.gamma_tol)
                {
                    if (this->m_contacts[i].property.gamma != this->m_contacts[i].property.gamma_min)
                    {
                        row += 2;
                    }
                    else
                    {
                        auto S_i = xt::view(m_S_Vector, xt::range(row + 1, row + 1 + dim));
                        S_i      = dt * this->m_contacts[i].property.mu * this->m_contacts[i].sij * this->m_contacts[i].nij;
                        row += 4;
                    }
                }
                else
                {
                    ++row;
                }
            }
        }

        auto global2local(const xt::xtensor<double, 1>& x) const
        {
            assert(x.size() == 3 * this->m_contacts.size());
            xt::xtensor<double, 1> out = xt::zeros<double>({size()});
            std::size_t row            = 0;
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                if (this->m_contacts[i].property.gamma < -this->m_contacts[i].property.gamma_tol)
                {
                    if (this->m_contacts[i].property.gamma != this->m_contacts[i].property.gamma_min)
                    {
                        out[row]     = xt::linalg::dot(xt::view(x, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
                        out[row + 1] = -xt::linalg::dot(xt::view(x, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
                        row += 2;
                    }
                    else
                    {
                        out[row] = -xt::linalg::dot(xt::view(x, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
                        xt::view(out, xt::range(row + 1, row + 1 + dim)) = xt::view(x, xt::range(3 * i, 3 * i + dim));
                        row += 4;
                    }
                }
                else
                {
                    out[row] = xt::linalg::dot(xt::view(x, xt::range(3 * i, 3 * i + dim)), this->m_contacts[i].nij)[0];
                    ++row;
                }
            }
            return out;
        }

        auto local2global(const xt::xtensor<double, 1>& x) const
        {
            assert(x.size() == size());
            xt::xtensor<double, 1> out = xt::zeros<double>({3 * this->m_contacts.size()});
            std::size_t row            = 0;
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                if (this->m_contacts[i].property.gamma < -this->m_contacts[i].property.gamma_tol)
                {
                    if (this->m_contacts[i].property.gamma != this->m_contacts[i].property.gamma_min)
                    {
                        xt::view(out, xt::range(3 * i, 3 * i + dim)) = (x[row] - x[row + 1]) * this->m_contacts[i].nij;
                        row += 2;
                    }
                    else
                    {
                        xt::view(out, xt::range(3 * i, 3 * i + dim)) = (-x[row]) * this->m_contacts[i].nij
                                                                     + xt::view(x, xt::range(row + 1, row + 1 + dim));
                        row += 4;
                    }
                }
                else
                {
                    xt::view(out, xt::range(3 * i, 3 * i + dim)) = x[row] * this->m_contacts[i].nij;
                    row++;
                }
            }
            return out;
        }

        std::size_t size() const
        {
            return m_size;
        }

        const auto& S_Vector() const
        {
            return m_S_Vector;
        }

        void projection(xt::xtensor<double, 1>& lambda) const
        {
            assert(lambda.size() == size());
            std::size_t row = 0;
            for (std::size_t i = 0; i < this->m_contacts.size(); ++i)
            {
                if (this->m_contacts[i].property.gamma < -this->m_contacts[i].property.gamma_tol)
                {
                    if (this->m_contacts[i].property.gamma != this->m_contacts[i].property.gamma_min)
                    {
                        auto lambda_visqu = xt::view(lambda, xt::range(row, row + 2));
                        lambda_visqu      = xt::maximum(lambda_visqu, 0.);
                        row += 2;
                    }
                    else
                    {
                        auto lambda_i                           = xt::view(lambda, xt::range(row, row + 1 + dim));
                        auto lambda_f_i                         = xt::view(lambda, xt::range(row + 1, row + 1 + dim));
                        auto lambda_n                           = xt::linalg::dot(lambda_f_i, this->m_contacts[i].nij)[0];
                        auto lambda_t                           = xt::eval(lambda_f_i - lambda_n * this->m_contacts[i].nij);
                        auto norm                               = xt::norm_l2(lambda_t)[0];
                        double mu                               = this->m_contacts[i].property.mu;
                        xt::xtensor<double, 1> lambda_proj_fric = xt::zeros<double>({1 + dim});
                        auto lambda_fproj                       = xt::view(lambda_proj_fric, xt::range(1, 1 + dim));
                        if (norm > mu * lambda_n)
                        {
                            if (lambda_n <= -mu * norm)
                            {
                                lambda_fproj = xt::zeros<double>({dim});
                            }
                            else
                            {
                                auto new_norm     = (mu * mu) * (norm + lambda_n / mu) / (mu * mu + 1);
                                auto new_lambda_n = new_norm / mu;
                                lambda_t /= norm;
                                lambda_fproj = new_norm * lambda_t + new_lambda_n * this->m_contacts[i].nij;
                            }
                        }
                        else
                        {
                            lambda_fproj = lambda_f_i;
                        }

                        xt::xtensor<double, 1> lambda_proj_moins = xt::zeros<double>({1 + dim});
                        lambda_proj_moins[0]                     = std::max(lambda[row], 0.);
                        if (xt::norm_l2(xt::eval(lambda_proj_fric - lambda_i))[0] < xt::norm_l2(xt::eval(lambda_proj_moins - lambda_i))[0])
                        {
                            lambda_i = lambda_proj_fric;
                        }
                        else
                        {
                            lambda_i = lambda_proj_moins;
                        }
                        row += 4;
                    }
                }
                else
                {
                    lambda[row] = std::max(lambda[row], 0.);
                    ++row;
                }
            }
        }

      private:

        std::size_t m_size;
        xt::xtensor<double, 1> m_S_Vector;
    };

    template <std::size_t dim, class Type, class Contacts>
    auto make_lagrange_multplier(const Contacts& contacts, double dt)
    {
        return LagrangeMultiplier<dim, Type, Contacts>(contacts, dt);
    }

}

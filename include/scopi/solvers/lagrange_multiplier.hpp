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

        LagrangeMultiplier(const Contacts& contacts)
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

        void projection(xt::xtensor<double, 1>& lambda) const
        {
            assert(lambda.size() == m_size);
            lambda = xt::maximum(lambda, 0.);
        }

      private:

        std::size_t m_size;
    };

    template <std::size_t dim_, class Contacts>
    class LagrangeMultiplier<dim_, Friction, Contacts> : public LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Friction, Contacts>>
    {
      public:

        static constexpr std::size_t dim = dim_;

        using base = LagrangeMultiplierBase<Contacts, LagrangeMultiplier<dim_, Friction, Contacts>>;

        LagrangeMultiplier(const Contacts& contacts)
            : base(contacts)
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
    };

    template <std::size_t dim, class Type, class Contacts>
    auto make_lagrange_multplier(const Contacts& contacts)
    {
        return LagrangeMultiplier<dim, Type, Contacts>(contacts);
    }
}
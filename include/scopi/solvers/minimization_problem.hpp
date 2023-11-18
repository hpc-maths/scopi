#pragma once

#include <xtensor/xtensor.hpp>

#include "../matrix/velocities.hpp"
#include "lagrange_multiplier.hpp"

namespace scopi
{
    template <class Particles>
    inline auto M_inverse(const Particles& particles)
    {
        static constexpr std::size_t dim = Particles::dim;

        xt::xtensor<double, 1> out = xt::zeros<double>({6 * particles.nb_active()});

        std::size_t offset = 3 * particles.nb_active();
        for (std::size_t i = 0; i < particles.nb_active(); ++i)
        {
            xt::view(out, xt::range(3 * i, 3 * i + dim)) = 1. / particles.m()[particles.nb_inactive() + i];
            if constexpr (dim == 2)
            {
                out[offset + 3 * i + 2] = 1. / particles.j()[particles.nb_inactive() + i];
            }
            else if constexpr (dim == 3)
            {
                xt::view(out, xt::range(offset + 3 * i, offset + 3 * i + dim)) = 1. / particles.j()[particles.nb_inactive() + i];
            }
        }
        return out;
    }

    template <class Contacts, class Particles>
    auto CVector(double dt, const Contacts& contacts, const Particles& particles)
    {
        static constexpr std::size_t dim = Particles::dim;
        AMatrix A(contacts, particles);
        DMatrix D(contacts);

        xt::xtensor<double, 1> U = xt::zeros<double>({6 * particles.nb_active()});

        std::size_t offset = 3 * particles.nb_active();
        for (std::size_t i = 0; i < particles.nb_active(); ++i)
        {
            xt::view(U, xt::range(3 * i, 3 * i + dim)) = particles.v()[particles.nb_inactive() + i];
            if constexpr (dim == 2)
            {
                U[offset + 3 * i + 2] = particles.omega()[particles.nb_inactive() + i];
            }
            else if constexpr (dim == 3)
            {
                xt::view(U, xt::range(offset + 3 * i, offset + 3 * i + dim)) = particles.omega()[particles.nb_inactive() + i];
            }
        }

        xt::xtensor<double, 1> normal = xt::zeros<double>({3 * contacts.size()});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            xt::view(normal, xt::range(3 * i, 3 * i + dim)) = contacts[i].nij;
        }

        // std::cout << "normal " << normal << std::endl;
        // std::cout << "U " << U << std::endl;
        // std::cout << "D " << D.mat_mult(normal) << std::endl;
        // std::cout << "dt " << dt << std::endl;
        // std::cout << "A " << dt * A.mat_mult(U) << std::endl;

        xt::xtensor<double, 1> value = D.mat_mult(normal) + dt * A.mat_mult(U);
        return value;
    }

    template <class Contacts, class Particles>
    struct QMatrix
    {
      public:

        QMatrix(double dt, const Contacts& contacts, const Particles& particles)
            : m_dt(dt)
            , m_A(contacts, particles)
            , m_AT(contacts, particles)
            , m_invM(M_inverse(particles))
        {
        }

        inline auto operator()(const xt::xtensor<double, 1>& lambda) const
        {
            // std::cout << std::endl << "lambda " << lambda << std::endl;
            // std::cout << std::endl << "m_At lambda " << m_AT.mat_mult(lambda) << std::endl;
            // std::cout << std::endl << "invM  " << m_invM << std::endl;
            // std::cout << std::endl << "m_A  " << m_A.mat_mult(m_invM * m_AT.mat_mult(lambda)) << std::endl;
            return m_dt * m_dt * m_A.mat_mult(m_invM * m_AT.mat_mult(lambda));
        }

        inline auto velocities(const xt::xtensor<double, 1>& lambda) const
        {
            return xt::eval(m_invM * m_AT.mat_mult(lambda));
        }

      private:

        double m_dt;
        AMatrix<Contacts, Particles> m_A;
        ATMatrix<Contacts, Particles> m_AT;
        xt::xtensor<double, 1> m_invM;
    };

    template <class Problem, class Contacts, class Particles>
    class minimization_problem
    {
      public:

        static constexpr std::size_t dim = Particles::dim;

        inline minimization_problem(double dt, const Contacts& contacts, const Particles& particles)
            : m_Q(dt, contacts, particles)
            , m_C(CVector(dt, contacts, particles))
            , m_lagrange(make_lagrange_multplier<Particles::dim, Problem>(contacts))
        {
            PLOG_DEBUG << "m_C " << m_C << " " << m_lagrange.global2local(m_C) << std::endl;
        }

        inline xt::xtensor<double, 1> gradient(const xt::xtensor<double, 1>& lambda) const
        {
            // std::cout << "local2global " << m_lagrange.local2global(lambda) << std::endl;
            // std::cout << "Q " << m_Q(m_lagrange.local2global(lambda)) << std::endl;
            // std::cout << "Q +C" << m_Q(m_lagrange.local2global(lambda) + m_C) << std::endl;

            return m_lagrange.global2local(m_Q(m_lagrange.local2global(lambda)) + m_C);
        }

        inline double operator()(const xt::xtensor<double, 1>& lambda) const
        {
            auto lambda_global = m_lagrange.local2global(lambda);
            return xt::linalg::dot(lambda_global, 0.5 * m_Q(lambda_global) + m_C)[0];
        }

        inline auto velocities(const xt::xtensor<double, 1>& lambda) const
        {
            return m_Q.velocities(m_lagrange.local2global(lambda));
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

        const QMatrix<Contacts, Particles> m_Q;
        const xt::xtensor<double, 1> m_C;
        const LagrangeMultiplier<Particles::dim, Problem, Contacts> m_lagrange;
    };

    template <class Problem, class Contacts, class Particles>
    auto make_minimization_problem(double dt, const Contacts& contacts, const Particles& particles)
    {
        return minimization_problem<Problem, Contacts, Particles>(dt, contacts, particles);
    }
}
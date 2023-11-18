#pragma once

#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xnoalias.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor/xview.hpp>

namespace scopi
{

    namespace detail
    {
        template <class Mat, class Vec>
        inline auto mat_mult(const Mat& A, const Vec& x)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> y;
            y(0) = A(0, 0) * x(0) + A(0, 1) * x(1) + A(0, 2) * x(2);
            y(1) = A(1, 0) * x(0) + A(1, 1) * x(1) + A(1, 2) * x(2);
            y(2) = A(2, 0) * x(0) + A(2, 1) * x(1) + A(2, 2) * x(2);
            return y;
        }

        template <class X, class Y>
        inline auto cross(std::integral_constant<std::size_t, 2>, const X& x, const Y& y)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> out;
            out(0) = x(1) * y(2);
            out(1) = -x(0) * y(2);
            out(2) = x(0) * y(1) - x(1) * y(0);
            return out;
        }

        template <class X, class Y>
        inline auto cross(std::integral_constant<std::size_t, 3>, const X& x, const Y& y)
        {
            xt::xtensor_fixed<double, xt::xshape<3>> out;
            out(0) = x(1) * y(2) - x(2) * y(1);
            out(1) = x(2) * y(0) - x(0) * y(2);
            out(2) = x(0) * y(1) - x(1) * y(0);
            return out;
        }

        template <std::size_t dim, class X, class Y>
        inline auto cross(const X& x, const Y& y)
        {
            return cross(std::integral_constant<std::size_t, dim>{}, x, y);
        }
    } // namespace detail

    template <class Contacts_t, class Particles_t>
    class AMatrix
    {
      public:

        static constexpr std::size_t dim = Particles_t::dim;

        AMatrix(const Contacts_t& contacts, const Particles_t& particles)
            : m_contacts{contacts}
            , m_particles{particles}
            , m_work{xt::zeros<double>({3 * contacts.size()})}
        {
        }

        const auto& mat_mult(const xt::xtensor<double, 1>& u) const
        {
            m_work.fill(0.);
            std::size_t active_offset = m_particles.nb_inactive();
            std::size_t rot_offset    = 3 * m_particles.nb_active();
            std::size_t row           = 0;

            for (auto& c : m_contacts)
            {
                auto view = xt::view(m_work, xt::range(row, row + 3));
                if (c.i >= active_offset)
                {
                    std::size_t start = (c.i - active_offset) * 3;
                    auto v_i          = xt::view(u, xt::range(start, start + 3));

                    start += rot_offset;
                    auto omega_i = xt::view(u, xt::range(start, start + 3));
                    auto rij_i   = c.pi - m_particles.pos()(c.i);
                    auto Ri      = rotation_matrix<3>(m_particles.q()(c.i));

                    view += v_i - detail::cross<dim>(rij_i, detail::mat_mult(Ri, omega_i));
                }
                if (c.j >= active_offset)
                {
                    std::size_t start = (c.j - active_offset) * 3;
                    auto v_j          = xt::view(u, xt::range(start, start + 3));

                    start += rot_offset;
                    auto omega_j = xt::view(u, xt::range(start, start + 3));
                    auto rij_j   = c.pj - m_particles.pos()(c.j);
                    auto Rj      = rotation_matrix<3>(m_particles.q()(c.j));

                    view -= v_j - detail::cross<dim>(rij_j, detail::mat_mult(Rj, omega_j));
                }
                row += 3;
            }
            return m_work;
        }

      private:

        const Contacts_t& m_contacts;
        const Particles_t& m_particles;
        mutable xt::xtensor<double, 1> m_work;
    };

    template <class Contacts_t, class Particles_t>
    class ATMatrix
    {
      public:

        static constexpr std::size_t dim = Particles_t::dim;

        ATMatrix(const Contacts_t& contacts, const Particles_t& particles)
            : m_contacts{contacts}
            , m_particles{particles}
            , m_work{xt::zeros<double>({6 * particles.nb_active()})}
        {
        }

        const auto& mat_mult(const xt::xtensor<double, 1>& f) const
        {
            m_work.fill(0.);
            std::size_t active_offset = m_particles.nb_inactive();
            std::size_t rot_offset    = 3 * m_particles.nb_active();
            std::size_t row           = 0;

            for (auto& c : m_contacts)
            {
                auto f_view = xt::view(f, xt::range(row, row + 3));

                if (c.i >= active_offset)
                {
                    std::size_t start = (c.i - active_offset) * 3;
                    auto v_i          = xt::view(m_work, xt::range(start, start + 3));

                    v_i += f_view;

                    start += rot_offset;
                    auto omega_i = xt::view(m_work, xt::range(start, start + 3));
                    auto rij_i   = c.pi - m_particles.pos()(c.i);
                    auto R_i     = rotation_matrix<3>(m_particles.q()(c.i));

                    omega_i += detail::mat_mult(xt::transpose(R_i), detail::cross<dim>(rij_i, f_view));
                }
                if (c.j >= active_offset)
                {
                    std::size_t start = (c.j - active_offset) * 3;
                    auto v_j          = xt::view(m_work, xt::range(start, start + 3));

                    v_j -= f_view;

                    start += rot_offset;
                    auto omega_j = xt::view(m_work, xt::range(start, start + 3));
                    auto rij_j   = c.pj - m_particles.pos()(c.j);
                    auto R_j     = rotation_matrix<3>(m_particles.q()(c.j));

                    omega_j -= detail::mat_mult(xt::transpose(R_j), detail::cross<dim>(rij_j, f_view));
                }
                row += 3;
            }
            return m_work;
        }

      private:

        const Contacts_t& m_contacts;
        const Particles_t& m_particles;
        mutable xt::xtensor<double, 1> m_work;
    };

    template <class Contacts_t>
    class DMatrix
    {
      public:

        DMatrix(const Contacts_t& contacts)
            : m_contacts{contacts}
            , m_work{xt::zeros<double>({3 * contacts.size()})}
        {
        }

        const auto& mat_mult(const xt::xtensor<double, 1>& f)
        {
            m_work.fill(0.);
            std::size_t row = 0;

            for (auto& c : m_contacts)
            {
                auto f_view   = xt::view(f, xt::range(row, row + c.nij.shape(0)));
                auto out_view = xt::view(m_work, xt::range(row, row + c.nij.shape(0)));

                xt::noalias(out_view) = c.dij * f_view;

                row += 3;
            }
            return m_work;
        }

      private:

        const Contacts_t& m_contacts;
        mutable xt::xtensor<double, 1> m_work;
    };
} // namespace scopi
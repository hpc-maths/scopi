#pragma once

#include <xtensor/xnpy.hpp>
#include <xtensor/xtensor.hpp>

#include "../matrix/velocities.hpp"
#include "../objects/neighbor.hpp"
#include "lagrange_multiplier.hpp"

using namespace xt::placeholders;

namespace scopi
{
    namespace detail
    {
        template <std::size_t dim, class problem_t>
        void sort_contacts_over_j(std::vector<neighbor<dim, problem_t>>& contacts, std::size_t index)
        {
            std::sort(contacts.begin(),
                      contacts.begin() + index,
                      [](const auto& a, const auto& b)
                      {
                          if (a.j < b.j)
                          {
                              return true;
                          }

                          if (a.j == b.j)
                          {
                              return a.i < b.i;
                          }

                          return false;
                      });
        }
    }

    template <class Contacts_t>
    class BMatrix
    {
      public:

        BMatrix(double dt, Contacts_t&& contacts)
            : m_dt(dt)
            , m_contacts{std::move(contacts)}
        {
        }

        auto mat_mult(const xt::xtensor<double, 2>& Vr) const
        {
            xt::xtensor<double, 2> out = xt::zeros<double>({m_contacts.size(), Vr.shape(1)});

            std::size_t row = 0;
            for (auto& c : m_contacts)
            {
                auto view = xt::view(out, row, xt::all());
                if (c.i != std::numeric_limits<std::size_t>::max()) // c.i and c.j are the indices of the contacts involved.
                {
                    std::size_t start = c.i * 2;
                    view -= m_dt * xt::linalg::dot(xt::view(c.nij, xt::range(0, 2)), xt::view(Vr, xt::range(start, start + 2), xt::all()));
                }
                if (c.j != std::numeric_limits<std::size_t>::max())
                {
                    std::size_t start = c.j * 2;
                    view += m_dt * xt::linalg::dot(xt::view(c.nij, xt::range(0, 2)), xt::view(Vr, xt::range(start, start + 2), xt::all()));
                }
                row++;
            }
            return out;
        }

        // For each contact, the function computes the dot product of the normal vector with the velocities of the involved particles and
        // updates the corresponding rows of out.

      private:

        double m_dt;
        const Contacts_t m_contacts;
    };

    template <class Contacts>
    auto compute_B_and_D(double dt, Contacts contacts) // Attention: copy of contacts here because we reorder the contact array (must be
                                                       // removed and use only a permutation array)
    {
        std::size_t index = 0;
        for (auto& c : contacts)
        {
            if (c.i < 8)
            {
                index++;
            }
            else
            {
                break;
            }
        }

        detail::sort_contacts_over_j(contacts, index);

        using contacts_t = std::decay_t<decltype(contacts)>; // contacts_t is the type of contacts
        contacts_t contacts2rom;                             // new vector to store
        int j = -1;
        for (std::size_t ic = 0; ic < index; ++ic) // iterates through the sorted contacts
        {
            auto& c = contacts[ic];
            if (c.j != static_cast<std::size_t>(j))
            {
                contacts2rom.push_back(c);
                j = c.j;
            }
            while (contacts[ic + 1].j == static_cast<std::size_t>(j))
            {
                if (contacts2rom.back().dij > contacts[ic + 1].dij)
                {
                    contacts2rom.back() = contacts[ic + 1];
                }
                ic++;
            }
            std::swap(contacts2rom.back().pi, contacts2rom.back().pj);
            contacts2rom.back().nij *= -1;
            contacts2rom.back().i = contacts2rom.back().j - 8;
            contacts2rom.back().j = std::numeric_limits<std::size_t>::max(); // Sets i to an invalid state
                                                                             // (std::numeric_limits<std::size_t>::max()), indicating it is
                                                                             // no longer relevant.
        }

        // std::cout << "contact number between particles: " << contacts.size() - index << " (contacts size: " << contacts.size() << "
        // index: " << index << ")" << std::endl;
        for (std::size_t ic = index; ic < contacts.size(); ++ic)
        {
            auto& c = contacts[ic];
            contacts2rom.push_back(c);
            contacts2rom.back().j -= 8;
            contacts2rom.back().i -= 8;
        }

        sort_contacts(contacts2rom);

        xt::xtensor<double, 1> D = xt::empty<double>({contacts2rom.size()});
        std::size_t ic           = 0;
        for (auto& c : contacts2rom)
        {
            D[ic++] = c.dij;
        }

        auto B = BMatrix(dt, std::move(contacts2rom));

        return std::make_pair(B, D);
    }

    template <class Problem, class Contacts, class Particles>
    class minimization_problem_rom
    {
      public:

        static constexpr std::size_t dim = Particles::dim;

        inline minimization_problem_rom(double dt, const Contacts& contacts, const Particles& particles)
            : m_lagrange(make_lagrange_multplier<Particles::dim, Problem>(contacts, dt))
        {
            std::string npy_filename_Vr = "/Users/loic/Work/scopi/scopi/build/files2SCoPI/RB_v.npy";
            std::string npy_filename_Wr = "/Users/loic/Work/scopi/scopi/build/files2SCoPI/RB_lambda.npy";

            auto t  = xt::load_npy<float>(npy_filename_Vr);
            m_V     = xt::load_npy<float>(npy_filename_Vr);
            auto VT = xt::eval(xt::transpose(m_V));
            auto W  = xt::load_npy<float>(npy_filename_Wr);
            auto WT = xt::eval(xt::transpose(W));

            auto [B, D] = compute_B_and_D(dt, contacts);

            auto BV                       = B.mat_mult(m_V);
            xt::xtensor<double, 2> m_Bhat = xt::linalg::dot(WT, BV);

            using namespace xt::placeholders;
            auto U = xt::view(particles.vd(), xt::range(8, _));

            m_Uc           = xt::empty<double>({U.size() * 2});
            std::size_t ii = 0;
            for (auto& u : U)
            {
                xt::view(m_Uc, xt::range(ii, ii + 2)) = u;
                ii += 2;
            }
            m_Qhat = xt::linalg::dot(m_Bhat, xt::transpose(m_Bhat));
            m_Chat = -(xt::linalg::dot(m_Bhat, xt::linalg::dot(VT, m_Uc)) - xt::linalg::dot(WT, D));
        }

        inline xt::xtensor<double, 1> gradient(const xt::xtensor<double, 1>& lambda) const
        {
            return xt::linalg::dot(m_Qhat, lambda) + m_Chat;
        }

        inline double operator()(const xt::xtensor<double, 1>& lambda) const
        {
            return xt::linalg::dot(lambda, 0.5 * xt::linalg::dot(m_Qhat, lambda) + m_Chat)[0];
        }

        inline auto velocities(const xt::xtensor<double, 1>& lambda) const
        {
            return xt::linalg::dot(m_V, xt::linalg::dot(m_V, m_Uc) - xt::linalg::dot(xt::transpose(m_Bhat), lambda));
        }

        void projection(xt::xtensor<double, 1>& lambda) const
        {
            m_lagrange.projection(lambda);
        }

        std::size_t size() const
        {
            return m_Chat.size();
        }

      private:

        xt::xtensor<double, 2> m_Qhat;
        xt::xtensor<double, 2> m_Bhat;
        xt::xtensor<double, 1> m_Chat;
        xt::xtensor<double, 1> m_Uc;
        xt::xtensor<double, 2> m_V;
        const LagrangeMultiplier<Particles::dim, Problem, Contacts> m_lagrange;
    };

    template <class Problem, class Contacts, class Particles>
    auto make_minimization_problem_rom(double dt, const Contacts& contacts, const Particles& particles)
    {
        return minimization_problem_rom<Problem, Contacts, Particles>(dt, contacts, particles);
    }
}

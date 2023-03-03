#pragma once

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../objects/methods/closest_points.hpp"
#include "../objects/methods/select.hpp"
#include "../quaternion.hpp"
#include "../objects/neighbor.hpp"
#include "../params.hpp"
#include "ProblemBase.hpp"

namespace scopi
{
    class DryWithoutFriction;

    /**
     * @class ProblemParams<DryWithoutFriction>
     * @brief Parameters for DryWithoutFriction.
     *
     * Specialization of ProblemParam.
     *
     * Defined for compatibility.
     */
    template<>
    struct ProblemParams<DryWithoutFriction>
    {};

    /**
     * @class DryWithoutFriction.
     * @brief Problem that models contacts without friction and without viscosity.
     *
     * See ProblemBase for the notations.
     * The constraint is
     * \f[
     *      \mathbf{d} + \mathbb{B} \mathbf{u} \ge 0,
     * \f]
     * with \f$ \mathbf{d} \in \mathbb{R}^{N_c} \f$, \f$ \mathbf{u} \in \mathbb{R}^{6N} \f$, and \f$ \mathbb{B} \in \mathbb{R}^{N_c \times 6 N} \f$.
     * We impose that the distance between all the particles should be non-negative.
     * For worms, we also impose that the distance between spheres in a worm is non-positive.
     * More exactly, we impose that minus the distance is non-negative.
     */
    class DryWithoutFriction : public ProblemBase<ProblemParams<DryWithoutFriction>>
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         * @param dt [in] Time step.
         */
        DryWithoutFriction(std::size_t nparts, double dt);

        /**
         * @brief Get the number of rows in the matrix.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance.
         *
         * @return Number of rows in the matrix.
         */
        template <std::size_t dim>
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts);
        /**
         * @brief Create vector \f$ \mathbf{d} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         */
        template<std::size_t dim>
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts);

        /**
         * @brief Extra steps before solving the optimization problem.
         *
         * For compatibility with the other problems.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         */
        template<std::size_t dim, class optim_solver_t>
        void extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts,
                                      const optim_solver_t& solver);
        /**
         * @brief Extra steps after solving the optimization problem.
         *
         * For compatibility with the other problems.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param lambda [in] Lagrange multipliers.
         * @param u_tilde [in] Vector \f$ \mathbf{d} + \mathbb{B} \mathbf{u} - \mathbf{f}(\mathbf{u}) \f$, where \f$ \mathbf{u} \f$ is the solution of the optimization problem.
         */
        template<std::size_t dim, class optim_solver_t>
        void extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                     const optim_solver_t& solver);
        /**
         * @brief Whether the optimization problem should be solved.
         *
         * For compatibility with the other problems.
         */
        bool should_solve() const;

    public:
        /**
         * @brief Construct the COO storage of the matrix \f$ \mathbb{B} \f$ for the constraint.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for positions).
         * @param contacts [in] Array of contacts.
         * @param firstCol [in] Index of the first column (solver-dependent).
         */
        template <std::size_t dim>
        void create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts);

        /**
         * @brief Matrix-free product \f$ \mathbf{r} = \mathbf{r} - \mathbb{B} \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param c [in] Contact of the computed row \c row.
         * @param particles [in] Array of particles (to get positions).
         * @param U [in] Vector \f$ \mathbf{u} \f$.
         * @param R [in/out] Vector \f$ \mathbf{r} \f$.
         * @param active_offset [in] Index of the first active particle.
         * @param row [in] Index of the computed row.
         */
        template<std::size_t dim>
        void matrix_free_gemv_A(const neighbor<dim>& c,
                                const scopi_container<dim>& particles,
                                const xt::xtensor<double, 1>& U,
                                xt::xtensor<double, 1>& R,
                                std::size_t active_offset,
                                std::size_t row);
        /**
         * @brief Matrix-free product \f$ \mathbf{u} = \mathbb{B}^T \mathbf{l} + \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param c [in] Contact of the computed row \c row.
         * @param particles [in] Array of particles (to get positions).
         * @param L [in] Vector \f$ \mathbf{l} \f$.
         * @param U [in/out] Vector \f$ \mathbf{u} \f$.
         * @param active_offset [in] Index of the first active particle.
         * @param row [in] Index of the computed row.
         */
        template<std::size_t dim>
        void matrix_free_gemv_transpose_A(const neighbor<dim>& c,
                                          const scopi_container<dim>& particles,
                                          const xt::xtensor<double, 1>& L,
                                          xt::xtensor<double, 1>& U,
                                          std::size_t active_offset,
                                          std::size_t row);
    };

    template<std::size_t dim>
    void DryWithoutFriction::create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                                          const std::vector<neighbor<dim>>& contacts)
    {
        matrix_positive_distance(particles, contacts, 1);

    }

    template <std::size_t dim>
    std::size_t DryWithoutFriction::number_row_matrix(const std::vector<neighbor<dim>>& contacts)
    {
        return contacts.size();
    }

    template<std::size_t dim>
    void DryWithoutFriction::create_vector_distances(const std::vector<neighbor<dim>>& contacts)
    {
        this->m_distances = xt::zeros<double>({contacts.size()});
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            this->m_distances[i] = contacts[i].dij;
        }
    }

    template<std::size_t dim>
    void DryWithoutFriction::matrix_free_gemv_A(const neighbor<dim>& c,
                                                const scopi_container<dim>& particles,
                                                const xt::xtensor<double, 1>& U,
                                                xt::xtensor<double, 1>& R,
                                                std::size_t active_offset,
                                                std::size_t row)
    {
        if (c.i >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                R(row) -= (-this->m_dt*c.nij[d]) * U((c.i - active_offset)*3 + d);
            }
        }
        if (c.j >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                R(row) -= (this->m_dt*c.nij[d]) * U((c.j - active_offset)*3 + d);
            }
        }

        auto ri_cross = cross_product<dim>(c.pi - particles.pos()(c.i));
        auto rj_cross = cross_product<dim>(c.pj - particles.pos()(c.j));
        auto Ri = rotation_matrix<3>(particles.q()(c.i));
        auto Rj = rotation_matrix<3>(particles.q()(c.j));

        if (c.i >= active_offset)
        {
            std::size_t ind_part = c.i - active_offset;
            auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
                R(row) -= (this->m_dt*(c.nij[0]*dot(0, ip) + c.nij[1]*dot(1, ip) + c.nij[2]*dot(2, ip)))
                    * U(3*particles.nb_active() + 3*ind_part + ip);
            }
        }

        if (c.j >= active_offset)
        {
            std::size_t ind_part = c.j - active_offset;
            auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
                R(row) -= (-this->m_dt*(c.nij[0]*dot(0, ip) + c.nij[1]*dot(1, ip) + c.nij[2]*dot(2, ip)))
                    * U(3*particles.nb_active() + 3*ind_part + ip);
            }
        }
    }

    template<std::size_t dim>
    void DryWithoutFriction::matrix_free_gemv_transpose_A(const neighbor<dim>& c,
                                                          const scopi_container<dim>& particles,
                                                          const xt::xtensor<double, 1>& L,
                                                          xt::xtensor<double, 1>& U,
                                                          std::size_t active_offset,
                                                          std::size_t row)
    {
        if (c.i >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
#pragma omp atomic
                U((c.i - active_offset)*3 + d) += -L(row) * this->m_dt * c.nij[d];
            }
        }
        if (c.j >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
#pragma omp atomic
                U((c.j - active_offset)*3 + d) += L(row) * this->m_dt * c.nij[d];
            }
        }

        auto ri_cross = cross_product<dim>(c.pi - particles.pos()(c.i));
        auto rj_cross = cross_product<dim>(c.pj - particles.pos()(c.j));
        auto Ri = rotation_matrix<3>(particles.q()(c.i));
        auto Rj = rotation_matrix<3>(particles.q()(c.j));

        if (c.i >= active_offset)
        {
            std::size_t ind_part = c.i - active_offset;
            auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
#pragma omp atomic
                U(3*particles.nb_active() + 3*ind_part + ip) += L(row) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
            }
        }

        if (c.j >= active_offset)
        {
            std::size_t ind_part = c.j - active_offset;
            auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
#pragma omp atomic
                U(3*particles.nb_active() + 3*ind_part + ip) += L(row) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
            }
        }
    }

    template<std::size_t dim, class optim_solver_t>
    void DryWithoutFriction::extra_steps_before_solve(const std::vector<neighbor<dim>>&, const optim_solver_t&)
    {
        this->m_should_solve = true;
    }

    template<std::size_t dim, class optim_solver_t>
    void DryWithoutFriction::extra_steps_after_solve(const std::vector<neighbor<dim>>&,
                                                     const optim_solver_t&)
    {
        this->m_should_solve = false;
    }

}


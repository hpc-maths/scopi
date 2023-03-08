#pragma once

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../utils.hpp"
#include "../objects/neighbor.hpp"
#include "../quaternion.hpp"

namespace scopi
{
    /**
     * @class ProblemBase
     * @brief Shared methods for problems.
     *
     * All problems (models) solve
     * \f[
     *      \min \frac{1}{2} \mathbf{u} \cdot \mathbb{P} \mathbf{u} + \mathbf{u} \cdot \mathbf{c}
     * \f]
     * under constraint
     * \f[
     *      \mathbf{d} + \mathbb{B} \mathbf{u} \ge \mathbf{f} (\mathbf{u}).
     * \f]
     * The vector \f$ \mathbf{c} \f$ is known and does not depends on the problem.
     * The function \f$ \mathbf{f} \f$ differs with the problem.
     * So does the implentation of the vector \f$ \mathbf{d} \f$ and the matrix \f$ \mathbb{B} \f$.
     * However, they share some elements, they are set by this class.
     *
     * In the documentation of other classes, \f$ N \f$ is the number of particles and \f$ N_c \f$ is the number of contacts.
     *
     * Different solvers can be used to solve the problem, see OptimBase.
     */
    template<class Params>
    class ProblemBase
    {
    public:
        using params_t = Params;

        xt::xtensor<double, 1>& distances();
        std::vector<int>& A_rows();
        std::vector<int>& A_cols();
        std::vector<double>& A_values();

        params_t& get_params();

        std::size_t size() const;

        /**
         * @brief Matrix-free product \f$ \mathbf{u} = \mathbb{P}^{-1} \mathbf{u} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (to get masses and moments of inertia).
         * @param U [in/out] Vector \f$ \mathbf{u} \f$.
         * @param active_offset [in] Index of the first active particle.
         * @param row [in] Row of \f$ \mathbf{u} \f$ to compute.
         */
        template<std::size_t dim>
        void matrix_free_gemv_inv_P(const scopi_container<dim>& particles,
                                    xt::xtensor<double, 1>& U,
                                    std::size_t active_offset,
                                    std::size_t row);
    protected:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of active particles.
         * @param dt [in] Time step.
         */
        ProblemBase(std::size_t nparts, double dt);

        /**
         * @brief COO storage of the shared row of \f$ \mathbb{B} \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (to get the position).
         * @param contacts [in] Array of contacts.
         * @param firstCol [in] Index of the first column (solver-dependent).
         * @param nb_row_per_contact [in] Number of rows per contact (problem-dependent).
         */
        template<std::size_t dim>
        void matrix_positive_distance(const scopi_container<dim>& particles,
                                      const std::vector<neighbor<dim>>& contacts,
                                      std::size_t nb_row_per_contact);

        params_t m_params;

    private:
        /**
         * @brief 2D implementation of rows in matrix-free product \f$ \mathbf{u} = \mathbb{P}^{-1} \mathbf{u} \f$ that involve moments of inertia.
         *
         * Some rows in matrix_free_gemv_inv_P involve the moment of inertia.
         * The implentation is different in 2D or in 3D.
         *
         * @param particles [in] Array of particles (for the moments of inertia).
         * @param U [in/out] Vector \f$ \mathbf{u} \f$.
         * @param active_offset [in] Index of the first active particle.
         * @param row [in] Row of \f$ \mathbf{u} \f$ to compute.
         */
        void matrix_free_gemv_inv_P_moment(const scopi_container<2>& particles,
                                           xt::xtensor<double, 1>& U,
                                           std::size_t active_offset,
                                           std::size_t row);
        /**
         * @brief 3D implementation of rows in matrix-free product \f$ \mathbf{u} = \mathbb{P}^{-1} \mathbf{u} \f$ that involve moments of inertia.
         *
         * Some rows in matrix_free_gemv_inv_P involve the moment of inertia.
         * The implentation is different in 2D or in 3D.
         *
         * @param particles [in] Array of particles (for the moments of inertia).
         * @param U [in/out] Vector \f$ \mathbf{u} \f$.
         * @param active_offset [in] Index of the first active particle.
         * @param row [in] Row of \f$ \mathbf{u} \f$ to compute.
         */
        void matrix_free_gemv_inv_P_moment(const scopi_container<3>& particles,
                                           xt::xtensor<double, 1>& U,
                                           std::size_t active_offset,
                                           std::size_t row);

    protected:
        /**
         * @brief Number of particles.
         */
        std::size_t m_nparticles;
        /**
         * @brief Time step.
         */
        double m_dt;
        /**
         * @brief Rows' indices of \f$ \mathbb{B} \f$ in COO storage.
         *
         * Modified by the problem.
         */
        std::vector<int> m_A_rows;
        /**
         * @brief Columns' indices of \f$ \mathbb{B} \f$ in COO storage.
         *
         * Modified by the problem.
         */
        std::vector<int> m_A_cols;
        /**
         * @brief Values of \f$ \mathbb{B} \f$ in COO storage.
         *
         * Modified by the problem.
         */
        std::vector<double> m_A_values;
        /**
         * @brief Vector \f$ \mathbf{d} \f$.
         *
         * Modified by the problem.
         */
        xt::xtensor<double, 1> m_distances;
        /**
         * @brief Whether the optimization problem should be solved.
         *
         * Modified by the problem.
         * Used to handles problems that require several resolutions of the optimization problem per time step.
         */
        bool m_should_solve;
    };

    template<class Params>
    template<std::size_t dim>
    void ProblemBase<Params>::matrix_free_gemv_inv_P(const scopi_container<dim>& particles,
                                             xt::xtensor<double, 1>& U,
                                             std::size_t active_offset,
                                             std::size_t row)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            U(3*row + d) /= (-1.*particles.m()(active_offset + row));
        }
        matrix_free_gemv_inv_P_moment(particles, U, active_offset, row);
    }

    template<class Params>
    template<std::size_t dim>
    void ProblemBase<Params>::matrix_positive_distance(const scopi_container<dim>& particles,
                                               const std::vector<neighbor<dim>>& contacts,
                                               std::size_t nb_row_per_contact)
    {
        std::size_t active_offset = particles.nb_inactive();
        std::size_t nb_rows = contacts.size();
        m_A_rows.clear();
        m_A_cols.clear();
        m_A_values.clear();
        m_A_rows.reserve(12*nb_rows*nb_row_per_contact);
        m_A_cols.reserve(12*nb_rows*nb_row_per_contact);
        m_A_values.reserve(12*nb_rows*nb_row_per_contact);

        std::size_t ic = 0;
        for (auto &c: contacts)
        {
            if (c.i >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    m_A_rows.push_back(nb_row_per_contact*ic);
                    m_A_cols.push_back((c.i - active_offset)*3 + d);
                    m_A_values.push_back(-m_dt*c.nij[d]);
                }
            }

            if (c.j >= active_offset)
            {
                for (std::size_t d = 0; d < 3; ++d)
                {
                    m_A_rows.push_back(nb_row_per_contact*ic);
                    m_A_cols.push_back((c.j - active_offset)*3 + d);
                    m_A_values.push_back(m_dt*c.nij[d]);
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
                    m_A_rows.push_back(nb_row_per_contact*ic);
                    m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                    m_A_values.push_back(m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }

            if (c.j >= active_offset)
            {
                std::size_t ind_part = c.j - active_offset;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                for (std::size_t ip = 0; ip < 3; ++ip)
                {
                    m_A_rows.push_back(nb_row_per_contact*ic);
                    m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                    m_A_values.push_back(-m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }

            ++ic;
        }
    }

    template<class Params>
    ProblemBase<Params>::ProblemBase(std::size_t nparts, double dt)
    : m_params()
    , m_nparticles(nparts)
    , m_dt(dt)
    {}

    template<class Params>
    auto ProblemBase<Params>::get_params() -> params_t&
    {
        return m_params;
    }

    template<class Params>
    std::size_t ProblemBase<Params>::size() const
    {
        return m_nparticles;
    }

    template<class Params>
    xt::xtensor<double, 1>& ProblemBase<Params>::distances()
    {
        return m_distances;
    }

    template<class Params>
    std::vector<int>& ProblemBase<Params>::A_rows()
    {
        return m_A_rows;
    }



    template<class Params>
    std::vector<int>& ProblemBase<Params>::A_cols()
    {
        return m_A_cols;
    }

    template<class Params>
    std::vector<double>& ProblemBase<Params>::A_values()
    {
        return m_A_values;
    }

    template<class Params>
    void ProblemBase<Params>::matrix_free_gemv_inv_P_moment(const scopi_container<2>& particles,
                                                    xt::xtensor<double, 1>& U,
                                                    std::size_t active_offset,
                                                    std::size_t row)
    {
        auto nparticles = particles.nb_active();
        U(3*nparticles + 3*row + 2) /= (-1.*particles.j()(active_offset + row));
    }

    template<class Params>
    void ProblemBase<Params>::matrix_free_gemv_inv_P_moment(const scopi_container<3>& particles,
                                                    xt::xtensor<double, 1>& U,
                                                    std::size_t active_offset,
                                                    std::size_t row)
    {
        auto nparticles = particles.nb_active();
        for (std::size_t d = 0; d < 3; ++d)
        {
            U(3*nparticles + 3*row + d) /= (-1.*particles.j()(active_offset + row)[d]);
        }
    }
}

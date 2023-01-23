#pragma once

#include <cmath>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <xtensor/xtensor.hpp>
#include <xtensor/xfixed.hpp>

#include "../types.hpp"
#include "../container.hpp"
#include "../quaternion.hpp"
#include "../objects/neighbor.hpp"
#include "../utils.hpp"
#include "../params.hpp"
#include "ProblemBase.hpp"

namespace scopi
{
    /**
     * @brief Exact solution for a sphere falling on an inclined plane in 2D.
     *
     * - Notations:
     *      - We consider two referentials:
     *          - \f$ (x, y) \f$, the usual cartesian referential;
     *          - \f$ (x', y') \f$ the referential where \f$ x' \f$ is aligned with the plane (rotation of angle \f$ \alpha \f$ of \f$ (x, y) \f$.
     *      - \f$ t_i = \sqrt{\frac{2 \left( y_0-r \right)}{g \cos \alpha}} \f$ is the time of the impact between the sphere and the plane.
     *      - \f$ v_t^- \f$ is the tangential velocity just before the impact, \f$ v_t^- = g \sin \alpha t_i \f$.
     *      - \f$ v_n^- \f$ is the normal velocity just before the impact, \f$ v_n^- = - g \cos \alpha t_i \f$.
     *      - \f$ (x_i', r) \f$ is the position of the sphere in the referential \f$ (x', y') \f$ at impact.
     *
     * - If \f$ t \le t_i \f$, the sphere is in free-fall and
     * \f[
     *      \left\{
     *           \begin{aligned}
     *              x(t) &= \frac{g}{2} \sin \alpha t^2\\
     *              y(t) &= y_0 - \frac{g}{2} \cos \alpha t^2\\
     *              \omega(t) &= 0.
     *         \end{aligned}
     *      \right.
     * \f]
     *
     * - If \f$ t \ge t_i \f$, there is two posibility, depending on \f$ \mu \f$ and \f$ \tan \alpha \f$.
     *      - If \f$ \mu \ge \frac{\tan \alpha}{3} \f$ (no slip),
     *      \f[
     *          \left\{
     *              \begin{aligned}
     *                  x'(t) &= \frac{1}{3} g \sin \alpha \left( t - t_i \right) ^2 + \frac{2}{3} v_t^- \left( t - t_i \right) + x_i'\\
     *                  y'(t) &= r\\
     *                  \theta(t) &= - \frac{1}{3} \frac{g \sin \alpha}{r} \left( t - t_i \right)^2 - \frac{2}{3} \frac{v_t^-}{r} \left( t - t_i \right).
     *              \end{aligned}
     *          \right.
     *      \f]
     *      - If \f$ \mu \le \frac{\tan \alpha}{3} \f$ (sliding motion),
     *      \f[
     *          \left\{
     *              \begin{aligned}
     *                  x'(t) &= \frac{1}{2} g \left( \sin \alpha - \mu \cos \alpha \right) \left( t - t_i \right)^2 + \left( v_t^- + \mu v_n^- \right) \left( t - t_i \right)  x_i'\\
     *                  y'(t) &= R\\
     *                  \theta (t) &= - \frac{\mu g \cos \alpha}{r} \left( t - t_i \right)^2 + \frac{2}{r} \mu v_n^- \left( t - t_i \right)
     *              \end{aligned}
     *          \right.
     *      \f]
     *
     * @param alpha [in] Angle between the plane and the horizontal axis. Between \f$ 0 \f$ and \f$ \frac{\pi}{2} \f$.
     * @param mu [in] Friction coefficient.
     * @param t [in] Time.
     * @param r [in] Radius of the spheres.
     * @param g [in] Gravity.
     * @param y0 [in] Iniital height of the center of the sphere.
     *
     * @return Position of the sphere (x and y coordinates) and angle of rotation.
     */
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan(double alpha, double mu, double t, double r, double g, double y0);
    /**
     * @brief Exact velocity for a sphere falling on an inclined plane in 2D.
     *
     * See \c analytical_solution_sphere_plan for the notations.
     *
     * @param alpha [in] Angle between the plane and the horizontal axis. Between \f$ 0 \f$ and \f$ \frac{\pi}{2} \f$.
     * @param mu [in] Friction coefficient.
     * @param t [in] Time.
     * @param r [in] Radius of the spheres.
     * @param g [in] Gravity.
     * @param y0 [in] Iniital height of the center of the sphere.
     *
     * @return Velocity of the sphere (x and y coordinates) and rotation (\f$ \omega \f$).
     */
    std::pair<type::position_t<2>, double> analytical_solution_sphere_plan_velocity(double alpha, double mu, double t, double r, double g, double y0);

    class DryWithFriction;

    /**
     * @brief Shared functions to model contacts with friction and without viscosity.
     *
     * See ProblemBase.hpp for the notations.
     * The constraint is
     * \f[
     *      \d_{\ij} + \B \u_{\ij} \ge \norm{\T \u_{\ij}}
     * \f]
     * for all contacts \f$ (\ij) \f$.
     * \f$ \d \in \R^{\Nc} \f$, \f$ \u \in \R^{6\N} \f$, \f$ \B \in \R^{\Nc \times 6 N} \f$, and \f$ \T \in R^{3 \Nc \times 6\N} \f$.
     *
     * Only one matrix is built.
     * It contains both matrices $\f$ \B \f$ and \f$ T \f$.
     * A contact \f$ (\ij) \f$ corresponds to four rows in the matrix, one for \f$ \B \f$ and three for \f$ T \f$.
     * Therefore, the matrix is in \f$ \R^{4\Nc \times 6N} \f$ and \f$ \d \in \R^{4\Nc} \f$.
     *
     */
    class DryWithFrictionBase : protected ProblemBase
    {
    protected:
        /**
         * @brief Constructor.
         *
         * @param nparticles [in] Number of particles.
         * @param dt [in] Time step.
         * @param mu [in] Friction coefficient.
         */
        DryWithFrictionBase(std::size_t nparticles, double dt, double mu);

        /**
         * @brief Construct the COO storage of the matrices \f$ \B \f$ and \f$ \T \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for positions).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance (for compatibility with other problems).
         * @param firstCol [in] Index of the first column (solver-dependent).
         */
        template <std::size_t dim>
        void create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts,
                                          std::size_t firstCol);
    private:
        /**
         * @brief Friction coefficient.
         */
        double m_mu;
    };

    template<std::size_t dim>
    void DryWithFrictionBase::create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                                           const std::vector<neighbor<dim>>& contacts,
                                                           std::size_t firstCol)
    {
        matrix_positive_distance(particles, contacts, firstCol, 4);
        // std::size_t active_offset = particles.nb_inactive();
        // std::size_t ic = 0;
        // for (auto &c: contacts)
        // {
        //     if (c.i >= active_offset)
        //     {
        //         for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
        //         {
        //             for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
        //             {
        //                 this->m_A_rows.push_back(4*ic + 1 + ind_row);
        //                 this->m_A_cols.push_back(firstCol + (c.i - active_offset)*3 + ind_col);
        //                 this->m_A_values.push_back(-this->m_dt*m_mu*c.nij[ind_row]*c.nij[ind_col]);
        //                 if(ind_row == ind_col)
        //                 {
        //                     this->m_A_values[this->m_A_values.size()-1] += this->m_dt*m_mu;
        //                 }
        //             }
        //         }
        //     }

        //     if (c.j >= active_offset)
        //     {
        //         for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
        //         {
        //             for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
        //             {
        //                 this->m_A_rows.push_back(4*ic + 1 + ind_row);
        //                 this->m_A_cols.push_back(firstCol + (c.j - active_offset)*3 + ind_col);
        //                 this->m_A_values.push_back(this->m_dt*m_mu*c.nij[ind_row]*c.nij[ind_col]);
        //                 if(ind_row == ind_col)
        //                 {
        //                     this->m_A_values[this->m_A_values.size()-1] -= this->m_dt*m_mu;
        //                 }
        //             }
        //         }
        //     }

        //     auto ri_cross = cross_product<dim>(c.pi - particles.pos()(c.i));
        //     auto rj_cross = cross_product<dim>(c.pj - particles.pos()(c.j));
        //     auto Ri = rotation_matrix<3>(particles.q()(c.i));
        //     auto Rj = rotation_matrix<3>(particles.q()(c.j));

        //     if (c.i >= active_offset)
        //     {
        //         std::size_t ind_part = c.i - active_offset;
        //         auto dot = xt::eval(xt::linalg::dot(ri_cross, Ri));
        //         for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
        //         {
        //             for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
        //             {
        //                 this->m_A_rows.push_back(4*ic + 1 + ind_row);
        //                 this->m_A_cols.push_back(firstCol + 3*particles.nb_active() + 3*ind_part + ind_col);
        //                 this->m_A_values.push_back(-m_mu*this->m_dt*dot(ind_row, ind_col) + m_mu*this->m_dt*(c.nij[0]*dot(0, ind_col)+c.nij[1]*dot(1, ind_col)+c.nij[2]*dot(2, ind_col)));
        //             }
        //         }
        //     }

        //     if (c.j >= active_offset)
        //     {
        //         std::size_t ind_part = c.j - active_offset;
        //         auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
        //         for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
        //         {
        //             for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
        //             {
        //                 this->m_A_rows.push_back(4*ic + 1 + ind_row);
        //                 this->m_A_cols.push_back(firstCol + 3*particles.nb_active() + 3*ind_part + ind_col);
        //                 this->m_A_values.push_back(m_mu*this->m_dt*dot(ind_row, ind_col) - m_mu*this->m_dt*(c.nij[0]*dot(0, ind_col)+c.nij[1]*dot(1, ind_col)+c.nij[2]*dot(2, ind_col)));
        //             }
        //         }
        //     }
        //     ++ic;
        // }
    }
}


#pragma once

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <vector>
#include <xtensor/xtensor.hpp>

#include "../container.hpp"
#include "../quaternion.hpp"
#include "../objects/neighbor.hpp"
#include "../utils.hpp"

#include "ProblemBase.hpp"
#include "ViscousBase.hpp"

namespace scopi
{
    template<std::size_t dim>
    class ViscousWithFriction;

    /**
     * @brief Parameters for ViscousWithFriction<dim>.
     *
     * Specialization of ProblemParams in params.hpp
     *
     * @tparam dim Dimension (2 or 3).
     */
    template<std::size_t dim>
    struct ProblemParams<ViscousWithFriction<dim>>
    {
        /**
         * @brief Default constructor.
         */
        ProblemParams();
        /**
         * @brief Copy constructor.
         *
         * @param params Parameters to by copied.
         */
        ProblemParams(const ProblemParams<ViscousWithFriction<dim>>& params);

        /**
         * @brief Friction coefficient.
         *
         * Default value is 0.
         * \note \c mu > 0
         */
        double mu;
        /**
         * @brief \f$ \gm \f$
         *
         * Default value is -3.
         * \note \c gamma_min < 0
         */
        double gamma_min;
        /**
         * @brief Tolerance to consider \f$ \g < 0 \f$ .
         *
         * Default value is \f$ 10^{-6} \f$.
         * \note \c tol > 0
         */
        double tol;
    };

    /**
     * @brief Problem that models contacts without friction and with viscosity.
     *
     * See ProblemBase.hpp for the notations.
     * We introduce the variable \f$ \g \f$ such that, for each contact \f$ \ij \f$,  we impose
     * - \f$ \d_{\ij} + \B \u_{\ij} \ge 0 \f$ if \f$ \g_{\ij} = 0 \f$;
     * - \f$ \d_{\ij} + \B \u_{\ij} = 0 \f$ if \f$ \gm < \g_{\ij} < 0 \f$;
     * - \f$ \d_{\ij} + \B \u_{\ij} \ge ||\T \u_{\ij}|| \f$ if \f$ \g_{\ij} < \gm \f$.
     *
     * \f$ \d \in \mathbb{R}^{\Nc} \f$, \f$ \u \in \mathbb{R}^{6\N} \f$, \f$ \B \in \mathbb{R}^{\Nc \times 6 N} \f$, and \f$ \T \in R^{3 \Nc \times 6\N} \f$.
     *
     * For each contact \f$ \ij \f$, \f$ \g_{\ij} \f$ verifies
     * - \f$ \g_{\ij} = 0 \f$ if particles \c i and \c j are not in contact;
     * - \f$ \frac{\mathrm{d} \g_{\ij}}{\mathrm{d} t} = - \left( \lm_{\ij}^+ - \lm_{\ij}^- \right) \f$ else. 
     *
     * \f$ \lm^+ \f$ (resp. \f$ \lm^- \f$) is the Lagrange multiplier associated with the constraint \f$ \d + \B \u \ge 0 \f$ (resp. \f$ -\d - \B \u \ge 0 \f$).
     * By convention, \f$ \lm^+ \ge 0 \f$ and \f$ \lm^- \ge 0 \f$. 
     *
     * Only one matrix is built.
     * See \c create_vector_distances for the order of the rows of the matrix.
     *
     * @tparam dim Dimension (2 or 3).
     */
    template<std::size_t dim>
    class ViscousWithFriction: protected ProblemBase
                             , protected ViscousBase<dim>
    {
    protected:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         * @param dt [in] Time step.
         * @param problem_params [in] Parameters.
         */
        ViscousWithFriction(std::size_t nparts, double dt, const ProblemParams<ViscousWithFriction<dim>>& problem_params);
        /**
         * @brief Constructor.
         *
         * \todo Is it necessary or is a rest from a previous attempt?
         *
         * @param nparts [in] Number of particles.
         */
        ViscousWithFriction(std::size_t nparts);

        /**
         * @brief Construct the COO storage of the matrices \f$ \B \f$ and \f$ \T \f$.
         *
         * See \c create_vector_distances for the order of the rows of the matrix.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for positions).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance (for compatibility with other problems).
         * @param firstCol [in] Index of the first column (solver-dependent).
         */
        void create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts,
                                          const std::vector<neighbor<dim>>& contacts_worms,
                                          std::size_t firstCol);
        /**
         * @brief Get the number of rows in the matrix.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance (for compatibility with other models).
         *
         * @return Number of rows in the matrix.
         */
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts,
                                      const std::vector<neighbor<dim>>& contacts_worms);
        /**
         * @brief Create vector \f$ \d \f$.
         *
         * For each contact \f$ \ij \f$, depending on the constraint, \f$ \d_{\ij} \f$ can be of the form:
         *  - one element if \f$ \g_{\ij} = 0 \f$;
         *  - four elements if \f$ \g_{\ij} < \gm \f$;
         *  - one element corresponding to \f$ D > 0 \f$ and a second element corresponding to \f$ D < 0 \f$, after all the other constraints, if \f$ \gm < \g_{\ij} < 0 \f$.
         *
         *  In other words, \f$ d \f$ is a block vector like
         *  \f[
         *      \d = \left( \text{mix of blocks with one or four elements corresponding to } D > 0  \text{ or to the friction model, blocks of one element corresponding to } D < 0 \right).
         * \f]
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         * @param contacts_worms [in] Array of contacts to impose non-positive distance (for compatibility with other models).
         */
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts,
                                     const std::vector<neighbor<dim>>& contacts_worms);

        /**
         * @brief Returns the number of contacts \f$ \ij \f$ with \f$ \g_{\ij} < \gm \f$.
         */
        std::size_t get_nb_gamma_min();
        /**
         * @brief Set \f$ \g_{\ij}^n \f$ from the previous time step, compute the number of contacts with \f$ \g_{\ij} < 0 \f$ and \f$ \g_{\ij} < \gm \f$.
         *
         * Look if particles \c i and \c j were already in contact.
         *
         * @param contacts_new
         */
        void extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts_new);
        /**
         * @brief Compute the value of \f$ \g^{n+1} \f$.
         *
         * \f[
         *      \g^{n+1}_{\ij} = \g^n_{\ij} - \Delta t \left( \lm_{\ij}^+ - \lm_{\ij}^- \right).
         * \f]
         *
         * @param contacts [in] Array of contacts.
         * @param lambda [in] Lagrange multipliers.
         * @param u_tilde [in] Vector \f$ \d + \B \u - \constraintFunction(\u) \f$, where \f$ \u \f$ is the solution of the optimization problem.
         */
        void extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                     const xt::xtensor<double, 1>& lambda,
                                     const xt::xtensor<double, 2>& u_tilde);
        /**
         * @brief Whether the optimization problem should be solved.
         *
         * For compatibility with the other problems.
         */
        bool should_solve_optimization_problem();

    private:
        /**
         * @brief For ViscousWithFrictionFixedPoint
         *
         * \todo To be removed.
         *
         * @param contacts
         * @param lambda
         * @param particles
         * @param u
         */
        void correct_lambda(const std::vector<neighbor<dim>>& contacts,
                            xt::xtensor<double, 1> lambda,
                            const scopi_container<dim>& particles,
                            const xt::xtensor<double, 2>& u);
        /**
         * @brief For ViscousWithFrictionFixedPoint.
         *
         * \todo To be removed.
         */
        void setup_first_resolution();
        /**
         * @brief For ViscousWithFrictionFixedPoint.
         *
         * \todo To be removed.
         */
        void setup_projection();

        /**
         * @brief Number of contacts \f$ \ij \f$ with \f$ \g_{\ij} < \gm \f$.
         */
        std::size_t m_nb_gamma_min;
        /**
         * @brief For ViscousWithFrictionFixedPoint.
         *
         * \todo To be removed.
         */
        std::vector<double> m_lambda;
        /**
         * @brief For ViscousWithFrictionFixedPoint.
         *
         * \todo To be removed.
         */
        bool m_projection;

        /**
         * @brief Parameters
         */
        ProblemParams<ViscousWithFriction<dim>> m_params;
    };

    template<std::size_t dim>
    void ViscousWithFriction<dim>::create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                                                const std::vector<neighbor<dim>>& contacts,
                                                                const std::vector<neighbor<dim>>& contacts_worms,
                                                                std::size_t firstCol)
    {
        std::size_t active_offset = particles.nb_inactive();
        std::size_t size = 6 * this->number_row_matrix(contacts, contacts_worms);
        this->m_A_rows.resize(size);
        this->m_A_cols.resize(size);
        this->m_A_values.resize(size);

        std::size_t ic = 0;
        std::size_t index = 0;
        for (auto &c: contacts)
        {
            if (this->m_gamma[ic] != m_params.gamma_min || m_projection)
            {
                if (c.i >= active_offset)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows[index] = ic;
                        this->m_A_cols[index] = firstCol + (c.i - active_offset)*3 + d;
                        this->m_A_values[index] = -this->m_dt*c.nij[d];
                        index++;
                        if (this->m_gamma[ic] < -m_params.tol)
                        {
                            this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + ic;
                            this->m_A_cols[index] = firstCol + (c.i - active_offset)*3 + d;
                            this->m_A_values[index] = this->m_dt*c.nij[d];
                            index++;
                        }
                    }
                }

                if (c.j >= active_offset)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows[index] = ic;
                        this->m_A_cols[index] = firstCol + (c.j - active_offset)*3 + d;
                        this->m_A_values[index] = this->m_dt*c.nij[d];
                        index++;
                        if (this->m_gamma[ic] < -m_params.tol)
                        {
                            this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + ic;
                            this->m_A_cols[index] = firstCol + (c.j - active_offset)*3 + d;
                            this->m_A_values[index] = -this->m_dt*c.nij[d];
                            index++;
                        }
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
                        this->m_A_rows[index] = ic;
                        this->m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                        this->m_A_values[index] = this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                        index++;
                        if (this->m_gamma[ic] < -m_params.tol)
                        {
                            this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + ic;
                            this->m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                            this->m_A_values[index] = -this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                            index++;
                        }
                    }
                }

                if (c.j >= active_offset)
                {
                    std::size_t ind_part = c.j - active_offset;
                    auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                    for (std::size_t ip = 0; ip < 3; ++ip)
                    {
                        this->m_A_rows[index] = ic;
                        this->m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                        this->m_A_values[index] = -this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                        index++;
                        if (this->m_gamma[ic] < -m_params.tol)
                        {
                            this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + ic;
                            this->m_A_cols[index] = firstCol + 3*particles.nb_active() + 3*ind_part + ip;
                            this->m_A_values[index] = this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                            index++;
                        }
                    }
                }

            }
            else
            {
                if (c.i >= active_offset)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*ic;
                        this->m_A_cols[index] = firstCol + (c.i - active_offset)*3 + d;
                        this->m_A_values[index] = -this->m_dt*c.nij[d];
                        index++;
                    }
                    for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
                    {
                        for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
                        {
                            this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*ic + 1 + ind_row;
                            this->m_A_cols[index] = firstCol + (c.i - active_offset)*3 + ind_col;
                            this->m_A_values[index] = -this->m_dt*m_params.mu*c.nij[ind_row]*c.nij[ind_col];
                            if(ind_row == ind_col)
                            {
                                this->m_A_values[index] += this->m_dt*m_params.mu;
                            }
                            index++;
                        }
                    }
                }

                if (c.j >= active_offset)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*ic;
                        this->m_A_cols[index] = firstCol + (c.j - active_offset)*3 + d;
                        this->m_A_values[index] = this->m_dt*c.nij[d];
                        index++;
                    }
                    for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
                    {
                        for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
                        {
                            this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*ic + 1 + ind_row;
                            this->m_A_cols[index] = firstCol + (c.j - active_offset)*3 + ind_col;
                            this->m_A_values[index] = this->m_dt*m_params.mu*c.nij[ind_row]*c.nij[ind_col];
                            if(ind_row == ind_col)
                            {
                                this->m_A_values[index] -= this->m_dt*m_params.mu;
                            }
                            index++;
                        }
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
                        this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*ic;
                        this->m_A_cols[index] = firstCol + 3*this->m_nparticles + 3*ind_part + ip;
                        this->m_A_values[index] = this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                        index++;
                    }
                    for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
                    {
                        for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
                        {
                            this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*ic + 1 + ind_row;
                            this->m_A_cols[index] = firstCol + 3*this->m_nparticles + 3*ind_part + ind_col;
                            this->m_A_values[index] = -m_params.mu*this->m_dt*dot(ind_row, ind_col) + m_params.mu*this->m_dt*(c.nij[0]*dot(0, ind_col)+c.nij[1]*dot(1, ind_col)+c.nij[2]*dot(2, ind_col));
                            index++;
                        }
                    }
                }

                if (c.j >= active_offset)
                {
                    std::size_t ind_part = c.j - active_offset;
                    auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                    for (std::size_t ip = 0; ip < 3; ++ip)
                    {
                        this->m_A_rows[index] = contacts.size() - m_nb_gamma_min +this-> m_nb_gamma_neg + 4*ic;
                        this->m_A_cols[index] = firstCol + 3*this->m_nparticles + 3*ind_part + ip;
                        this->m_A_values[index] = -this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip));
                        index++;
                    }
                    for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
                    {
                        for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
                        {
                            this->m_A_rows[index] = contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*ic + 1 + ind_row;
                            this->m_A_cols[index] = firstCol + 3*this->m_nparticles + 3*ind_part + ind_col;
                            this->m_A_values[index] = m_params.mu*this->m_dt*dot(ind_row, ind_col) - m_params.mu*this->m_dt*(c.nij[0]*dot(0, ind_col)+c.nij[1]*dot(1, ind_col)+c.nij[2]*dot(2, ind_col));
                            index++;
                        }
                    }
                }
            }
            ++ic;
        }
        this->m_A_rows.resize(index);
        this->m_A_cols.resize(index);
        this->m_A_values.resize(index);
    }

    template<std::size_t dim>
    void ViscousWithFriction<dim>::setup_first_resolution()
    {
        m_projection = false;
    }

    template<std::size_t dim>
    void ViscousWithFriction<dim>::setup_projection()
    {
        m_projection = true;
        this->m_nb_gamma_neg += m_nb_gamma_min;
        m_nb_gamma_min = 0;
    }

    template<std::size_t dim>
    ViscousWithFriction<dim>::ViscousWithFriction(std::size_t nparticles, double dt, const ProblemParams<ViscousWithFriction<dim>>& problem_params)
    : ProblemBase(nparticles, dt)
    , ViscousBase<dim>()
    , m_params(problem_params)
    {}

    template<std::size_t dim>
    ViscousWithFriction<dim>::ViscousWithFriction(std::size_t nparticles)
    : ProblemBase(nparticles, 0.)
    , ViscousBase<dim>()
    , m_params()
    {}

    template<std::size_t dim>
    void ViscousWithFriction<dim>::extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts_new)
    {
        this->m_should_solve = true;
        this->set_gamma_base(contacts_new);
        this->m_nb_gamma_neg = 0;
        m_nb_gamma_min = 0;
        for (auto& g : this->m_gamma)
        {
            if (g < -m_params.tol && g > m_params.gamma_min)
            {
                this->m_nb_gamma_neg++;
            }
            else if (g == m_params.gamma_min)
            {
                m_nb_gamma_min++;
            }
        }
    }

    template<std::size_t dim>
    void ViscousWithFriction<dim>::correct_lambda(const std::vector<neighbor<dim>>& contacts,
                                                  xt::xtensor<double, 1> lambda,
                                                  const scopi_container<dim>& particles,
                                                  const xt::xtensor<double, 2>& u)
    {
        // TODO will work only for sphere and plan test
        m_lambda.resize(contacts.size());
        std::size_t ind_gamma_neg = 0;
        std::size_t ind_gamma_min = 0;

        for (std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            if (this->m_gamma[ic] != m_params.gamma_min)
            {
                if (this->m_gamma[ic] < - m_params.tol)
                {
                    m_lambda[ic] = lambda(ic) - lambda(this->m_gamma.size() - m_nb_gamma_min + ind_gamma_neg);
                    ind_gamma_neg++;
                }
                else
                {
                    m_lambda[ic] = lambda(ic);
                }
            }
            else
            {
                m_lambda[ic] = lambda(this->m_gamma.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*ind_gamma_min);
                if (m_lambda[ic] < m_params.tol)
                {
                    m_lambda[ic] = + xt::linalg::dot(xt::view(u, contacts[ic].j, xt::range(_, dim)) - particles.v()(contacts[ic].j), contacts[ic].nij)(0)/this->m_dt;
                }
                ind_gamma_min++; 
            }
        }
    }

    template<std::size_t dim>
    void ViscousWithFriction<dim>::extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                                           const xt::xtensor<double, 1>&,
                                                           const xt::xtensor<double, 2>&)
    {
        this->m_should_solve = false;
        this->m_contacts_old = contacts;
        this->m_gamma_old.resize(this->m_gamma.size());
        for (std::size_t ic = 0; ic < this->m_gamma.size(); ++ic)
        {
            this->m_gamma_old[ic] = std::max(m_params.gamma_min, std::min(0., this->m_gamma[ic] - this->m_dt * m_lambda[ic]));
            // for Mosek
            if (this->m_gamma_old[ic] - m_params.gamma_min < m_params.tol)
                this->m_gamma_old[ic] = m_params.gamma_min;
            if (this->m_gamma_old[ic] > -m_params.tol)
                this->m_gamma_old[ic] = 0.;
            PLOG_WARNING << this->m_gamma[ic];
        }
    }

    template<std::size_t dim>
    std::size_t ViscousWithFriction<dim>::number_row_matrix(const std::vector<neighbor<dim>>& contacts, 
                                                            const std::vector<neighbor<dim>>&)
    {
        return contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*m_nb_gamma_min;
    }

    template<std::size_t dim>
    void ViscousWithFriction<dim>::create_vector_distances(const std::vector<neighbor<dim>>& contacts,
                                                           const std::vector<neighbor<dim>>&)
    {
        this->m_distances = xt::zeros<double>({contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*m_nb_gamma_min});
        std::size_t index_dry = 0;
        std::size_t index_friciton = 0;
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            if (this->m_gamma[i] != m_params.gamma_min || m_projection)
            {
                this->m_distances[index_dry] = contacts[i].dij;
                if(this->m_gamma[i] < -m_params.tol)
                {
                    this->m_distances[contacts.size() - m_nb_gamma_min + index_dry] = -contacts[i].dij;
                }
                index_dry++;
            }
            else
            {
                this->m_distances[contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*index_friciton] = contacts[i].dij;
                index_friciton++;
            }
        }
    }

    template<std::size_t dim>
    std::size_t ViscousWithFriction<dim>::get_nb_gamma_min()
    {
        return m_nb_gamma_min;
    }

    template<std::size_t dim>
    bool ViscousWithFriction<dim>::should_solve_optimization_problem()
    {
        return this->m_should_solve;
    }

    template<std::size_t dim>
    ProblemParams<ViscousWithFriction<dim>>::ProblemParams()
    : mu(0.)
    , gamma_min(-3.)
    , tol(1e-6)
    {}

    template<std::size_t dim>
    ProblemParams<ViscousWithFriction<dim>>::ProblemParams(const ProblemParams<ViscousWithFriction<dim>>& params)
    : mu(params.mu)
    , gamma_min(params.gamma_min)
    , tol(params.tol)
    {}

}


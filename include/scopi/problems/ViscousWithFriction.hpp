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
     * @class ProblemParams<ViscousWithFriction>
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
         * @brief \f$ \gamma_{\min} \f$
         *
         * Default value is -3.
         * \note \c gamma_min < 0
         */
        double gamma_min;
        /**
         * @brief Tolerance to consider \f$ \gamma < 0 \f$ .
         *
         * Default value is \f$ 10^{-6} \f$.
         * \note \c tol > 0
         */
        double tol;
        /**
         * @brief Tolerance for the fixed point algorithm.
         *
         * Default value is \f$ 10^{-2} \f$.
         * \note \c tol_fixed_point > 0
         */
        double tol_fixed_point;
        /**
         * @brief Maximum number of iterations for the fixed point algorithm.
         *
         * Default value is 20.
         * \note \c max_iter_fixed_point > 0
         */
        std::size_t max_iter_fixed_point;
    };

    /**
     * @class ViscousWithFriction
     * @brief Problem that models contacts without friction and with viscosity.
     *
     * See ProblemBase for the notations.
     * We introduce the variable \f$ \gamma \f$ such that, for each contact \f$ ij \f$,  we impose
     * - \f$ \mathbf{d}_{ij} + \mathbb{B} \mathbf{u}_{ij} \ge 0 \f$ if \f$ \gamma_{ij} = 0 \f$;
     * - \f$ \mathbf{d}_{ij} + \mathbb{B} \mathbf{u}_{ij} = 0 \f$ if \f$ \gamma_{\min} < \gamma_{ij} < 0 \f$;
     * - \f$ \mathbf{d}_{ij} + \mathbb{B} \mathbf{u}_{ij} \ge ||\mathbb{T} \mathbf{u}_{ij}|| \f$ if \f$ \gamma_{ij} < \gamma_{\min} \f$.
     *
     * \f$ \mathbf{d} \in \mathbb{R}^{N_c} \f$, \f$ \mathbf{u} \in \mathbb{R}^{6N} \f$, \f$ \mathbb{B} \in \mathbb{R}^{N_c \times 6 N} \f$, and \f$ \mathbb{T} \in R^{3 N_c \times 6N} \f$.
     *
     * For each contact \f$ ij \f$, \f$ \gamma_{ij} \f$ verifies
     * - \f$ \gamma_{ij} = 0 \f$ if particles \c i and \c j are not in contact;
     * - \f$ \frac{\mathrm{d} \gamma_{ij}}{\mathrm{d} t} = - \left( \mathbf{\lambda}_{ij}^+ - \mathbf{\lambda}_{ij}^- \right) \f$ else.
     *
     * \f$ \mathbf{\lambda}^+ \f$ (resp. \f$ \mathbf{\lambda}^- \f$) is the Lagrange multiplier associated with the constraint \f$ \mathbf{d} + \mathbb{B} \mathbf{u} \ge 0 \f$ (resp. \f$ -\mathbf{d} - \mathbb{B} \mathbf{u} \ge 0 \f$).
     * By convention, \f$ \mathbf{\lambda}^+ \ge 0 \f$ and \f$ \mathbf{\lambda}^- \ge 0 \f$.
     *
     * Only one matrix is built.
     * See \c create_vector_distances for the order of the rows of the matrix.
     *
     * @tparam dim Dimension (2 or 3).
     */
    template<std::size_t dim>
    class ViscousWithFriction: public ProblemBase<ProblemParams<ViscousWithFriction<dim>>>
                             , public ViscousBase<dim>
    {
    public:
        /**
         * @brief Constructor.
         *
         * @param nparts [in] Number of particles.
         * @param dt [in] Time step.
         * @param problethis->m_params [in] Parameters.
         */
        ViscousWithFriction(std::size_t nparts, double dt);
        /**
         * @brief Constructor.
         *
         * \todo Is it necessary or is a rest from a previous attempt?
         *
         * @param nparts [in] Number of particles.
         */
        ViscousWithFriction(std::size_t nparts);

        /**
         * @brief Construct the COO storage of the matrices \f$ \mathbb{B} \f$ and \f$ \mathbb{T} \f$.
         *
         * See \c create_vector_distances for the order of the rows of the matrix.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for positions).
         * @param contacts [in] Array of contacts.
         */
        void create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                          const std::vector<neighbor<dim>>& contacts);
        /**
         * @brief Get the number of rows in the matrix.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         *
         * @return Number of rows in the matrix.
         */
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts) const;
        /**
         * @brief Create vector \f$ \mathbf{d} \f$.
         *
         * For each contact \f$ ij \f$, depending on the constraint, \f$ \mathbf{d}_{ij} \f$ can be of the form:
         *  - one element if \f$ \gamma_{ij} = 0 \f$;
         *  - four elements if \f$ \gamma_{ij} < \gamma_{\min} \f$;
         *  - one element corresponding to \f$ D > 0 \f$ and a second element corresponding to \f$ D < 0 \f$, after all the other constraints, if \f$ \gamma_{\min} < \gamma_{ij} < 0 \f$.
         *
         *  In other words, \f$ d \f$ is a block vector like
         *  \f[
         *      \mathbf{d} = \left( \text{mix of blocks with one or four elements corresponding to } D > 0  \text{ or to the friction model, blocks of one element corresponding to } D < 0 \right).
         * \f]
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         */
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts);

        /**
         * @brief Returns the number of contacts \f$ ij \f$ with \f$ \gamma_{ij} < \gamma_{\min} \f$.
         */
        std::size_t get_nb_gamma_min() const;
        /**
         * @brief Set \f$ \gamma_{ij}^n \f$ from the previous time step, compute the number of contacts with \f$ \gamma_{ij} < 0 \f$ and \f$ \gamma_{ij} < \gamma_{\min} \f$.
         *
         * Look if particles \c i and \c j were already in contact.
         *
         * @param contacts_new
         */
        template<class optim_solver_t>
        void extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts_new, optim_solver_t&);
        /**
         * @brief Compute the value of \f$ \gamma^{n+1} \f$.
         *
         * \f[
         *      \gamma^{n+1}_{ij} = \max \left( \gamma_{\min}, \gamma^n_{ij} - \Delta t \left( \mathbf{\lambda}_{ij}^+ - \mathbf{\lambda}_{ij}^- \right) \right).
         * \f]
         *
         * @param contacts [in] Array of contacts.
         * @param lambda [in] Lagrange multipliers.
         * @param u_tilde [in] Vector \f$ \mathbf{d} + \mathbb{B} \mathbf{u} - \mathbf{f}(\mathbf{u}) \f$, where \f$ \mathbf{u} \f$ is the solution of the optimization problem.
         */
        template<class optim_solver_t>
        void extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                     optim_solver_t& solver);
        /**
         * @brief Whether the optimization problem should be solved.
         *
         * For compatibility with the other problems.
         */
        bool should_solve() const;

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
         * @brief Number of contacts \f$ ij \f$ with \f$ \gamma_{ij} < \gamma_{\min} \f$.
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
         * @brief \f$ \mathbf{s}^{k+1} \f$.
         */
        xt::xtensor<double, 1> m_s;
        /**
         * @brief \f$ \mathbf{s}^{k} \f$.
         */
        xt::xtensor<double, 1> m_s_old;
        /**
         * @brief Number of iterations in the fixed point algorithm (\f$ k \f$).
         */
        std::size_t m_nb_iter;

    };

    template<std::size_t dim>
    void ViscousWithFriction<dim>::create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                                                const std::vector<neighbor<dim>>& contacts)
    {
        //Set up A matrix
        std::size_t active_offset = particles.nb_inactive();

        std::size_t nb_rows = this->number_row_matrix(contacts);
        this->m_A_rows.clear();
        this->m_A_cols.clear();
        this->m_A_values.clear();
        this->m_A_rows.reserve(12*nb_rows);
        this->m_A_cols.reserve(12*nb_rows);
        this->m_A_values.reserve(12*nb_rows);

        //Set up for the loop on contacts
        std::size_t ic = 0;
        std::size_t i_gamma = 0;
        std::size_t i_dry = 0;
        std::size_t i_friction = 0;

        for (auto &c: contacts)
        {
            //No Friction
            if (this->m_gamma[ic] != this->m_params.gamma_min)
            {
                if (c.i >= active_offset)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows.push_back(i_dry);
                        this->m_A_cols.push_back((c.i - active_offset)*3 + d);
                        this->m_A_values.push_back(-this->m_dt*c.nij[d]);
                        //Viscous
                        if (this->m_gamma[ic] < -this->m_params.tol)
                        {
                            this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + i_gamma);
                            this->m_A_cols.push_back((c.i - active_offset)*3 + d);
                            this->m_A_values.push_back(this->m_dt*c.nij[d]);
                        }
                    }
                }

                if (c.j >= active_offset)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows.push_back(i_dry);
                        this->m_A_cols.push_back((c.j - active_offset)*3 + d);
                        this->m_A_values.push_back(this->m_dt*c.nij[d]);
                        //Viscous
                        if (this->m_gamma[ic] < -this->m_params.tol)
                        {
                            this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + i_gamma);
                            this->m_A_cols.push_back((c.j - active_offset)*3 + d);
                            this->m_A_values.push_back(-this->m_dt*c.nij[d]);
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
                        this->m_A_rows.push_back(i_dry);
                        this->m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                        this->m_A_values.push_back(this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                        //Viscous
                        if (this->m_gamma[ic] < -this->m_params.tol)
                        {
                            this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + i_gamma);
                            this->m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                            this->m_A_values.push_back(-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                        }
                    }
                }

                if (c.j >= active_offset)
                {
                    std::size_t ind_part = c.j - active_offset;
                    auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                    for (std::size_t ip = 0; ip < 3; ++ip)
                    {
                        this->m_A_rows.push_back(i_dry);
                        this->m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                        this->m_A_values.push_back(-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                        //Viscous
                        if (this->m_gamma[ic] < -this->m_params.tol)
                        {
                            this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + i_gamma);
                            this->m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                            this->m_A_values.push_back(this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                        }
                    }
                }
                i_dry++;
                if (this->m_gamma[ic] < -this->m_params.tol)
                {
                    i_gamma++;
                }
            }
            else
            {
                if (c.i >= active_offset)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*i_friction);
                        this->m_A_cols.push_back((c.i - active_offset)*3 + d);
                        this->m_A_values.push_back(-this->m_dt*c.nij[d]);
                    }
                    for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
                    {
                        for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
                        {
                            this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*i_friction + 1 + ind_row);
                            this->m_A_cols.push_back((c.i - active_offset)*3 + ind_col);
                            this->m_A_values.push_back(-this->m_dt*this->m_params.mu*c.nij[ind_row]*c.nij[ind_col]);
                            if(ind_row == ind_col)
                            {
                                this->m_A_values[this->m_A_values.size()-1] += this->m_dt*this->m_params.mu;
                            }
                        }
                    }
                    //Viscous
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows.push_back(contacts.size() -m_nb_gamma_min + i_gamma);
                        this->m_A_cols.push_back((c.i - active_offset)*3 + d);
                        this->m_A_values.push_back(this->m_dt*c.nij[d]);
                    }
                }

                if (c.j >= active_offset)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*i_friction);
                        this->m_A_cols.push_back((c.j - active_offset)*3 + d);
                        this->m_A_values.push_back(this->m_dt*c.nij[d]);
                    }
                    for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
                    {
                        for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
                        {
                            this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*i_friction + 1 + ind_row);
                            this->m_A_cols.push_back((c.j - active_offset)*3 + ind_col);
                            this->m_A_values.push_back(this->m_dt*this->m_params.mu*c.nij[ind_row]*c.nij[ind_col]);
                            if(ind_row == ind_col)
                            {
                                this->m_A_values[this->m_A_values.size()-1] -= this->m_dt*this->m_params.mu;
                            }
                        }
                    }
                    //Viscous
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + i_gamma);
                        this->m_A_cols.push_back((c.j - active_offset)*3 + d);
                        this->m_A_values.push_back(-this->m_dt*c.nij[d]);
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
                        this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*i_friction);
                        this->m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                        this->m_A_values.push_back(this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                    for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
                    {
                        for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
                        {
                            this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*i_friction + 1 + ind_row);
                            this->m_A_cols.push_back(3*this->m_nparticles + 3*ind_part + ind_col);
                            this->m_A_values.push_back(-this->m_params.mu*this->m_dt*dot(ind_row, ind_col) + this->m_params.mu*this->m_dt*c.nij[ind_row]*(c.nij[0]*dot(0, ind_col)+c.nij[1]*dot(1, ind_col)+c.nij[2]*dot(2, ind_col)));
                        }
                    }
                    //Viscous
                    for (std::size_t ip = 0; ip < 3; ++ip)
                    {
                        this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + i_gamma);
                        this->m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                        this->m_A_values.push_back(-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                }

                if (c.j >= active_offset)
                {
                    std::size_t ind_part = c.j - active_offset;
                    auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                    for (std::size_t ip = 0; ip < 3; ++ip)
                    {
                        this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*i_friction);
                        this->m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                        this->m_A_values.push_back(-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                    for (std::size_t ind_row = 0; ind_row < 3; ++ind_row)
                    {
                        for (std::size_t ind_col = 0; ind_col < 3; ++ind_col)
                        {
                            this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*i_friction + 1 + ind_row);
                            this->m_A_cols.push_back(3*this->m_nparticles + 3*ind_part + ind_col);
                            this->m_A_values.push_back(this->m_params.mu*this->m_dt*dot(ind_row, ind_col) - this->m_params.mu*this->m_dt*c.nij[ind_row]*(c.nij[0]*dot(0, ind_col)+c.nij[1]*dot(1, ind_col)+c.nij[2]*dot(2, ind_col)));
                        }
                    }
                    //Viscous
                    for (std::size_t ip = 0; ip < 3; ++ip)
                    {
                        this->m_A_rows.push_back(contacts.size() - m_nb_gamma_min + i_gamma);
                        this->m_A_cols.push_back(3*particles.nb_active() + 3*ind_part + ip);
                        this->m_A_values.push_back(this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                }
                i_gamma++;
                i_friction++;
            }
            ++ic;
        }
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
    ViscousWithFriction<dim>::ViscousWithFriction(std::size_t nparticles, double dt)
    : ProblemBase<ProblemParams<ViscousWithFriction<dim>>>(nparticles, dt)
    , ViscousBase<dim>()
    {}

    template<std::size_t dim>
    ViscousWithFriction<dim>::ViscousWithFriction(std::size_t nparticles)
    : ProblemBase<ProblemParams<ViscousWithFriction<dim>>>(nparticles, 0.)
    , ViscousBase<dim>()
    {}

    template<std::size_t dim>
    template<class optim_solver_t>
    void ViscousWithFriction<dim>::extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts_new, optim_solver_t&)
    {
        //Set up gamma for viscous
        this->m_should_solve = true;
        this->set_gamma_base(contacts_new);
        this->m_nb_gamma_neg = 0;
        m_nb_gamma_min = 0;

        for (auto& g : this->m_gamma)
        {
            if (g < -this->m_params.tol)
            {
                this->m_nb_gamma_neg++;
            }
            if (g == this->m_params.gamma_min)
            {
                m_nb_gamma_min++;
            }
        }
        //Set up fixed point problem
        m_nb_iter = 0;
        m_s_old = xt::zeros<double>({m_nb_gamma_min});
        m_s = xt::ones<double>({m_nb_gamma_min});
    }

    template<std::size_t dim>
    void ViscousWithFriction<dim>::correct_lambda(const std::vector<neighbor<dim>>& contacts,
                                                  xt::xtensor<double, 1> lambda,
                                                  const scopi_container<dim>& particles,
                                                  const xt::xtensor<double, 2>& u)
    {
        m_lambda.resize(contacts.size());
        std::size_t ind_gamma_neg = 0;
        std::size_t ind_gamma_min = 0;

        for (std::size_t ic = 0; ic < contacts.size(); ++ic)
        {
            if (this->m_gamma[ic] != this->m_params.gamma_min)
            {
                if (this->m_gamma[ic] < - this->m_params.tol)
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
                if (m_lambda[ic] < this->m_params.tol)
                {
                    m_lambda[ic] = + xt::linalg::dot(xt::view(u, contacts[ic].j, xt::range(_, dim)) - particles.v()(contacts[ic].j), contacts[ic].nij)(0)/this->m_dt;
                }
                ind_gamma_min++;
            }
        }
    }

    template<std::size_t dim>
    template<class optim_solver_t>
    void ViscousWithFriction<dim>::extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                                           optim_solver_t& optim_solver)
    {
        if(contacts.size() == 0 )
        {
            this->m_should_solve = false;
            return;
        }
        m_nb_iter++;
        m_s_old = m_s;
        if(m_nb_gamma_min != 0)
        {
            auto data = optim_solver.constraint_data() + contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg;
            xt::xtensor<double, 2> u_tilde;
            if (data)
            {
                u_tilde = xt::adapt(reinterpret_cast<double*>(data ), {m_nb_gamma_min, 4UL});
            }
            // TODO use xtensor functions to avoid loop
            for (std::size_t i = 0; i < m_nb_gamma_min; ++i)
            {
                m_s(i) = xt::linalg::norm(xt::view(u_tilde, i, xt::range(1, _)))/(this->m_dt*this->m_params.mu);
            }
            this->m_should_solve = (xt::linalg::norm(m_s_old - m_s)/(xt::linalg::norm(m_s)+1.) > this->m_params.tol_fixed_point && m_nb_iter < this->m_params.max_iter_fixed_point);
        }
        else
        {
            this->m_should_solve = false;
        }
        if (!this->m_should_solve)
        {
            if (m_nb_iter == this->m_params.max_iter_fixed_point)
            {
                std::cout << "ATTENTION PT FIXE N'A PAS CONVERGE" << std::endl;
            }
            auto lambda = optim_solver.get_lagrange_multiplier(contacts);
            this->m_contacts_old = contacts;
            this->m_gamma_old.resize(this->m_gamma.size());
            std::size_t ind_gamma_dry = 0;
            std::size_t ind_gamma_neg = 0;
            std::size_t ind_gamma_min = 0;
            for (std::size_t i = 0; i < this->m_gamma_old.size(); ++i)
            {
                double f_contact;
                if (this->m_gamma[i]!=this->m_params.gamma_min)
                {
                    if (this->m_gamma[i] < -this->m_params.tol)
                    {
                        f_contact = lambda(ind_gamma_dry) - lambda(this->m_gamma.size() - m_nb_gamma_min + ind_gamma_neg);
                        ind_gamma_neg++;
                        ind_gamma_dry++;
                    }
                    else
                    {
                        f_contact = lambda(ind_gamma_dry);
                        ind_gamma_dry++;
                    }
                }
                else
                {
                    f_contact = lambda(this->m_gamma.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*ind_gamma_min) - lambda(this->m_gamma.size() - m_nb_gamma_min + ind_gamma_neg);
                    ind_gamma_neg++;
                    ind_gamma_min++;
                }
                this->m_gamma_old[i] = std::max(this->m_params.gamma_min, std::min(0., this->m_gamma[i] - this->m_dt * f_contact));
                // for Mosek
                if (this->m_gamma_old[i] - this->m_params.gamma_min < this->m_params.tol)
                    this->m_gamma_old[i] = this->m_params.gamma_min;
                if (this->m_gamma_old[i] > -this->m_params.tol)
                    this->m_gamma_old[i] = 0.;
                PLOG_INFO << this->m_gamma[i];
            }
        }
    }

    template<std::size_t dim>
    std::size_t ViscousWithFriction<dim>::number_row_matrix(const std::vector<neighbor<dim>>& contacts) const
    {
        return contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*m_nb_gamma_min;
    }

    template<std::size_t dim>
    void ViscousWithFriction<dim>::create_vector_distances(const std::vector<neighbor<dim>>& contacts)
    {
        this->m_distances = xt::zeros<double>({contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*m_nb_gamma_min});
        std::size_t index_dry = 0;
        std::size_t index_friciton = 0;
        std::size_t index_gamma_neg =0;

        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            //NoFriction
            if (this->m_gamma[i] != this->m_params.gamma_min)
            {
                this->m_distances[index_dry] = contacts[i].dij;
                index_dry++;
            }
            //Viscous
            if(this->m_gamma[i] < -this->m_params.tol)
            {
               this->m_distances[contacts.size() - m_nb_gamma_min + index_gamma_neg] = -contacts[i].dij;
               std::cout << "distance visqueux " << i << " = " << - contacts[i].dij <<std::endl;
               index_gamma_neg++;
            }
            //Friction
            if (this->m_gamma[i] == this->m_params.gamma_min)
            {
                this->m_distances[contacts.size() - m_nb_gamma_min + this->m_nb_gamma_neg + 4*index_friciton] = contacts[i].dij+ this->m_params.mu*this->m_dt*m_s(i);
                index_friciton++;
            }
        }
    }

    template<std::size_t dim>
    std::size_t ViscousWithFriction<dim>::get_nb_gamma_min() const
    {
        return m_nb_gamma_min;
    }

    template<std::size_t dim>
    bool ViscousWithFriction<dim>::should_solve() const
    {
        return this->m_should_solve;
    }

    template<std::size_t dim>
    ProblemParams<ViscousWithFriction<dim>>::ProblemParams()
    : mu(0.1)
    , gamma_min(-3.)
    , tol(1e-6)
    , tol_fixed_point(1e-2)
    , max_iter_fixed_point(20)

    {}

    template<std::size_t dim>
    ProblemParams<ViscousWithFriction<dim>>::ProblemParams(const ProblemParams<ViscousWithFriction<dim>>& params)
    : mu(params.mu)
    , gamma_min(params.gamma_min)
    , tol(params.tol)
    , tol_fixed_point(params.tol_fixed_point)
    , max_iter_fixed_point(params.max_iter_fixed_point)
    {}

}


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
    class ViscousWithoutFriction;

    /**
     * @class ProblemParams<ViscousWithoutFriction>
     * @brief Parameters for ViscousWithoutFriction<dim>.
     *
     * Specialization of ProblemParams in params.hpp
     *
     * @tparam dim Dimension (2 or 3).
     */
    template<std::size_t dim>
    struct ProblemParams<ViscousWithoutFriction<dim>>
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
        ProblemParams(const ProblemParams<ViscousWithoutFriction<dim>>& params);

        /**
         * @brief Tolerance to consider \f$ \gamma < 0 \f$ .
         *
         * Default value is \f$ 10^{-6} \f$.
         * \note \c tol > 0
         */
        double tol;
    };

    /**
     * @class ViscousWithoutFriction
     * @brief Problem that models contacts without friction and with viscosity.
     *
     * See ProblemBase for the notations.
     * The constraint is
     * \f[
     *      \mathbf{d} + \mathbb{B} \mathbf{u} \ge 0,
     * \f]
     * with \f$ \mathbf{d} \in \mathbb{R}^{N_c} \f$, \f$ \mathbf{u} \in \mathbb{R}^{6N} \f$, and \f$ \mathbb{B} \in \mathbb{R}^{N_c \times 6 N} \f$.
     * We impose that the distance between all the particles should be non-negative.
     * We also consider the variable \f$ \gamma \f$, such that we impose
     * - \f$ D_{ij} > 0 \f$ if \f$ \gamma_{ij} = 0 \f$;
     * - \f$ D_{ij} = 0 \f$ if \f$ \gamma_{ij} < 0 \f$.
     *
     * For each contact \f$ ij \f$, \f$ \gamma_{ij} \f$ verifies
     * - \f$ \gamma_{ij} = 0 \f$ if particles \c i and \c j are not in contact;
     * - \f$ \frac{\mathrm{d} \gamma_{ij}}{\mathrm{d} t} = - \left( \mathbf{\lambda}_{ij}^+ - \mathbf{\lambda}_{ij}^- \right) \f$ else.
     *
     * \f$ \mathbf{\lambda}^+ \f$ (resp. \f$ \mathbf{\lambda}^- \f$) is the Lagrange multiplier associated with the constraint \f$ \mathbf{d} + \mathbb{B} \mathbf{u} \ge 0 \f$ (resp. \f$ -\mathbf{d} - \mathbb{B} \mathbf{u} \ge 0 \f$).
     * By convention, \f$ \mathbf{\lambda}^+ \ge 0 \f$ and \f$ \mathbf{\lambda}^- \ge 0 \f$.
     *
     * @tparam dim Dimension (2 or 3).
     */
    template<std::size_t dim>
    class ViscousWithoutFriction: protected ProblemBase
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
        ViscousWithoutFriction(std::size_t nparts, double dt, const ProblemParams<ViscousWithoutFriction<dim>>& problem_params);

        /**
         * @brief Construct the COO storage of the matrix \f$ \mathbb{B} \f$ for the constraint.
         *
         * See \c create_vector_distances for the order of the rows of the matrix.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles (for positions).
         * @param contacts [in] Array of contacts.
         * @param firstCol [in] Index of the first column (solver-dependent).
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
        std::size_t number_row_matrix(const std::vector<neighbor<dim>>& contacts);
        /**
         * @brief Create vector \f$ \mathbf{d} \f$.
         *
         * \f$ \mathbf{d} \f$ contains all the distances associated with the constraint \f$ D > 0 \f$, then the distances associated with the constraint \f$ D < 0 \f$.
         *
         * @tparam dim Dimension (2 or 3).
         * @param contacts [in] Array of contacts.
         */
        void create_vector_distances(const std::vector<neighbor<dim>>& contacts);

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
        void matrix_free_gemv_transpose_A(const neighbor<dim>& c,
                                          const scopi_container<dim>& particles,
                                          const xt::xtensor<double, 1>& L,
                                          xt::xtensor<double, 1>& U,
                                          std::size_t active_offset,
                                          std::size_t row);

        /**
         * @brief Set \f$ \gamma_{ij}^n \f$ from the previous time step and compute the number of contacts with \f$ \gamma_{ij} < 0 \f$.
         *
         * Look if particles \c i and \c j were already in contact.
         *
         * @param contacts_new
         */
        void extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts_new);
        /**
         * @brief Compute the value of \f$ \gamma^{n+1} \f$.
         *
         * \f[
         *      \gamma^{n+1}_{ij} = \gamma^n_{ij} - \Delta t \left( \mathbf{\lambda}_{ij}^+ - \mathbf{\lambda}_{ij}^- \right).
         * \f]
         *
         * @param contacts [in] Array of contacts.
         * @param lambda [in] Lagrange multipliers.
         * @param u_tilde [in] Vector \f$ \mathbf{d} + \mathbb{B} \mathbf{u} - \mathbf{f}(\mathbf{u}) \f$, where \f$ \mathbf{u} \f$ is the solution of the optimization problem.
         */
        template<class ScopiSolver>
        void extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                      ScopiSolver* solver);
        /**
         * @brief Whether the optimization problem should be solved.
         *
         * For compatibility with the other problems.
         */
        bool should_solve_optimization_problem();

    private:
        /**
         * @brief Parameters.
         */
        ProblemParams<ViscousWithoutFriction<dim>> m_params;
    };

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::create_matrix_constraint_coo(const scopi_container<dim>& particles,
                                                                   const std::vector<neighbor<dim>>& contacts)
    {
        std::size_t active_offset = particles.nb_inactive();
        matrix_positive_distance(particles, contacts, 1);
        std::size_t ic = 0;
        std::size_t igamma = 0;
        for (auto &c: contacts)
        {
            if (c.i >= active_offset)
            {
                if (this->m_gamma[ic] < -m_params.tol)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows.push_back(contacts.size() + igamma);
                        this->m_A_cols.push_back((c.i - active_offset)*3 + d);
                        this->m_A_values.push_back(this->m_dt*c.nij[d]);
                    }
                }
            }

            if (c.j >= active_offset)
            {
                if (this->m_gamma[ic] < -m_params.tol)
                {
                    for (std::size_t d = 0; d < 3; ++d)
                    {
                        this->m_A_rows.push_back(contacts.size() + igamma);
                        this->m_A_cols.push_back( (c.j - active_offset)*3 + d);
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
                    if (this->m_gamma[ic] < -m_params.tol)
                    {
                        this->m_A_rows.push_back(contacts.size() + igamma);
                        this->m_A_cols.push_back(3*this->m_nparticles + 3*ind_part + ip);
                        this->m_A_values.push_back(-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                }
            }

            if (c.j >= active_offset)
            {
                std::size_t ind_part = c.j - active_offset;
                auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
                if (this->m_gamma[ic] < -m_params.tol)
                {
                    for (std::size_t ip = 0; ip < 3; ++ip)
                    {
                        this->m_A_rows.push_back(contacts.size() + igamma);
                        this->m_A_cols.push_back( 3*this->m_nparticles + 3*ind_part + ip);
                        this->m_A_values.push_back(this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                    }
                }
            }
            if (this->m_gamma[ic] < -m_params.tol)
            {
                igamma++;
            }
            ++ic;
        }
    }

    template<std::size_t dim>
    ViscousWithoutFriction<dim>::ViscousWithoutFriction(std::size_t nparticles, double dt, const ProblemParams<ViscousWithoutFriction<dim>>& problem_params)
    : ProblemBase(nparticles, dt)
    , ViscousBase<dim>()
    , m_params(problem_params)
    {}

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::extra_steps_before_solve(const std::vector<neighbor<dim>>& contacts_new)
    {
        this->m_should_solve = true;
        this->set_gamma_base(contacts_new);
        this->m_nb_gamma_neg = 0;
        for (auto& g : this->m_gamma)
        {
            if (g < -m_params.tol)
                this->m_nb_gamma_neg++;
        }
    }

    template<std::size_t dim>
    template<class ScopiSolver>
    void ViscousWithoutFriction<dim>::extra_steps_after_solve(const std::vector<neighbor<dim>>& contacts,
                                                               ScopiSolver* solver)
    {   
        this->m_should_solve = false;
        if (contacts.size() == 0)
        {
            return;
        }
        auto lambda = solver->get_lagrange_multiplier(contacts);
        this->m_contacts_old = contacts;
        this->m_gamma_old.resize(this->m_gamma.size());
        std::size_t ind_gamma_neg = 0;
        for (std::size_t i = 0; i < this->m_gamma_old.size(); ++i)
        {   std::cout<<"Gamma "<<i<<" = "<<this->m_gamma[i]<<std::endl;
            double f_contact;
            if (this->m_gamma[i] < -m_params.tol)
            {   
                f_contact = lambda(i) - lambda(this->m_gamma.size() + ind_gamma_neg);
                ind_gamma_neg++;
            }
            else
            {   
                f_contact = lambda(i);
            }
            this->m_gamma_old[i] = std::min(0., this->m_gamma[i] - this->m_dt * f_contact);
            if (this->m_gamma_old[i] > -m_params.tol)
                this->m_gamma_old[i] = 0.;
            // if (this->m_gamma_old[i] > -m_params.tol)
            //     this->m_gamma_old[i] = 0.;
            PLOG_WARNING << this->m_gamma[i];
        }
    }


    template<std::size_t dim>
    std::size_t ViscousWithoutFriction<dim>::number_row_matrix(const std::vector<neighbor<dim>>& contacts)
    {
        return contacts.size() + this->m_nb_gamma_neg;
    }

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::create_vector_distances(const std::vector<neighbor<dim>>& contacts)
    {
        this->m_distances = xt::zeros<double>({contacts.size() + this->m_nb_gamma_neg});
        std::size_t igamma = 0;
        for (std::size_t i = 0; i < contacts.size(); ++i)
        {
            this->m_distances[i] = contacts[i].dij;
            if(this->m_gamma[i] < -m_params.tol)
            {
                this->m_distances[contacts.size() + igamma] = -contacts[i].dij;
                igamma++;
            }
        }
    }

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::matrix_free_gemv_A(const neighbor<dim>& c,
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
                if (this->m_gamma[row] < -m_params.tol)
                {
                    R(this->m_gamma.size() + row) -= (this->m_dt*c.nij[d]) * U((c.i - active_offset)*3 + d);
                }
            }
        }

        if (c.j >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                R(row) -= (this->m_dt*c.nij[d]) * U((c.j - active_offset)*3 + d);
                if (this->m_gamma[row] < -m_params.tol)
                {
                    R(this->m_gamma.size() + row) -= (-this->m_dt*c.nij[d]) * U((c.j - active_offset)*3 + d);
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
                R(row) -= (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * U(3*this->m_nparticles + 3*ind_part + ip);
                if (this->m_gamma[row] < -m_params.tol)
                {
                    R(this->m_gamma.size() + row) -= (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * U(3*this->m_nparticles + 3*ind_part + ip);
                }
            }
        }

        if (c.j >= active_offset)
        {
            std::size_t ind_part = c.j - active_offset;
            auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
                R(row) -= (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * U(3*this->m_nparticles + 3*ind_part + ip);
                if (this->m_gamma[row] < -m_params.tol)
                {
                    R(this->m_gamma.size() + row) -= (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip))) * U(3*this->m_nparticles + 3*ind_part + ip);
                }
            }
        }
    }

    template<std::size_t dim>
    void ViscousWithoutFriction<dim>::matrix_free_gemv_transpose_A(const neighbor<dim>& c,
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
                U((c.i - active_offset)*3 + d) += L(row) * (-this->m_dt*c.nij[d]);
                if (this->m_gamma[row] < -m_params.tol)
                {
#pragma omp atomic
                    U((c.i - active_offset)*3 + d) += L(this->m_gamma.size() + row) * (this->m_dt*c.nij[d]);
                }
            }
        }

        if (c.j >= active_offset)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
#pragma omp atomic
                U((c.j - active_offset)*3 + d) += L(row) * (this->m_dt*c.nij[d]);
                if (this->m_gamma[row] < -m_params.tol)
                {
#pragma omp atomic
                    U((c.j - active_offset)*3 + d) += L(this->m_gamma.size() + row) * (-this->m_dt*c.nij[d]);
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
#pragma omp atomic
                U(3*this->m_nparticles + 3*ind_part + ip) += L(row) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                if (this->m_gamma[row] < -m_params.tol)
                {
#pragma omp atomic
                    U(3*this->m_nparticles + 3*ind_part + ip) += L(this->m_gamma.size() + row) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }
        }

        if (c.j >= active_offset)
        {
            std::size_t ind_part = c.j - active_offset;
            auto dot = xt::eval(xt::linalg::dot(rj_cross, Rj));
            for (std::size_t ip = 0; ip < 3; ++ip)
            {
#pragma omp atomic
                U(3*this->m_nparticles + 3*ind_part + ip) += L(row) * (-this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                if (this->m_gamma[row] < -m_params.tol)
                {
#pragma omp atomic
                    U(3*this->m_nparticles + 3*ind_part + ip) += L(this->m_gamma.size() + row) * (this->m_dt*(c.nij[0]*dot(0, ip)+c.nij[1]*dot(1, ip)+c.nij[2]*dot(2, ip)));
                }
            }
        }
    }

    template<std::size_t dim>
    bool ViscousWithoutFriction<dim>::should_solve_optimization_problem()
    {
        return this->m_should_solve;
    }

    template<std::size_t dim>
    ProblemParams<ViscousWithoutFriction<dim>>::ProblemParams()
    : tol(1e-6)
    {}

    template<std::size_t dim>
    ProblemParams<ViscousWithoutFriction<dim>>::ProblemParams(const ProblemParams<ViscousWithoutFriction<dim>>& params)
    : tol(params.tol)
    {}

}


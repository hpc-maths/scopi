#pragma once

#include <cstddef>
#include <iostream>
#include <filesystem>

#include <CLI/CLI.hpp>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{
    /**
     * @class OptimParams
     * @brief Parameters for the optimization solver.
     *
     * To be specialized.
     *
     * @tparam solver_t Type of the solver.
     */
    template<class optim_t>
    struct OptimParams
    {
    private:
        /**
         * @brief Constructor.
         */
        OptimParams();
    };

    /**
     * @class ProblemParams
     * @brief Parameters for the problem.
     *
     * To be specialized.
     *
     * @tparam problem_t Type of the problem.
     */
    template<class problem_t>
    struct ProblemParams
    {
    private:
        /**
         * @brief Constructor.
         */
        ProblemParams();
    };

    /**
     * @class ContactsParams
     * @brief Parameters for the contacts.
     *
     * To be specialized.
     *
     * @tparam contact_t Type of the contacts.
     */
    template<class contact_t>
    struct ContactsParams
    {
    private:
        /**
         * @brief Constructor.
         */
        ContactsParams();
    };

    /**
     * @class VapParams
     * @brief Parameters for the a priori velocity.
     *
     * To be specialized.
     *
     * @tparam vap_t Type of the a priori velocity.
     */
    template<class vap_t>
    struct VapParams
    {
    private:
        /**
         * @brief Constructor.
         */
        VapParams();
    };

    /**
     * @class ScopiParams
     * @brief Parameters for the main solver.
     */
    struct ScopiParams
    {
        /**
         * @brief Default constructor.
         */
        ScopiParams();

        void init_options(CLI::App& app);

        /**
         * @brief Frequency to write the output files.
         *
         * Write output files with the current iteration is a multiple of \c output_frequency.
         * If \c output_frequency is <tt> std::size_t(-1) </tt>, then no output is written.
         * Default value is 1.
         * \note \c output_frequency > 0
         */
        std::size_t output_frequency;

        std::filesystem::path path;
        /**
         * @brief Path to the output files.
         *
         * Directories have to exist before running the code.
         */
        std::string filename;
        /**
         * @brief Whether to write the velocity and the rotation velocity of the particles in the output files.
         *
         * Default value is false.
         */
        bool write_velocity;
        /**
         * @brief true for binary output files, false for text files.
         *
         * Default value is false.
         */
        bool binary_output;
    };

    /**
     * @class Params
     * @brief Global struct for the parameters.
     *
     * \todo Interface is not used friendly.
     *
     * @tparam solver_t Type of the optimization solver.
     */
    template<class solver_t>
    struct Params
    {
        using solver_params_t = ScopiParams;
        using optim_params_t = typename solver_t::optim_solver_t::params_t;
        using problem_params_t = typename solver_t::optim_solver_t::problem_t::params_t;
        using contact_params_t = typename solver_t::contact_t::params_t;
        using vap_params_t = typename solver_t::vap_t::params_t;

        /**
         * @brief Default constructor.
         */
        Params(solver_params_t& solver_params,
               optim_params_t& optim_params,
               problem_params_t& problem_params,
               contact_params_t& contact_params,
               vap_params_t& vap_params
        );

        /**
         * @brief Parameters for the optimization solver.
         */
        optim_params_t& optim_params;
        /**
         * @brief Parameters for the problem.
         */
        problem_params_t& problem_params;
        /**
         * @brief Parameters for the contacts.
         */
        contact_params_t& contact_params;
        /**
         * @brief Parameters for the a priori velocity.
         */
        vap_params_t& vap_params;
        /**
         * @brief Parameters for the main solver.
         */
        solver_params_t& solver_params;
    };

    template<class solver_t>
    Params<solver_t>::Params(solver_params_t& solver_params,
                             optim_params_t& optim_params,
                             problem_params_t& problem_params,
                             contact_params_t& contact_params,
                             vap_params_t& vap_params
    )
    : optim_params(optim_params)
    , problem_params(problem_params)
    , contact_params(contact_params)
    , vap_params(vap_params)
    , solver_params(solver_params)
    {}
}



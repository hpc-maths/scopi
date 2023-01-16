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
     * @tparam problem_t Type of the problem (redundant with \c solver_t).
     * @tparam contact_t Type of contacts.
     * @tparam vap_t Type of a priori velocity.
     */
    template<class solver_t,
             class contact_t,
             class vap_t>
    struct Params
    {
        /**
         * @brief Alias for problem type.
         */
        using problem_t = typename solver_t::problem_type;

        /**
         * @brief Default constructor.
         */
        Params();
        /**
         * @brief Copy constructor.
         *
         * @param params Parameters to be copied.
         */
        Params(const Params<solver_t, contact_t, vap_t>& params);

        /**
         * @brief Parameters for the optimization solver.
         */
        OptimParams<solver_t> optim_params;
        /**
         * @brief Parameters for the problem.
         */
        ProblemParams<problem_t> problem_params;
        /**
         * @brief Parameters for the contacts.
         */
        ContactsParams<contact_t> contacts_params;
        /**
         * @brief Parameters for the a priori velocity.
         */
        VapParams<vap_t> vap_params;
        /**
         * @brief Parameters for the main solver.
         */
        ScopiParams scopi_params;
    };

    template<class solver_t, class contact_t, class vap_t>
    Params<solver_t, contact_t, vap_t>::Params()
    : optim_params()
    , problem_params()
    , contacts_params()
    , vap_params()
    , scopi_params()
    {}

    template<class solver_t, class contact_t, class vap_t>
    Params<solver_t, contact_t, vap_t>::Params(const Params<solver_t, contact_t, vap_t>& params)
    : optim_params(params.optim_params)
    , problem_params(params.problem_params)
    , contacts_params(params.contacts_params)
    , vap_params(params.vap_params)
    , scopi_params(params.scopi_params)
    {}

}



#pragma once

#include <cstddef>
#include <iostream>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{
    template<class solver_t>
    struct OptimParams
    {
    private:
        OptimParams();
    };

    template<class problem_t>
    struct ProblemParams
    {
    private:
        ProblemParams();
    };

    template<class contact_t>
    struct ContactsParams
    {
    private:
        ContactsParams();
    };

    template<class vap_t>
    struct VapParams
    {
    private:
        VapParams();
    };

    struct ScopiParams
    {
        ScopiParams();
        ScopiParams(const ScopiParams& params);

        std::size_t output_frequency;
        std::string filename;
        /**
         * @brief Whether to write the velocity and the rotation velocity of the particles in the output files.
         *
         * Default value is false.
         */
        bool write_velocity;
    };

    // TODO fix compilation with default template parameters
    // template<class solver_t = OptimUzawaMatrixFreeOmp<DryWithoutFriction>,
    //          class problem_t = DryWithoutFriction,
    //          class contact_t = contact_kdtree,
    //          class vap_t = vap_fixed>
    template<class solver_t,
             class problem_t,
             class contact_t,
             class vap_t>
    struct Params
    {
        Params();
        Params(const Params<solver_t, problem_t, contact_t, vap_t>& params);

        OptimParams<solver_t> optim_params;
        ProblemParams<problem_t> problem_params;
        ContactsParams<contact_t> contacts_params;
        VapParams<vap_t> vap_params;
        ScopiParams scopi_params;
    };

    template<class solver_t, class problem_t, class contact_t, class vap_t> 
    Params<solver_t, problem_t, contact_t, vap_t>::Params()
    : optim_params()
    , problem_params()
    , contacts_params()
    , vap_params()
    , scopi_params()
    {}

    template<class solver_t, class problem_t, class contact_t, class vap_t>
    Params<solver_t, problem_t, contact_t, vap_t>::Params(const Params<solver_t, problem_t, contact_t, vap_t>& params)
    : optim_params(params.optim_params)
    , problem_params(params.problem_params)
    , contacts_params(params.contacts_params)
    , vap_params(params.vap_params)
    , scopi_params(params.scopi_params)
    {}

}



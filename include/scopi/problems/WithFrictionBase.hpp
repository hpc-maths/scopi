#pragma once

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"


namespace scopi
{
    class WithFrictionBase
    {
    public:
        void set_coeff_friction(double mu);

    protected:
        WithFrictionBase();
        double m_mu;
    };


}


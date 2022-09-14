#include "scopi/solvers/projection.hpp"

namespace scopi
{
    auto projection<DryWithoutFriction>::projection_cone(const xt::xtensor<double, 1>& l)
    {
        return xt::maximum( l, 0);
    }

}

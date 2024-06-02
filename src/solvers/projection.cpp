#include "scopi/solvers/projection.hpp"

namespace scopi
{
    xt::xtensor<double, 1> projection<DryWithoutFriction>::projection_cone(const xt::xtensor<double, 1>& l)
    {
        return xt::maximum(l, 0);
    }

}

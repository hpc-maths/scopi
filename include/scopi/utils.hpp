#pragma once

#include <chrono>
#include <vector>

#include <xtensor/xfixed.hpp>
#include <xtensor-blas/xlinalg.hpp>

#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"


/////////////////////////////
// Functions for the timer //
/////////////////////////////

/**
 * @brief Timer used in tic & toc.
 */
static auto tic_timer = std::chrono::high_resolution_clock::now();
/**
 * @brief Launching the timer.
 */
void tic();
/**
 * @brief Stopping the timer.
 *
 * @return Duration in seconds.
 */
double toc();

/**
 * @brief Recursive function to initialize 2D newton (superellipsoid)
 *
 * \todo Write documentation.
 *
 * @param binit
 * @param n
 * @param theta_g
 * @param theta_d
 * @param rx
 * @param ry
 * @param e
 *
 * @return
 */
std::vector<double> create_binit(std::vector<double> binit, int n,
  double theta_g, double theta_d, double rx, double ry, double e);


/**
 * @brief Sign function.
 *
 * @param val Variable to get the sign.
 *
 * @return -1 if \c val < 0, 1 if \c val > 0, and 0 otherwise.
 */
int sign(double val);

template<typename F, typename DF, typename U, typename A>
/**
 * @brief Newton method (with linesearch).
 *
 * \todo Write documentation.
 *
 * @param u0
 * @param f
 * @param grad_f
 * @param args
 * @param itermax
 * @param ftol
 * @param xtol
 *
 * @return
 */
auto newton_method(U u0, F f, DF grad_f, A args, const int itermax, const double ftol, const double xtol){
  // std::cout << "newton_method : u0 = " << u0 << std::endl;
  // std::cout << "newton_method : args = " << args << std::endl;
  int cc = 0;
  auto u = u0;
  while (cc<itermax) {
    auto d = xt::linalg::solve(grad_f(u,args), -f(u,args));
    auto var = xt::linalg::norm(d);
    if (var < xtol) {
      // std::cout << "newton_method : cvgce (xtol) after " << cc << " iterations => RETURN u = " << u << std::endl;
      return std::make_tuple(u,cc);
    }
    // std::cout << "newton_method : iteration " << cc << " => d = " << d << " var = " << var << std::endl;
    // linesearch
    double t = 1;
    auto ferr = xt::linalg::norm(f(u,args));
    // std::cout << "newton_method : iteration " << cc << " => ferr = " << ferr << std::endl;
    if (ferr < ftol) {
      // std::cout << "newton_method : cvgce (ftol) after " << cc << " iterations => RETURN u = " << u << std::endl;
      return std::make_tuple(u,cc);
    }
    while ((xt::linalg::norm(f(u+t*d,args)) > ferr) && (t>0.01)){
        t -= 0.01;
    }
    // std::cout << "newton_method : iteration " << cc << " => t = " << t << std::endl;
    u += t * d;
    // std::cout << "newton_method : iteration " << cc << " => u = " << u << std::endl;
    cc += 1;
  }
  PLOG_ERROR << "newton_method : !!!!!! FAILED !!!!!! after " << cc << " iterations => RETURN u = " << u;

  return std::make_tuple(u,-1);
}

namespace scopi
{
    // namespace detail
    // {
    //     using cross_t = xt::xtensor_fixed<double, xt::xshape<3, 3>>;

    //     template <class E>
    //     cross_t cross_product_impl(std::integral_constant<std::size_t, 2>, const xt::xexpression<E>& e)
    //     {
    //         return {{     0,    0,  e.derived_cast()(1)},
    //                 {     0,    0, -e.derived_cast()(0)},
    //                 { -e.derived_cast()(1), e.derived_cast()(0),     0}};
    //     }

    //     template <class E>
    //     cross_t cross_product_impl(std::integral_constant<std::size_t, 3>, const xt::xexpression<E>& e)
    //     {
    //         return {{     0, -e.derived_cast()(2),  e.derived_cast()(1)},
    //                 {  e.derived_cast()(2),     0, -e.derived_cast()(0)},
    //                 { -e.derived_cast()(1),  e.derived_cast()(0),     0}};
    //     }
    // }
    // template <std::size_t dim, class E>
    // auto cross_product(const xt::xexpression<E>& e)
    // {
    //     return detail::cross_product_impl(std::integral_constant<std::size_t, dim>{}, e);
    // }

    namespace detail
    {
        using cross_t = xt::xtensor_fixed<double, xt::xshape<3, 3>>;

        template <class E>
        cross_t cross_product_impl(std::integral_constant<std::size_t, 2>, const E& e)
        {
            return {{     0,    0,  e(1)},
                    {     0,    0, -e(0)},
                    { -e(1), e(0),     0}};
        }

        template <class E>
        cross_t cross_product_impl(std::integral_constant<std::size_t, 3>, const E& e)
        {
            return {{     0, -e(2),  e(1)},
                    {  e(2),     0, -e(0)},
                    { -e(1),  e(0),     0}};
        }
    }
    /**
     * @brief
     *
     * \todo Write documentation.
     *
     * @tparam dim
     * @tparam E
     * @param e
     *
     * @return
     */
    template <std::size_t dim, class E>
    auto cross_product(const E& e)
    {
        return detail::cross_product_impl(std::integral_constant<std::size_t, dim>{}, e);
    }

    /**
     * @brief Get rotation velocity (3D vector) from a scalar value (2D simulation)
     *
     * @param w Rotation velocity in 2D
     *
     * @return (0, 0, w)
     */
    xt::xtensor_fixed<double, xt::xshape<3>> get_omega(double w);
    /**
     * @brief Get rotation velocity (3D simulation)
     *
     * This function allows to be consistent between 2D and 3D simulations.
     *
     * @param w Rotation velocity in 3D
     *
     * @return  w
     */
    const xt::xtensor_fixed<double, xt::xshape<3>>& get_omega(const xt::xtensor_fixed<double, xt::xshape<3>>& w);


    template<std::size_t dim>
    class BoxDomain;

    template<std::size_t dim>
    class scopi_container;

    template<std::size_t dim>
    void add_objects_from_periodicity(const BoxDomain<dim>& box, scopi_container<dim>& particles, double dmax)
    {
        for (std::size_t d = 0; d < dim; ++d)
        {
            if (box.is_periodic(d))
            {
                for (std::size_t ip = 0; ip < particles.periodic_ptr(); ++ip)
                {
                    auto& pos = particles.pos()[ip];
                    if (pos[d] - dmax < box.lower_bound(d))
                    {
                        auto new_pos = pos;
                        new_pos[d] += box.upper_bound(d) - box.lower_bound(d);
                        particles.push_back(ip, new_pos);
                    }
                }
            }
        }
    }
}

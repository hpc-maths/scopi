#include <cstddef>
#include <memory>
#include <vector>
#include <xtensor/xmath.hpp>
#include <scopi/objects/types/worm.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>
#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/contact/contact_brute_force.hpp>

auto searchsorted(const std::vector<double>& v, const std::vector<double>& t)
{
    std::vector<std::size_t> out(t.size());
    std::for_each(t.begin(), t.cend(), [&, n=0](const float &e) mutable{
            auto lower = std::lower_bound (v.begin(), v.end(), e);
            out[n++] = std::distance(v.begin(), lower);
            });
    return out;
}

int main()
{
    plog::init(plog::error, "two_worms.log");

    constexpr std::size_t dim = 2;
    double dt = .005;
    std::size_t total_it = 1;
    scopi::scopi_container<dim> particles;
    // auto prop = scopi::property<dim>().mass(1.).moment_inertia(0.1);


    /*
    namespace nl = nlohmann;
    std::ifstream file("../data_tracking.json");
    if (file)
    {
        nl::json jf = nl::json::parse(file);
        std::size_t t = 0;
        for (std::size_t track_id = 0; track_id < 1048; ++track_id) // 1048 à changer en fonction de t
        {
            std::size_t index = t*11 + track_id;
            std::cout << jf["0_x"][to_string(index)] << "   " << jf["0_y"][to_string(index)] << std::endl;
        }
    }
    */
    // std::vector<double> xs({2., 3., 7., 14., 24., 32., 38., 42., 45.});
    // std::vector<double> ys({557., 548.,  538., 528., 519., 510., 500., 490., 481.});
    std::vector<double> xs({0.2, 0.3, 0.7, 1.4, 2.4, 3.2, 3.8, 4.2, 4.5});
    std::vector<double> ys({0.557, 0.548, 0.538, 0.528, 0.519, 0.510, 0.500, 0.490, 0.481});
    std::vector<double> u_i(xs.size()) ;
    std::vector<double> y2s(xs.size(), 0.) ;

    for (std::size_t i = 1; i < xs.size() - 1; ++i)
    {
        double xplus = xs[i+1] - xs[i];
        double xminus = xs[i] - xs[i-1];
        double sig = xminus/(xs[i+1] - xs[i-1]);

        u_i[i] = (ys[i+1] - ys[i])/xplus
            - (ys[i] - ys[i-1])/xminus;

        double p_i = sig*y2s[i-1] + 2.;
        y2s[i] = (sig - 1.)/p_i;
        u_i[i] = (6*u_i[i]/(xs[i+1] - xs[i-1]) - sig*u_i[i-1])/p_i;
    }

    for (std::size_t i = xs.size() - 2; i != std::size_t(-1); --i)
    {
        y2s[i] = y2s[i]*y2s[i+1] + u_i[i];
    }

    std::size_t discretization_spline = 10;
    std::vector<double> x(discretization_spline);
    for (std::size_t i = 0; i < x.size(); ++i)
    {
        x[i] = xs[0] + (i+1)*(xs[xs.size()-1] - xs[0])/(x.size()+1);
    }

    auto k = searchsorted(xs, x);

    std::vector<double> x_worm;
    std::vector<double> y_worm;
    double radius = 0.1;
    double dist = 0.;
    double x_prev = xs[0];
    double y_prev = ys[0];
    std::size_t nb_spheres = 0;
    for(std::size_t i = 0; i < x.size(); ++i)
    {
        std::size_t khi = k[i];
        std::size_t klo = khi - 1;
        double step = xs[khi] - xs[klo];
        double x_right = (xs[khi] - x[i])/step;
        double x_left = (x[i] - xs[klo])/step;
        double y = x_right*ys[klo] + x_left*ys[khi]+(
                   x_right*(x_right*x_right - 1)*y2s[klo]+
                   x_left*(x_left*x_left - 1)*y2s[khi])*step*step/6.;
        dist += std::sqrt((x[i]-x_prev)*(x[i]-x_prev) + (y-y_prev)*(y-y_prev));
        if (dist > (nb_spheres+1)*radius)
        {
            x_worm.push_back(x[i]);
            y_worm.push_back(y);
            nb_spheres++;
        }
        x_prev = x[i];
        y_prev = y;
    }

    for (std::size_t i = 0; i < x_worm.size(); ++i)
    {
        std::cout << x_worm[i] << "  " << y_worm[i] << std::endl;
    }

    /*
    double dist2 = (xs[0] - xs[xs.size()-1])*(xs[0] - xs[xs.size()-1]) + (ys[0] - ys[ys.size()-1])*(ys[0] - ys[ys.size()-1]);
    scopi::worm<dim> g({{x[0], y[0]}, {x[1], y[1]}, {x[2], y[2]}, {x[3], y[3]}, {x[4], y[4]}, {x[5], y[5]}},
            {{scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}, {scopi::quaternion(0.)}},
            std::sqrt(dist2)/(x.size()+1));

    particles.push_back(g);
    // particles.push_back(g, prop.desired_velocity({-1., 0.}));

    scopi::OptimParams<scopi::OptimMosek<scopi::DryWithoutFriction>> params;
    params.change_default_tol_mosek = false;
    scopi::ScopiSolver<dim, scopi::OptimMosek<scopi::DryWithoutFriction>, scopi::contact_kdtree> solver(particles, dt, params);
    solver.solve(total_it);
    */

    return 0;
}

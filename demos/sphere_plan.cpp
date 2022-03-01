#include <xtensor/xmath.hpp>
#include <scopi/objects/types/sphere.hpp>
#include <scopi/objects/types/plan.hpp>
#include <scopi/solver.hpp>
#include <scopi/property.hpp>

#include <scopi/solvers/OptimMosek.hpp>
#include <scopi/vap/vap_fpd.hpp>

int main()
{
    plog::init(plog::warning, "sphere_plan.log");

    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    double radius = 1.;
    double alpha = PI/3.;
    double g = 1.;

    std::vector<double> dt({0.1, 0.05, 0.01, 0.005, 0.001, 0.0005});
    std::vector<std::size_t> total_it({100, 200, 1000, 2000, 10000, 20000});

    for (std::size_t i = 0; i < dt.size(); ++i)
    {
        scopi::scopi_container<dim> particles;
        scopi::plan<dim> p({{-radius*std::cos(PI/2.-alpha), -radius*std::sin(PI/2.-alpha)}}, PI/2.-alpha);
        scopi::sphere<dim> s({{0., 0.}}, radius);
        particles.push_back(p, scopi::property<dim>().deactivate());
        particles.push_back(s, scopi::property<dim>().force({{0., -g}}));

        scopi::ScopiSolver<dim, scopi::OptimMosek, scopi::contact_kdtree, scopi::vap_fpd> solver(particles, dt[i]);
        solver.solve(total_it[i]);

        auto pos = particles.pos();
        double tf = dt[i]*(total_it[i]+1);
        auto analytical_sol = g/2.*std::sin(alpha)*tf*tf * xt::xtensor<double, 1>({std::cos(alpha), -std::sin(alpha)});
        PLOG_DEBUG << "pos = " << pos(1);
        PLOG_DEBUG << "sol = " << analytical_sol;
        double error = xt::linalg::norm(pos(1) - analytical_sol, 2) / xt::linalg::norm(analytical_sol);
        PLOG_WARNING << dt[i] << '\t' << error;
    }

    return 0;
}

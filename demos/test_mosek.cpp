#include <iostream>
#include <xtensor/xmath.hpp>
#include <scopi/object/plan.hpp>
#include <scopi/object/neighbor.hpp>

#include <scopi/functors.hpp>
#include <scopi/types.hpp>
#include <scopi/container.hpp>
#include <fusion.h>

using namespace mosek::fusion;
using namespace monty;

int main()
{
    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;

    scopi::sphere<dim> s1({{0.2, 0.}}, 0.1);
    scopi::sphere<dim> s2({{-0.2, 0.}}, 0.1);
    scopi::sphere<dim> s3({{0.5, 0.}}, 0.1);

    scopi::scopi_container<dim> particles;

    double dt = .1;
    particles.push_back(s1, {{0, 0}}, {{0, 0}}, {{0, 0}});
    particles.push_back(s2, {{0, 0}}, {{0.25, 0}}, {{0, 0}});
    particles.push_back(s3, {{0, 0}}, {{-0.15, 0}}, {{0, 0}});

    for (std::size_t nite=0; nite<10; ++nite)
    {
        std::cout << "Time iteration -> " << nite << std::endl;
        std::vector<scopi::neighbor<dim>> contacts;
        double dmax = 1;

        for(std::size_t i = 0; i < particles.size() - 1; ++i)
        {
            for(std::size_t j = i + 1; j < particles.size(); ++j)
            {
                auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                if (neigh.dij < dmax)
                {
                    contacts.emplace_back(std::move(neigh));
                    contacts.back().i = i;
                    contacts.back().j = j;
                }
            }
        }

        std::size_t N = particles.size();

        xt::xtensor<double, 1> MU = xt::zeros<double>({N*dim});
        xt::xtensor<double, 2> sqrtM = xt::eye<double>(N*dim);
        double mass = 1.;
        for (std::size_t i=0; i<N; ++i)
        {
            for (std::size_t d=0; d<dim; ++d)
            {
                MU(i*dim + d) = mass*particles.vd()[i][d]; // TODO: add mass into particles
            }
        }

        xt::xtensor<double, 1> distances = xt::zeros<double>({contacts.size()});
        for(std::size_t i=0; i<contacts.size(); ++i)
        {
            distances[i] = contacts[i].dij;
        }
        std::cout << "distances " << distances << std::endl;

        xt::xtensor<double, 2> A = xt::zeros<double>({dim*contacts.size(), N*dim});
        xt::xtensor<double, 2> B = xt::zeros<double>({contacts.size(), dim*contacts.size()});

        std::size_t ic = 0;
        for (auto &c: contacts)
        {
            for (std::size_t d=0; d<dim; ++d)
            {
                A(ic*dim + d, c.i*dim + d) = 1;
                A(ic*dim + d, c.j*dim + d) = -1;
                B(ic, ic*dim + d) = c.nij[d];
            }
            ++ic;
        }

        auto dtBA = xt::eval(dt*xt::linalg::dot(B, A));
        // std::cout << A << std::endl;
        // std::cout << B << std::endl;

        Model::t model = new Model("contact"); auto _M = finally([&]() { model->dispose(); });
        Variable::t u = model->variable("u", dim*N);
        Variable::t s = model->variable("s", 1);

        auto MU_mosek = std::make_shared<ndarray<double, 1>>(MU.data(), shape_t<1>({MU.shape(0)}));
        model->objective("minvar", ObjectiveSense::Minimize, Expr::sub(s, Expr::dot(MU_mosek, u)));

        auto sqrtM_mosek = std::make_shared<ndarray<double, 2>>(sqrtM.data(), shape_t<2>({sqrtM.shape(0), sqrtM.shape(1)}));

        auto D_mosek = std::make_shared<ndarray<double, 1>>(distances.data(), shape_t<1>({distances.shape(0)}));
        auto dtBA_mosek = std::make_shared<ndarray<double, 2>>(dtBA.data(), shape_t<2>({dtBA.shape(0), dtBA.shape(1)}));

        Constraint::t qc1 = model->constraint("qc1", Expr::add(D_mosek, Expr::mul(dtBA_mosek, u)), Domain::greaterThan(0.));
        Constraint::t qc2 = model->constraint("qc2", Expr::vstack(1, s, Expr::mul(sqrtM_mosek, u)), Domain::inRotatedQCone());

        model->solve();

        ndarray<double, 1> ulvl   = *(u->level());

        using position_type = typename decltype(particles)::position_type;
        auto uadapt = xt::adapt(reinterpret_cast<double*>(ulvl.raw()), {particles.size(), dim});
        std::cout << "uadapt = " << uadapt << std::endl;
        std::cout << "pos = " << particles.pos() << std::endl << std::endl;
        for (std::size_t i=0; i<particles.size(); ++i)
        {
            for (std::size_t d=0; d<dim; ++d)
            {
                particles.pos()(i)(d) += dt*uadapt(i, d);
            }
            // xt::view(particles.pos(), i) += dt*xt::view(uadapt, i);
        }
        // particles.pos() = particles.pos() + dt*uadapt;
        std::cout << "u = " << ulvl << std::endl;
        std::cout << "pos = " << particles.pos() << std::endl << std::endl;
    }
    // for(auto &c: contacts)
    // {
    //     std::cout << c << std::endl;
    // }

    return 0;
}
#include <iostream>
#include <fstream>
#include <strstream>

#include <xtensor/xmath.hpp>
#include <scopi/object/plan.hpp>
#include <scopi/object/neighbor.hpp>

#include <scopi/functors.hpp>
#include <scopi/types.hpp>
#include <scopi/container.hpp>
#include <fusion.h>
#include "nlohmann/json.hpp"

using namespace mosek::fusion;
using namespace monty;
namespace nl = nlohmann;

int main()
{
    constexpr std::size_t dim = 2;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .01;
    scopi::scopi_container<dim> particles;

    // by default the angle of the objects is 0
    // scopi::superellipsoid<dim> s1({{0.2, 0.}}, {{.1, .05}}, {{1}});
    // scopi::superellipsoid<dim> s2({{-0.2, 0.}}, {{.1, .05}}, {{1}});
    // scopi::superellipsoid<dim> s3({{0.5, 0.}}, {{.1, .05}}, {{1}});

    // particles.push_back(s1, {{0, 0}}, {{0, 0}}, 0, 0, {{0, 0}});
    // particles.push_back(s2, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
    // particles.push_back(s3, {{0, 0}}, {{-0.15, 0}}, 0, 0, {{0, 0}});

    scopi::superellipsoid<dim> s1({{-0.2, 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05}}, {{1}});
    scopi::superellipsoid<dim> s2({{0.2, 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05}}, {{1}});

    std::cout << "s1\n";
    s1.print();
    std::cout << s1.q() << "\n";
    std::cout << "s2\n";
    s2.print();
    std::cout << s2.q() << "\n";

    particles.push_back(s1, {{0, 0}}, {{0.25, 0}}, 0, 0, {{0, 0}});
    particles.push_back(s2, {{0, 0}}, {{-0.25, 0}}, 0, 0, {{0, 0}});

    std::size_t N = particles.size();
    xt::xtensor<double, 1> theta = xt::zeros<double>({N});
    theta(0) = PI/4;
    theta(1) = -PI/4;

    for (std::size_t nite=0; nite<300; ++nite)
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


        xt::xtensor<double, 1> MU = xt::zeros<double>({N*dim});
        xt::xtensor<double, 1> JW = xt::zeros<double>({(dim ==3)?N*dim:N});
        xt::xtensor<double, 2> sqrtM = xt::eye<double>(N*dim);
        xt::xtensor<double, 2> sqrtJ = xt::eye<double>((dim ==3)?N*dim:N);

        double mass = 1.;
        double moment = 1.;
        for (std::size_t i=0; i<N; ++i)
        {
            JW(i) = moment*particles.desired_omega()(i);
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

        xt::xtensor<double, 2> Au = xt::zeros<double>({dim*contacts.size(), N*dim});
        xt::xtensor<double, 2> Aw = xt::zeros<double>({dim*contacts.size(), {(dim ==3)?N*dim:N}});
        xt::xtensor<double, 2> B = xt::zeros<double>({contacts.size(), dim*contacts.size()});

        std::size_t ic = 0;
        for (auto &c: contacts)
        {
            auto r_i = c.pi - particles.pos()(c.i);
            auto r_j = c.pj - particles.pos()(c.j);
            for (std::size_t d=0; d<dim; ++d)
            {
                Au(ic*dim + d, c.i*dim + d) = 1;
                Au(ic*dim + d, c.j*dim + d) = -1;

                Aw(ic*dim + d, c.i) = std::pow(-1, d+1) * r_i(dim - d - 1);
                Aw(ic*dim + d, c.j) = std::pow(-1, d) * r_j(dim - d - 1);

                B(ic, ic*dim + d) = c.nij[d];
            }
            ++ic;
        }

        auto dtBAu = xt::eval(dt*xt::linalg::dot(B, Au));
        auto dtBAw = xt::eval(dt*xt::linalg::dot(B, Aw));
        // std::cout << A << std::endl;
        // std::cout << B << std::endl;

        Model::t model = new Model("contact"); auto _M = finally([&]() { model->dispose(); });
        Variable::t u = model->variable("u", dim*N);
        Variable::t w = model->variable("w", (dim ==3)?N*dim:N);
        Variable::t su = model->variable("su", 1);
        Variable::t sw = model->variable("sw", 1);

        auto MU_mosek = std::make_shared<ndarray<double, 1>>(MU.data(), shape_t<1>({MU.shape(0)}));
        auto JW_mosek = std::make_shared<ndarray<double, 1>>(JW.data(), shape_t<1>({JW.shape(0)}));
        model->objective("minvar", ObjectiveSense::Minimize, Expr::add(Expr::sub(su, Expr::dot(MU_mosek, u)), Expr::sub(sw, Expr::dot(JW_mosek, w))));

        auto sqrtM_mosek = std::make_shared<ndarray<double, 2>>(sqrtM.data(), shape_t<2>({sqrtM.shape(0), sqrtM.shape(1)}));
        auto sqrtJ_mosek = std::make_shared<ndarray<double, 2>>(sqrtJ.data(), shape_t<2>({sqrtJ.shape(0), sqrtJ.shape(1)}));

        auto D_mosek = std::make_shared<ndarray<double, 1>>(distances.data(), shape_t<1>({distances.shape(0)}));
        auto dtBAu_mosek = std::make_shared<ndarray<double, 2>>(dtBAu.data(), shape_t<2>({dtBAu.shape(0), dtBAu.shape(1)}));
        auto dtBAw_mosek = std::make_shared<ndarray<double, 2>>(dtBAw.data(), shape_t<2>({dtBAw.shape(0), dtBAw.shape(1)}));

        Constraint::t qc1 = model->constraint("qc1", Expr::add(Expr::add(D_mosek, Expr::mul(dtBAu_mosek, u)), Expr::mul(dtBAw_mosek, w)), Domain::greaterThan(0.));
        Constraint::t qc2 = model->constraint("qc2", Expr::vstack(1, su, Expr::mul(sqrtM_mosek, u)), Domain::inRotatedQCone());
        Constraint::t qc3 = model->constraint("qc3", Expr::vstack(1, sw, Expr::mul(sqrtJ_mosek, w)), Domain::inRotatedQCone());

        model->solve();

        ndarray<double, 1> ulvl   = *(u->level());
        ndarray<double, 1> wlvl   = *(w->level());

        using position_type = typename decltype(particles)::position_type;
        auto uadapt = xt::adapt(reinterpret_cast<double*>(ulvl.raw()), {particles.size(), dim});
        auto wadapt = xt::adapt(reinterpret_cast<double*>(wlvl.raw()), {particles.size()});
        std::cout << "uadapt = " << uadapt << std::endl;
        std::cout << "pos = " << particles.pos() << std::endl << std::endl;
        theta += dt*wadapt;
        for (std::size_t i=0; i<particles.size(); ++i)
        {
            for (std::size_t d=0; d<dim; ++d)
            {
                particles.pos()(i)(d) += dt*uadapt(i, d);
            }
            // xt::view(particles.pos(), i) += dt*xt::view(uadapt, i);
            particles.q()(i) = scopi::quaternion(theta(i));
        }


        // particles.pos() = particles.pos() + dt*uadapt;
        std::cout << "u = " << ulvl << std::endl;
        std::cout << "pos = " << particles.pos() << std::endl << std::endl;
        std::cout << "w = " << wlvl << std::endl;
        std::cout << "theta = " << theta << std::endl << std::endl;

        std::fstream my_file;
        std::ostrstream os;

        nl::json json_output;

        os << "scopi_objects_" << nite << ".json";
        std::ofstream file(os.str());

        json_output["objects"] = {};

        for(std::size_t i = 0; i < particles.size(); ++i)
        {
            json_output["objects"].push_back(scopi::write_objects_dispatcher<dim>::dispatch(*particles[i]));
        }

        json_output["contacts"] = {};

        for(std::size_t i=0; i<contacts.size(); ++i)
        {
            nl::json contact;

            contact["pi"] = contacts[i].pi;
            contact["pj"] = contacts[i].pj;
            contact["nij"] = contacts[i].nij;

            json_output["contacts"].push_back(contact);

        }


        file << json_output;
        // my_file.close();
    }
    // for(auto &c: contacts)
    // {
    //     std::cout << c << std::endl;
    // }

    return 0;
}
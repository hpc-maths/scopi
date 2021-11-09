#include <xtensor/xmath.hpp>
#include <scopi/objects/types/superellipsoid.hpp>
#include <scopi/solvers/mosek.hpp>

int main()
{
    constexpr std::size_t dim = 3;
    double PI = xt::numeric_constants<double>::PI;
    double dt = .01;
    std::size_t total_it = 100;
    scopi::scopi_container<dim> particles;

    // ellipsoids

    // // scopi::superellipsoid<dim> s0({{0., 0., 0.}}, {scopi::quaternion(0)}, {{.05, .05, .05}}, {{1, 1}});
    // // scopi::superellipsoid<dim> s0({{0., 0., 0.}}, {scopi::quaternion(PI/4)}, {{.01, .01, .01}}, {{0.2, 0.2}});
    // scopi::superellipsoid<dim> s1({{-0.1, 0., 0.}}, {scopi::quaternion(PI/3)}, {{.1, .05, .05}}, {{1, 0.6}});
    // scopi::superellipsoid<dim> s2({{0.1, 0., 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05, .05}}, {{1, 1}});
    // // particles.push_back(s0, {{0, 0, 0}}, {{0, 0, 0}}, 0, 0, {{0, 0, 0}});
    // particles.push_back(s1, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    // particles.push_back(s2, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    // std::size_t active_ptr = 0;
    //
    // mosek_solver(particles, dt, total_it, active_ptr);

    // scopi::superellipsoid<dim> s00({{ 0., 0., 0.}}, {scopi::quaternion(0)}, {{0.75, .75, .75}}, {{1, 1}});
    // scopi::superellipsoid<dim> s01({{ 0.0, -0.25, 0.}}, {scopi::quaternion(0)}, {{.02, .02, .02}}, {{1, 1}});
    // scopi::superellipsoid<dim> s02({{ 0.0,  0.0, 0.}}, {scopi::quaternion(0)}, {{.02, .02, .02}}, {{1, 1}});
    // scopi::superellipsoid<dim> s03({{ 0.0,  0.25, 0.}}, {scopi::quaternion(0)}, {{.02, .02, .02}}, {{1, 1}});
    // scopi::superellipsoid<dim> s04({{ 0.0,  0.50, 0.}}, {scopi::quaternion(0)}, {{.02, .02, .02}}, {{1, 1}});
    // particles.push_back(s00, {{0, 0, 0}}, {{0, 0, 0}}, 0, 0, {{0, 0, 0}});
    // particles.push_back(s01, {{0, 0, 0}}, {{0, 0, 0}}, 0, 0, {{0, 0, 0}});
    // particles.push_back(s02, {{0, 0, 0}}, {{0, 0, 0}}, 0, 0, {{0, 0, 0}});
    // particles.push_back(s03, {{0, 0, 0}}, {{0, 0, 0}}, 0, 0, {{0, 0, 0}});
    // particles.push_back(s04, {{0, 0, 0}}, {{0, 0, 0}}, 0, 0, {{0, 0, 0}});

    // scopi::superellipsoid<dim> s0({{-2.0178611416853434,0.345431980626663,0.8254290572954606}}, {scopi::quaternion(2.3617075505591725)}, {{0.49862116112894966,0.9886277466670015,0.42932581145473764}}, {{0.6848644987022457,0.6735067378217184}});
    // particles.push_back(s0, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    // scopi::superellipsoid<dim> s1({{-4.610650267055767,0.2464136921239782,-0.8245997253044465}}, {scopi::quaternion(0.23323611635693134)}, {{0.8778068602330731,0.9107193497809065,0.49343608136653444}}, {{0.8700221189724548,0.5240126660849849}});
    // particles.push_back(s1, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    // scopi::superellipsoid<dim> s2({{-3.0439381312067204,-0.16314476587578142,-0.169468049328197}}, {scopi::quaternion(1.3777606283309283)}, {{0.5165086518315685,0.6846904734780237,0.5509020336825824}}, {{0.6818219062587468,0.6731346211722796}});
    // particles.push_back(s2, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    // scopi::superellipsoid<dim> s3({{-4.577429876491228,-0.9478572511383545,0.9029562468292833}}, {scopi::quaternion(0.35152620456683115)}, {{0.7704571148118766,0.7982319473499402,0.8601164291012255}}, {{0.5821961544852035,0.6120358728835962}});
    // particles.push_back(s3, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    // // ---------------------------
    // scopi::superellipsoid<dim> s4({{2.0222480222223234,-0.041793271095827755,-0.7276386009555003}}, {scopi::quaternion(2.7430563971948354)}, {{0.6754276007274655,0.6618337932952352,0.505585551872838}}, {{0.9269209627707817,0.8493336566670571}});
    // particles.push_back(s4, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    // scopi::superellipsoid<dim> s5({{2.152031828407422,1.6039588429436198,1.5727646235348214}}, {scopi::quaternion(2.106656705063718)}, {{0.3794770582821218,0.9173394031927686,0.7617856423851983}}, {{0.5461455659255023,0.6240354181464485}});
    // particles.push_back(s5, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    // scopi::superellipsoid<dim> s6({{2.8489802128687088,-0.8684509558410438,-1.5346080595281255}}, {scopi::quaternion(2.2533546725528524)}, {{0.23539116519148706,0.8453657621843835,0.6363626424520752}}, {{0.7911493374463678,0.7408296364909785}});
    // particles.push_back(s6, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    // scopi::superellipsoid<dim> s7({{3.9442734087940363,1.532705314178327,-0.10156898872685982}}, {scopi::quaternion(0.4388920335105593)}, {{0.23770949093743132,0.5981450665484906,0.7195720769376363}}, {{0.9046358455029173,0.5857611629376464}});
    // particles.push_back(s7, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});    //


    scopi::superellipsoid<dim> s0({{0., 0., 0.}}, {scopi::quaternion(-PI/4)}, {{.02, .02, .02}}, {{1.0, 1.0}});
    // scopi::superellipsoid<dim> s1({{-0.2, 0., 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05, .05}}, {{1, 1}});
    // scopi::superellipsoid<dim> s2({{0.2, 0., 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05, .05}}, {{1, 1}});
    scopi::superellipsoid<dim> s1({{-0.2, 0., 0.}}, {scopi::quaternion(PI/4)}, {{.1, .05, .05}}, {{1., 1.}});
    scopi::superellipsoid<dim> s2({{0.2, 0., 0.}}, {scopi::quaternion(-PI/4)}, {{.1, .05, .05}}, {{1., 1.}});
    particles.push_back(s0, {{0, 0, 0}}, {{0., 0, 0}}, 0, 0, {{0, 0, 0}});
    particles.push_back(s1, {{0, 0, 0}}, {{0.25, 0, 0}}, 0, 0, {{0, 0, 0}});
    particles.push_back(s2, {{0, 0, 0}}, {{-0.25, 0, 0}}, 0, 0, {{0, 0, 0}});

    std::size_t active_ptr = 1;
    // std::size_t active_ptr = 0; // pas d'obstacles

    scopi::ScopiSolver<dim, scopi::ScsSolver<dim>> solver(particles, dt, active_ptr);
    solver.solve(total_it);

    return 0;
}

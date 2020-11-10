#include <iostream>
#include <xtensor/xio.hpp>
#include <xtensor-io/xhighfive.hpp>

#include <random>

#include <scopi/container.hpp>
#include <scopi/object/sphere.hpp>
#include <scopi/object/globule.hpp>
#include <scopi/contacts.hpp>
#include <scopi/projection.hpp>
#include <nanoflann/nanoflann.hpp>


int main()
{
    // constexpr std::size_t dim = 2;
    // constexpr std::size_t size = 1000000;
    // scopi::sphere<dim> s1({1, 2}, 0.5);
    // scopi::sphere<dim> s2({5, 2}, 0.4);
    // scopi::globule<dim> g1({0, 3}, 0.1), g2({1, 3}, 0.2);
    // scopi::scopi_container<dim> particles;
    //
    // // particles.reserve(size);
    // // for(std::size_t i = 0; i < size; ++i)
    // // {
    // //     particles.push_back(s1);
    // // }
    //
    // particles.push_back(g1);
    // particles.push_back(g2);
    // particles.push_back(s1);
    // particles.push_back(s2);
    //
    // // particles.pos() *= 2;
    // std::cout << "particles.pos() = \n" << particles.pos() << "\n\n";
    //
    // for(std::size_t i = 0; i < particles.size(); ++i)
    // {
    //     std::cout << " i = " << i <<" particles.size() = " << particles.size() << "\n";
    //     particles[i]->print();
    //     particles[i]->pos() += 2;
    //     std::cout << particles[i]->pos() << "\n\n";
    // }
    //
    // std::cout << "particles.pos() = \n" << particles.pos() << "\n\n";

    std::default_random_engine generator;
    std::uniform_real_distribution<double> dist_pos(-10.0,10.0);
    std::uniform_real_distribution<double> dist_v(-1.0,1.0);
    std::uniform_real_distribution<double> dist_vd(-1.0,1.0);
    std::uniform_real_distribution<double> dist_f(-1.0,1.0);
    std::uniform_real_distribution<double> dist_r(0.0,2.0);

    constexpr std::size_t dim = 3;
    scopi::scopi_container<dim> particles;

    // auto tic_timer = std::chrono::high_resolution_clock::now();
    tic();

    constexpr std::size_t size = 5000;
    particles.reserve(size);

    // scopi::sphere<dim> s1({1, 2}, {4, 5}, {7, 8}, {10, 11}, 0.4);
    scopi::sphere<dim> s1({1, 2, 3}, {4, 5, 6}, {7, 8, 9}, {10, 11, 12}, 0.4);
    s1.print();
    std::cout << "s1.pos() : " << s1.pos() << std::endl;
    std::cout << "s1.v() : " << s1.v() << std::endl;
    std::cout << "s1.vd() : " << s1.vd() << std::endl;
    std::cout << "s1.f() : " << s1.f() << std::endl;
    std::cout << "s1 type name : " << typeid(s1).name() << std::endl;
    std::cout << "s1.radius() : " << s1.radius() << std::endl;

    particles.push_back(s1);

    std::cout << "particles[0]->pos() : " << particles[0]->pos() << std::endl;
    std::cout << "particles[0]->v() : " << particles[0]->v() << std::endl;
    std::cout << "particles[0]->vd() : " << particles[0]->vd() << std::endl;
    std::cout << "particles[0]->f() : " << particles[0]->f() << std::endl;
    std::cout << "particles[0] type name : " << typeid(particles[0]).name() << std::endl;
    // std::cout << particles[0]->radius() << std::endl;
    // for(std::size_t i = 0; i < size; ++i){
    //   // std::cout << "xyz = " << dist_pos(generator) << " "<< dist_pos(generator) << " "<< dist_pos(generator) << " r = "<< dist_r(generator) << std::endl;
    //   scopi::sphere<dim> s(
    //     {
    //       dist_pos(generator),
    //       dist_pos(generator),
    //       dist_pos(generator)
    //     },
    //     {
    //       dist_v(generator),
    //       dist_v(generator),
    //       dist_v(generator)
    //     },
    //     {
    //       dist_vd(generator),
    //       dist_vd(generator),
    //       dist_vd(generator)
    //     },
    //     {
    //       dist_f(generator),
    //       dist_f(generator),
    //       dist_f(generator)
    //     },
    //     dist_r(generator)
    //   );
    //   particles.push_back(s);
    // }
    //
    // // auto toc_timer = std::chrono::high_resolution_clock::now();
    // // std::chrono::duration<double> time_span = toc_timer - tic_timer;
    // // double duration = time_span.count();
    // auto duration = toc();
    // std::cout << "\n-- C++ -- CPUTIME (build particles) = " << duration << std::endl;
    //
    // // for(std::size_t i = 0; i < particles.size(); ++i){
    // //   std::cout << i << " pos = "<< particles[i]->pos() << std::endl;
    // //   std::cout << i << " v = "<< particles[i]->v() << std::endl;
    // //   std::cout << i << " vd = "<< particles[i]->vd() << std::endl;
    // //   std::cout << i << " f = "<< particles[i]->f() << std::endl;
    // // }
    // // std::cout << "particles.pos() = \n" << particles.pos() << "\n\n";
    // // std::cout << "particles.v() = \n" << particles.v() << "\n\n";
    // // std::cout << "particles.vd() = \n" << particles.vd() << "\n\n";
    // // std::cout << "particles.f() = \n" << particles.f() << "\n\n";
    //
    //
    // // Contacts
    // double dxc = 0.2;
    // scopi::Contacts<dim> contacts(dxc);
    // contacts.compute_contacts(particles);
    // // contacts.print();
    //
    //
    // // Projection
    // std::size_t maxiter = 40000;
    // double dmin = 0.1;
    // double dt = 0.1;
    // double rho = 0.2;
    // double tol = 1.0e-2;
    // scopi::Projection<dim> proj(maxiter, rho, dmin, tol, dt);
    // proj.run(particles,contacts);
    //
    // // auto test = xt::ones<double>({10, });
    // // xt::dump_hdf5("pos.h5", "./", test);

}

#include <iostream>                        // Standard library import for std::accumulate
#include <numeric>
#include <typeinfo>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>           // Pybind11 import to define Python bindings
#include <xtensor/xmath.hpp>             // xtensor import for the C++ universal functions
#include <xtensor/xarray.hpp>              // xtensor import for the C++ universal functions
#define FORCE_IMPORT_ARRAY                // numpy C api loading
#include <xtensor-python/pyarray.hpp>     // Numpy bindings
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xtensor.hpp>

#include "scopi/particles.hpp"
#include "scopi/contacts.hpp"
#include "scopi/projection.hpp"

// #include "scopi/soa_xtensor.hpp"
// #include <typeinfo>

// struct particle
// {
//     double x, y, z;
//     double radius;
// };
//
// SOA_DEFINE_TYPE(particle, x, y, z, radius);
//
// class People {
//   People()
//
//
//   private:
//     soa::vector<particle> _particles;
// };
//
//
//
// soa::vector<particle> particles;
//
//
// auto add_particle(){    //xt::pyarray<double>& img){
//
//   size_t pos = particles.size();
//   std::cout<<"pos = "<<particles.size()<<std::endl;
//   particles.resize(pos+1);
//   particles.x[pos-1] = 12.2;
//   particles.y[pos-1] = 4.2;
//   particles.z[pos-1] = 3.2;
//   particles.radius[pos-1] = 2.2;
//     auto print_particles = [](const auto &particle) {  std::cout << " particules : " << particle.x << " " << particle.y  << " " << particle.z  << " " <<  particle.radius << "\n"; };
//   std::for_each(particles.begin(), particles.end(), print_particles);
//   std::cout << "\n";
//
//   // constexpr std::size_t npart = 2;
//   // constexpr std::size_t dim = 3;
//   // particles.resize(npart);
//   // particles.x = xt::random::rand<double>({npart}) * 0.1;
//   // particles.y = xt::random::rand<double>({npart}) * 0.1;
//   // particles.z = xt::random::rand<double>({npart}) * 0.1;
//   // particles.radius = xt::random::rand<double>({npart}) * 0.1;
//   // //std::cout << typeid(particles.x).name() << '\n';
//   // //xt::pyarray<double> a({{1,2,3}, {4,5,6}});
//   // //xt :: xarray < int > a = { 1 , 2 };
//   // xt::xtensor<double, 2> a = {{1., 2.}, {3., 4.}};
//   // return particles.x;
// }
// // int test()
// // {
// //
// //     soa::vector<particle> v;
// //     v.resize(20);
// //
// //     v.x[0] = 1.2;
// //
// //     std::cout << v.x[0] << "\n";
// //     std::cout << v[0].x << "\n";
// //     std::cout << v.size() << "\n";
// //
// //     auto print = [](const auto &particle) { std::cout << particle.x << "\n"; };
// //     auto print_ = [](const auto &x) { std::cout << x << "\n"; };
// //     std::for_each(v.rbegin(), v.rend(), print);
// //     std::cout << "\n";
// //
// //     std::for_each(v.x.rbegin(), v.x.rend(), print_);
// //     std::cout << "\n";
// //
// //     soa::vector<particle> v1;
// //     v1 = std::move(v);
// //     std::for_each(v1.begin(), v1.end(), print);
// //     return 0;
// // }
//
PYBIND11_MODULE(scopi, m)
{
  xt::import_numpy();
  m.doc() = "Module to manage particles";

  pybind11::class_<Particles>(m, "Particles")
          .def(pybind11::init<const std::string &>())
          .def("print", &Particles::print)
          .def("add_particles_in_ball", &Particles::add_particles_in_ball)
          .def("add_particles_in_box", &Particles::add_particles_in_box)
          //.def("data", &Particles::data)
          .def("get_data", &Particles::get_data)
          .def("get_x", &Particles::get_x)
          .def("get_y", &Particles::get_y)
          .def("get_z", &Particles::get_z)
          .def("get_r", &Particles::get_r)
          ;

  pybind11::class_<Contacts>(m, "Contacts")
          .def(pybind11::init<const double>())
          .def("print", &Contacts::print)
          .def("get_data", &Contacts::get_data)
          .def("compute_contacts", &Contacts::compute_contacts);

  pybind11::class_<Projection>(m, "Projection")
          .def(pybind11::init<
            const std::size_t,
            const double,
            const double,
            const double,
            const double>())
          .def("print", &Projection::print)
          .def("run", &Projection::run);

  // m.def("add_particle",
  //  add_particle,
  //  "add_particle blabla");
}

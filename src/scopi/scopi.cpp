#include <iostream>                        // Standard library import for std::accumulate
#include <numeric>
#include <typeinfo>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>           // Pybind11 import to define Python bindings
#define FORCE_IMPORT_ARRAY                // numpy C api loading

#include "scopi/particles.hpp"
#include "scopi/contacts.hpp"
#include "scopi/projection.hpp"

PYBIND11_MODULE(scopi, m)
{
  xt::import_numpy();
  m.doc() = "Module to manage particles";

  pybind11::class_<Particles>(m, "Particles")
          .def(pybind11::init<const std::string &>())
          .def("print", &Particles::print)
          .def("add_particles_in_ball", &Particles::add_particles_in_ball)
          .def("add_particles_in_box", &Particles::add_particles_in_box)
          .def("set_vap", &Particles::set_vap)
          //.def("data", &Particles::data)
          .def("get_data", &Particles::get_data)
          .def("get_positions", &Particles::get_positions)
          .def("get_vap", &Particles::get_vap)
          .def("get_v", &Particles::get_v)
          .def("get_x", &Particles::get_x)
          .def("get_y", &Particles::get_y)
          .def("get_z", &Particles::get_z)
          .def("get_r", &Particles::get_r)
          .def("move", &Particles::move)
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
}

#include <iostream>
#include <fstream>
#include <typeinfo>

#include <xtensor/xfixed.hpp>
#include <xtensor/xio.hpp>

#include <scopi/container.hpp>
#include <scopi/object/sphere.hpp>
#include <scopi/object/superellipsoid.hpp>
#include <scopi/object/plan.hpp>
#include <scopi/object/neighbor.hpp>

#include <scopi/functors.hpp>
#include <scopi/types.hpp>

#include "fusion.h"

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/embed.h>

using namespace mosek::fusion;
using namespace monty;

int main()
{
    constexpr std::size_t dim = 2;

    double theta = 4*std::atan(1.)/4;

    scopi::type::rotation<dim> rotation{{{std::cos(theta), -std::sin(theta)},
                                         {std::sin(theta), std::cos(theta)}} };

    scopi::type::position<dim> translation = {3, 1};

    auto s1_pos = xt::eval(translation + 0.5*xt::view(rotation, xt::all(), 0));
    auto s2_pos = xt::eval(translation - 0.5*xt::view(rotation, xt::all(), 0));
    auto s3_pos = xt::eval(translation - 0.2*xt::view(rotation, xt::all(), 0));

    scopi::sphere<dim> s1({s1_pos}, 0.4);
    scopi::sphere<dim> s2({s2_pos}, 0.4);
    scopi::sphere<dim> s3({s3_pos}, 0.3);

    // scopi::sphere<dim> s1({s1_pos}, 0.4);
    // scopi::sphere<dim> s2({s2_pos}, 0.4);

    scopi::type::position<dim> pos1 = {1, -2};
    scopi::type::position<dim> radius1 = {1, 1};
    scopi::type::position<dim-1> squareness1 = {0.5};
    scopi::type::quaternion quat1 = { std::cos(0.5*theta), 0, 0, std::sin(0.5*theta) };
    // scopi::type::quaternion quat1 = { -0.66693806,  0, 0, 0.74511316 };
    scopi::superellipsoid<dim> se1({pos1}, {quat1}, {radius1}, {squareness1});
    // se1.print();
    scopi::type::position<dim> pos2 = {12, 4};
    scopi::type::position<dim> radius2 = {2, 3};
    scopi::type::position<dim-1> squareness2 = {1.};
    scopi::type::quaternion quat2 = { 1,  0, 0, 0 };
    scopi::superellipsoid<dim> se2({pos2}, {quat2}, {radius2}, {squareness2});
    // se2.print();
    // // std::cout << "se1.point(0.43)   = " << se1.point(0.43) <<   " se2.point(0.65)   = " << se2.point(0.65) << std::endl;
    // // std::cout << "se1.normal(0.43)  = " << se1.normal(0.43) <<  " se2.normal(0.65)  = " << se2.normal(0.65) << std::endl;
    // // std::cout << "se1.tangent(0.43) = " << se1.tangent(0.43) << " se2.tangent(0.65) = " << se2.tangent(0.65) << std::endl;

    scopi::plan<dim> p1({translation}, theta);

    scopi::scopi_container<dim> particles;

    scopi::type::position<dim> dummy = {0, 0};

    particles.push_back(s1, {dummy}, {dummy}, {dummy});
    particles.push_back(s2, {dummy}, {dummy}, {dummy});
    particles.push_back(s3, {dummy}, {dummy}, {dummy});
    particles.push_back(p1, {dummy}, {dummy}, {dummy});

    // // particles.push_back(s1, {dummy}, {dummy}, {dummy});
    // // particles.push_back(s2, {dummy}, {dummy}, {dummy});
    // // particles.push_back(p1, {dummy}, {dummy}, {dummy});
    particles.push_back(se1, {dummy}, {dummy}, {dummy});
    particles.push_back(se2, {dummy}, {dummy}, {dummy});
    // particles[0]->print();

    std::vector<scopi::neighbor<dim>> contacts;
    double dmax = 0.05;

    for(std::size_t i = 0; i < particles.size() - 1; ++i)
    {
        for(std::size_t j = i + 1; j < particles.size(); ++j)
        {
            auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
            if (neigh.dij < dmax)
            {
                contacts.emplace_back(std::move(neigh));
            }
        }
    }
    std::cout << contacts.size() << std::endl;

    // test mosek

    Model::t M = new Model("pow1"); auto _M = finally([&]() { M->dispose(); });


    // Example POW1 https://docs.mosek.com/9.2/cxxfusion/tutorial-pow-shared.html

    Variable::t x  = M->variable("x", 3, Domain::unbounded());
    Variable::t x3 = M->variable();
    Variable::t x4 = M->variable();

    // Create the linear constraint
    auto aval = new_array_ptr<double, 1>({1.0, 1.0, 0.5});
    M->constraint(Expr::dot(x, aval), Domain::equalsTo(2.0));

    // Create the conic constraints
    M->constraint(Var::vstack(x->slice(0,2), x3), Domain::inPPowerCone(0.2));
    M->constraint(Expr::vstack(x->index(2), 1.0, x4), Domain::inPPowerCone(0.4));

    auto cval = new_array_ptr<double, 1>({1.0, 1.0, -1.0});
    M->objective(ObjectiveSense::Maximize, Expr::dot(cval, Var::vstack(x3, x4, x->index(0))));

    // Solve the problem
    M->solve();

    // Get the linear solution values
    ndarray<double, 1> xlvl   = *(x->level());
    std::cout << "x,y,z = " << xlvl << std::endl;

    // Example CQO1 https://docs.mosek.com/9.2/cxxfusion/tutorial-cqo-shared.html

    // Model::t M = new Model("cqo1"); auto _M = finally([&]() { M->dispose(); });
    // Variable::t x  = M->variable("x", 3, Domain::greaterThan(0.0));
    // Variable::t y  = M->variable("y", 3, Domain::unbounded());
    // // Create the aliases
    // //      z1 = [ y[0],x[0],x[1] ]
    // //  and z2 = [ y[1],y[2],x[2] ]
    // Variable::t z1 = Var::vstack(y->index(0),  x->slice(0, 2));
    // Variable::t z2 = Var::vstack(y->slice(1, 3), x->index(2));
    //
    // // Create the constraint
    // //      x[0] + x[1] + 2.0 x[2] = 1.0
    // auto aval = new_array_ptr<double, 1>({1.0, 1.0, 2.0});
    // M->constraint("lc", Expr::dot(aval, x), Domain::equalsTo(1.0));
    //
    // // Create the constraints
    // //      z1 belongs to C_3
    // //      z2 belongs to K_3
    // // where C_3 and K_3 are respectively the quadratic and
    // // rotated quadratic cone of size 3, i.e.
    // //                 z1[0] >= sqrt(z1[1]^2 + z1[2]^2)
    // //  and  2.0 z2[0] z2[1] >= z2[2]^2
    // Constraint::t qc1 = M->constraint("qc1", z1, Domain::inQCone());
    // Constraint::t qc2 = M->constraint("qc2", z2, Domain::inRotatedQCone());
    //
    // // Set the objective function to (y[0] + y[1] + y[2])
    // M->objective("obj", ObjectiveSense::Minimize, Expr::sum(y));
    //
    // // Solve the problem
    // M->solve();
    //
    // // Get the linear solution values
    // ndarray<double, 1> xlvl   = *(x->level());
    // ndarray<double, 1> ylvl   = *(y->level());
    // // Get conic solution of qc1
    // ndarray<double, 1> qc1lvl = *(qc1->level());
    // ndarray<double, 1> qc1dl  = *(qc1->dual());
    //
    // std::cout << "x1,x2,x2 = " << xlvl << std::endl;
    // std::cout << "y1,y2,y3 = " << ylvl << std::endl;
    // std::cout << "qc1 levels = " << qc1lvl << std::endl;
    // std::cout << "qc1 dual conic var levels = " << qc1dl << std::endl;


    pybind11::scoped_interpreter guard{};
    pybind11::gil_scoped_acquire acquire;
    pybind11::dict scope;
    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};
    pybind11::array_t<double> xn(arr1.size(), arr1.data());
    scope["x"] = xn;
    std::cout << "particles.size() = "<< particles.size() << std::endl;
    std::cout << "particles.pos().shape() = "<< xt::adapt(particles.pos().shape()) << std::endl;
    std::cout << "particles.pos() = "<< particles.pos() << std::endl;
    auto pp = xt::zeros<double>({particles.size(), dim});
    for(std::size_t ip = 0; ip < particles.size() - 1; ++ip){
      std::cout << "ip = "<< ip << " particles[ip]->pos() = "<< particles[ip]->pos() << std::endl;
      std::cout << "ip = "<< ip << " particles[ip]->q() = "<< particles[ip]->q() << std::endl;
      // if (scopi::sphere<dim,false> * d = dynamic_cast< scopi::sphere<dim,false> * >(particles[ip]))
      // {
      //   std::cout << "sucess" << std::endl;
      // }
      // else{
      //   std::cout << "fail" << std::endl;
      // }
    }

    // auto pos = particles.pos();
    // auto pos3 = pos.reshape({-1, 1}); // xt::flatten(pos);
    // auto arr2 = xt::flatten(arr1);
    // std::cout << "pos.shape() = "<< xt::adapt(pos.shape()) << std::endl;
    // std::cout << "pos3.shape() = "<< xt::adapt(pos3.shape()) << std::endl;
    // std::cout << "arr1.shape() = "<< xt::adapt(arr1.shape()) << std::endl;
    // std::cout << "arr2.shape() = "<< xt::adapt(arr2.shape()) << std::endl;
    // std::cout << "pos = "<< pos3 << std::endl;
    // std::cout << typeid(pos).name() << std::endl;
    // std::cout << pos.size() << std::endl;
    // std::cout << pos.data()  << std::endl;
    // // pybind11::array_t<double> pp(pos.size(), pos.data());


    pybind11::exec(R"(

      import numpy as np
      import pyvista as pv

      print("pyvista version : ",pv.__version__)
      print("x = ",_scope_)
      x = _scope_["x"]
      print("x = ",x)
      print("x.shape = ",x.shape)
      ## rx ry rz n e
      param_superellipsoids = np.array([[0.25, 0.25, 0.25, 1,   1],
                                        [1,    0.25, 0.25, 1,   1],
                                        [0.25, 1,    0.25, 1,   1],
                                        [0.25, 0.25, 1,    1,   1],
                                        [0.25, 0.5,  0.75, 0.9, 0.8]])

      #geoms = [pv.ParametricSuperEllipsoid(xradius=rx, yradius=ry, zradius=rz, n1=n1, n2=n2) for rx, ry, rz, n1, n2 in param_superellipsoids]



    )",
    pybind11::globals(),
    pybind11::dict(pybind11::arg("_scope_") = scope));

    std::fstream my_file;
    my_file.open("scopi_objects.json", std::ios::out);
    my_file << "{ \"objects\": [ ";
    for(std::size_t i = 0; i < particles.size(); ++i)
    {
        auto content = scopi::write_objects_dispatcher<dim>::dispatch(*particles[i]);
        std::cout << content << std::endl;
        my_file << "{" << content ;
        if (i == particles.size()-1) {
          my_file << "} ] }" << std::endl;
        }
        else {
          my_file << "}," << std::endl;
        }
    }
    my_file.close();

}

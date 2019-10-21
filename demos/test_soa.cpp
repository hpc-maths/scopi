#include <iostream>

#include "scopi/soa.hpp"
#include "nanoflann/nanoflann_sf.hpp"

struct particle
{
    double x, y, z;
    double r;
};

SOA_DEFINE_TYPE(particle, x, y, z, r);

struct contact
{
    int i, j;
    double d, ex, ey, ez;
};

SOA_DEFINE_TYPE(contact, i, j, d, ex, ey, ez);


int main()
{

    // soa::vector<particle> v;
    // v.resize(20);
    //
    // v.x[0] = 1.2;
    //
    // std::cout << v.x[0] << "\n";
    // std::cout << v[0].x << "\n";
    // std::cout << v.size() << "\n";
    //
    // auto print = [](const auto &particle) { std::cout << particle.x << "\n"; };
    // auto print_ = [](const auto &x) { std::cout << x << "\n"; };
    // std::for_each(v.rbegin(), v.rend(), print);
    // std::cout << "\n";
    //
    // std::for_each(v.x.rbegin(), v.x.rend(), print_);
    // std::cout << "\n";
    //
    // soa::vector<particle> v1;
    // v1 = std::move(v);
    // std::for_each(v1.begin(), v1.end(), print);
    // std::cout << "\n";
    // v1.push_back({1., 1., 1., 0});
    // std::cout << v.size() << "\n";
    // std::cout << v.capacity() << "\n";

    // int Np = 4;
    // soa::vector<particle> particles;
    // particles.resize(Np);
    //
    // particles.x[0] = 0.0; particles.y[0] = 0.0;
    // particles.z[0] = 0.0; particles.r[0] = 0.1;
    //
    // particles.x[1] = 1.0; particles.y[1] = 0.0;
    // particles.z[1] = 0.0; particles.r[1] = 0.5;
    //
    // particles.x[2] = 0.0; particles.y[2] = 1.0;
    // particles.z[2] = 1.0; particles.r[2] = 1.0;
    //
    // particles.x[3] = 0.0; particles.y[3] = 0.0;
    // particles.z[3] = 1.0; particles.r[3] = 1.5;
    //
    // auto print = [](const auto &particle) { std::cout << "(" <<
    //   particle.x << "," << particle.y << "," <<  particle.z << "," <<  particle.r << ")\n"; };
    // std::for_each(particles.rbegin(), particles.rend(), print);
    //

    int Np = 1000000;
    soa::vector<particle> particles;
    particles.resize(Np);
    double max_range = 10;
    double rmin = 0.1;
    double rmax = 0.1;
    for (int i = 0; i < Np; i++){
      particles[i].x = max_range * (rand() % 1000) / 1000.0;
      particles[i].y = max_range * (rand() % 1000) / 1000.0;
      particles[i].z = max_range * (rand() % 1000) / 1000.0;
      particles[i].r = rmin + (rmax-rmin) * (rand() % 1000) / 1000.0;
    }
    // auto print = [](const auto &particle) { std::cout << "(" <<
    //   particle.x << "," << particle.y << "," <<  particle.z << "," <<  particle.r << ")\n"; };
    // std::for_each(particles.begin(), particles.end(), print);


    // construct a kd-tree index:
    typedef nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, soa::vector<particle> > ,
      soa::vector<particle>,
      3 /* dim */
    > my_kd_tree_t;

    my_kd_tree_t   index(3 /*dim*/, particles, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */) );

    index.buildIndex();

    soa::vector<contact> contacts;
    for (int i = 0; i < Np; i++){
      double query_pt[4] = {particles[i].x,particles[i].y,particles[i].z,particles[i].r};
      //double query_pt[4] = {particles.x[i],particles.y[i],particles.z[i],particles.r[i]};
      //const double radius = sqrt(0.01);
      const double radius = 0.02;
      std::vector<std::pair<int, double> > indices_dists;
      nanoflann::RadiusResultSet<double,int> resultSet(radius, indices_dists);
      //const int nMatches = index.findNeighbors(resultSet, query_pt, nanoflann::SearchParams());
      std::vector<std::pair<unsigned long,double> >   ret_matches;
      const int nMatches = index.radiusSearch(query_pt, radius, ret_matches, nanoflann::SearchParams());
      //std::cout<<"pt "<<i<<" nMatches = "<<nMatches<<std::endl;
      for (int id=0;id<nMatches;id++){
        int j=ret_matches[id].first;
        double d = ret_matches[id].second;
        //std::cout<<"point = "<<j<<" dist = "<<d<<std::endl;
        if (i<j){
          double ex = particles.x[j]-particles.x[i];
          double ey = particles.y[j]-particles.y[i];
          double ez = particles.z[j]-particles.z[i];
          double norm = sqrt(ex*ex+ey*ey+ez*ez);
          contacts.push_back({i, j, norm-particles.r[i]-particles.r[j], ex/norm, ey/norm, ez/norm});
        }
      }
    }
    // auto print_contacts = [](const auto &contact) { std::cout << "(" <<
    //   contact.i << "," << contact.j << "," <<  contact.d << ","
    //   <<  contact.ex << "," << contact.ey << "," << contact.ez << ")\n"; };
    // std::for_each(contacts.rbegin(), contacts.rend(), print_contacts);
    std::cout<<"Nb de contacts : "<<contacts.size()<<std::endl;

    return 0;

}

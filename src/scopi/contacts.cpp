#include "scopi/contacts.hpp"

Contacts::~Contacts() {

}

Contacts::Contacts(const  double radius) : _radius(radius) {

}

void Contacts::compute_contacts(Particles& particles) {

  // particles.print();

  data.resize(0);

  constexpr std::size_t dim = 3;

  using my_kd_tree_t = typename nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, dim >;

  auto tic_timer = std::chrono::high_resolution_clock::now();

  // KdTree kd(xyzr);
  KdTree kd(particles.data);

  my_kd_tree_t index(
    dim, kd,
    nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)
  );
  index.buildIndex();

  auto toc_timer = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> time_span = toc_timer - tic_timer;
  double duration = time_span.count();
  std::cout << "\n-- C++ -- Contacts : CPUTIME (build index for contacts) = " << duration << std::endl;

  // for (const auto& p : particles.data) {
  for (std::size_t i=0; i<particles.data.size(); ++i) {

    double query_pt[3] = {particles.data.x[i], particles.data.y[i], particles.data.z[i]};

      std::vector<std::pair<size_t, double>> indices_dists;

      nanoflann::RadiusResultSet<double, std::size_t> resultSet(
        _radius, indices_dists);

      std::vector<std::pair<unsigned long, double>> ret_matches;

      const std::size_t nMatches = index.radiusSearch(query_pt, _radius, ret_matches,
          nanoflann::SearchParams());

      // std::cout << i << " nMatches = " << nMatches << std::endl;

      for (std::size_t ic = 0; ic < nMatches; ++ic) {

        std::size_t j = ret_matches[ic].first;
        //double dist = ret_matches[ic].second;

        if (i < j){

          // std::cout<<"contact : i = "<<i<<" j = "<<j<<std::endl;

          double ex = particles.data.x[j]-particles.data.x[i];
          double ey = particles.data.y[j]-particles.data.y[i];
          double ez = particles.data.z[j]-particles.data.z[i];
          double norm = std::sqrt(ex*ex+ey*ey+ez*ez);

          xt::xtensor<std::size_t, 1> xt_i = { i };
          xt::xtensor<std::size_t, 1> xt_j = { j };
          xt::xtensor<double, 1> xt_d = { norm-particles.data.r[i]-particles.data.r[j] };
          xt::xtensor<double, 1> xt_ex = { ex/norm };
          xt::xtensor<double, 1> xt_ey = { ey/norm };
          xt::xtensor<double, 1> xt_ez = { ez/norm };
          xt::xtensor<double, 1> xt_res = { 0 };
          xt::xtensor<double, 1> xt_lam = { 0 };

          xt_i = xt::concatenate(xt::xtuple(data.i,xt_i));
          xt_j = xt::concatenate(xt::xtuple(data.j,xt_j));
          xt_d = xt::concatenate(xt::xtuple(data.d,xt_d));
          xt_ex = xt::concatenate(xt::xtuple(data.ex,xt_ex));
          xt_ey = xt::concatenate(xt::xtuple(data.ey,xt_ey));
          xt_ez = xt::concatenate(xt::xtuple(data.ez,xt_ez));
          xt_res = xt::concatenate(xt::xtuple(data.res,xt_res));
          xt_lam = xt::concatenate(xt::xtuple(data.lam,xt_lam));

          data.resize(xt_i.size());
          data.i = xt_i;
          data.j = xt_j;
          data.d = xt_d;
          data.ex = xt_ex;
          data.ey = xt_ey;
          data.ez = xt_ez;
          data.res = xt_res;
          data.lam = xt_lam;

        }
      }
    }
}

xt::xtensor<double, 2> Contacts::get_data() const{
  return xt::stack(xt::xtuple(
    xt::flatten(data.i),
    xt::flatten(data.j),
    xt::flatten(data.d),
    xt::flatten(data.ex),
    xt::flatten(data.ey),
    xt::flatten(data.ez),
    xt::flatten(data.res),
    xt::flatten(data.lam) ),1);
}


void Contacts::print(){

  std::cout<<"\n-- C++ -- Contacts : number of contacts = "<<data.size()<<std::endl;
  auto print_data = [](const auto &c) {  std::cout << "-- C++ -- Contacts : " << c.i << " " << c.j  << " " << c.d  << " " << c.ex << " " << c.ey << " " << c.ez << " " << c.res << " " << c.lam << "\n"; };
  std::for_each(data.begin(), data.end(), print_data);
  std::cout<<"-- C++ -- Contacts : data memory address = "<<&data.d[0] << std::endl;

}

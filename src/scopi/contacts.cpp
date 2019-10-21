#include "scopi/contacts.hpp"


Contacts::~Contacts() {
}

Contacts::Contacts() {

}

xt::pyarray<double> Contacts::compute_contacts(xt::pyarray<double> &xyzr, const  double radius) {



  constexpr std::size_t dim = 3;

  using my_kd_tree_t = typename nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, dim >;

  KdTree kd(xyzr);

  my_kd_tree_t index(
    dim, kd,
    nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)
  );
  index.buildIndex();

  xt::pyarray<double> contacts = NULL;
  std::size_t cc = 0;

  for (std::size_t i = 0; i < xyzr.shape()[0]; ++i)
  {
    double query_pt[3] = {xyzr(i,0),xyzr(i,1),xyzr(i,2)};

    std::vector<std::pair<size_t, double>> indices_dists;
    nanoflann::RadiusResultSet<double, std::size_t> resultSet(
      radius, indices_dists);

    std::vector<std::pair<unsigned long, double>> ret_matches;
    const std::size_t nMatches = index.radiusSearch(query_pt, radius, ret_matches,
        nanoflann::SearchParams());

    //std::cout << "pt " << i << " nMatches = " << nMatches << std::endl;

    for (std::size_t ic = 0; ic < nMatches; ++ic)
    {
      std::size_t j = ret_matches[ic].first;
      //double dist = ret_matches[ic].second;
      if (i < j){

        //std::cout<<"contact : i = "<<i<<" j = "<<j<<std::endl;

        double ex = xyzr(j,0)-xyzr(i,0);
        double ey = xyzr(j,1)-xyzr(i,1);
        double ez = xyzr(j,2)-xyzr(i,2);
        double norm = std::sqrt(ex*ex+ey*ey+ez*ez);


        if (cc==0){
          contacts = {{double(i), double(j), norm-xyzr(i,3)-xyzr(j,3), ex/norm, ey/norm, ez/norm, 0.0, 0.0}};
        }
        else{
          contacts = xt::concatenate(xt::xtuple(contacts, xt::xarray<double>({{double(i), double(j), norm-xyzr(i,3)-xyzr(j,3), ex/norm, ey/norm, ez/norm, 0.0, 0.0}})));
        }

        cc += 1;
      }
    }
  }
  //std::cout<<"contacts = "<<contacts<<std::endl;

  return contacts;
}


void Contacts::print(){
  auto print_data = [](const auto &c) {  std::cout << "  Contacts : " << c.i << " " << c.j  << " " << c.d  << " " << c.ex << " " << c.ey << " " << c.ez << " " << c.res << " " << c.lam << "\n"; };
  std::for_each(_data.begin(), _data.end(), print_data);
  std::cout << "\n";

}

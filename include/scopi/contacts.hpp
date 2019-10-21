#ifndef DEF_Contacts
#define DEF_Contacts

#include <iostream>
#include <vector>
#include <typeinfo>

#include <xtensor/xtensor.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xdynamic_view.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor-python/pyarray.hpp>
#include "nanoflann/nanoflann.hpp"
#include "scopi/soa_xtensor.hpp"

struct contact
{
  std::size_t i, j;
  double d, ex, ey, ez, res, lam;
};

SOA_DEFINE_TYPE(contact, i, j, d, ex, ey, ez, res, lam);


class KdTree {

  public:
    KdTree(xt::pyarray<double> &xyzr ) : _xyzr{xyzr} {
    }
    inline std::size_t kdtree_get_point_count() const {
      return _xyzr.shape()[0];
    }
    inline double kdtree_get_pt(std::size_t idx, const std::size_t dim) const {
      return _xyzr(idx,dim);
    }
    template<class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const {
      return false;
    }

  private:
    xt::pyarray<double> &_xyzr;
};

/// @class Contacts
/// @bried This class manages contacts
class Contacts
{
public:

  /// @brief Constructor
  /// Instantiate np contacts
  Contacts();

  /// @brief Destructor
  ~Contacts();

  xt::pyarray<double> compute_contacts(xt::pyarray<double> &xyzr, const  double radius);

  /// @brief Print the contacts
  void print();

  xt::pyarray<double> xyzr();  // A REVOIR : copies...

private:

  soa::vector<contact> _data;

};

#endif

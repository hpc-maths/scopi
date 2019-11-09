#ifndef DEF_Contacts
#define DEF_Contacts

#include "scopi/particles.hpp"
#include "nanoflann/nanoflann.hpp"


struct contact
{
  std::size_t i, j;
  double d, ex, ey, ez, res, lam;
};

SOA_DEFINE_TYPE(contact, i, j, d, ex, ey, ez, res, lam);


class KdTree {

public:

  KdTree(soa::vector<particle> &p) : _p{p}{
  }

  inline std::size_t kdtree_get_point_count() const{
    return _p.size();
  }

  inline double kdtree_get_pt(std::size_t idx, const std::size_t dim) const{
    if (dim == 0)
    return _p.x[idx];
    else if (dim == 1)
    return _p.y[idx];
    else
    return _p.z[idx];
  }

  template<class BBOX>
  bool kdtree_get_bbox(BBOX & /* bb */) const{
    return false;
  }

private:
  soa::vector<particle> &_p;

};

/// @class Contacts
/// @bried This class manages contacts
class Contacts
{
public:

  /// @brief Constructor
  /// Instantiate np contacts
  Contacts(const double radius);

  /// @brief Destructor
  ~Contacts();

  void compute_contacts(Particles& particles);

  /// @brief Print the contacts
  void print();

  /// @brief Get contacts list : only for access from python
  xt::xtensor<double, 2> get_data() const;

  soa::vector<contact> data;

private:

  const double _radius;

};

#endif

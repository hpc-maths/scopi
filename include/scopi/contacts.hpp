#ifndef DEF_Contacts
#define DEF_Contacts

#include <nanoflann/nanoflann.hpp>

namespace scopi
{

  ///////////////////////
  // KdTree definition //
  ///////////////////////
  template<std::size_t dim>
  class KdTree
  {
    public:
      KdTree(scopi::scopi_container<dim> &p) : _p{p}{}
      inline std::size_t kdtree_get_point_count() const
      {
        return _p.size();
      }
      inline double kdtree_get_pt(std::size_t idx, const std::size_t d) const
      {
        return _p[idx]->pos()[d];
      }
      template<class BBOX>
      bool kdtree_get_bbox(BBOX & /* bb */) const
      {
        return false;
      }
    private:
      scopi::scopi_container<dim> &_p;
  };

  /////////////////////////
  // Contacts definition //
  /////////////////////////
  template<std::size_t dim>
  class Contacts
  {
    public:
      /// @brief Constructor
      /// Instantiate np contacts
      Contacts(const double search_radius);
      /// @brief Destructor
      ~Contacts();
      /// @brief Compute contacts
      void compute_contacts(scopi::scopi_container<dim> &p);
      /// @brief Print the contacts
      void print();
      /// @brief Return some private variables
      std::vector<int> i();
      std::vector<int> j();
      std::vector<double> ex();
      std::vector<double> ey();
      std::vector<double> ez();
      std::vector<double> d();
    private:
      const double _search_radius;
      std::vector<int> _i, _j;
      std::vector<double> _ex, _ey, _ez;
      std::vector<double> _d;
  };

  /////////////////////////////
  // Contacts implementation //
  /////////////////////////////

  template<std::size_t dim>
  Contacts<dim>::~Contacts()
  {
  }

  template<std::size_t dim>
  Contacts<dim>::Contacts(const double search_radius) : _search_radius(search_radius)
  {
  }

  template<std::size_t dim>
  void Contacts<dim>::compute_contacts(scopi::scopi_container<dim> &particles)
  {
    using my_kd_tree_t = typename nanoflann::KDTreeSingleIndexAdaptor<
      nanoflann::L2_Simple_Adaptor<double, scopi::KdTree<dim>>, scopi::KdTree<dim>, dim >;

    scopi::KdTree kd(particles);

    // auto tic_timer = std::chrono::high_resolution_clock::now();
    tic();

    my_kd_tree_t index(
      dim, kd,
      nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)
    );
    index.buildIndex();

    auto duration = toc();
    std::cout << "\n-- C++ -- Contacts: CPUTIME (build index...) = " << duration << std::endl;

    tic();

    _i.resize(0); _j.resize(0);
    _ex.resize(0); _ey.resize(0); _ez.resize(0);
    _d.resize(0);

    for (std::size_t i=0; i<particles.size(); ++i) {

      double query_pt[3] = {particles[i]->pos()[0], particles[i]->pos()[1], particles[i]->pos()[2]};

      std::vector<std::pair<size_t, double>> indices_dists;
      double radius = 2;

      nanoflann::RadiusResultSet<double, std::size_t> resultSet(
          radius, indices_dists);

        std::vector<std::pair<unsigned long, double>> ret_matches;

        const std::size_t nMatches = index.radiusSearch(query_pt, radius, ret_matches,
            nanoflann::SearchParams());

        // std::cout << i << " nMatches = " << nMatches << std::endl;

        for (std::size_t ic = 0; ic < nMatches; ++ic) {

          std::size_t j = ret_matches[ic].first;
          double dist = ret_matches[ic].second;

          if (i < j)
          {
            double ex = particles[j]->pos()[0]-particles[i]->pos()[0];
            double ey = particles[j]->pos()[1]-particles[i]->pos()[1];
            double ez = 0;
            if (dim==3)
            {
              ez = particles[j]->pos()[2]-particles[i]->pos()[2];
            }
            double norm = std::sqrt(ex*ex+ey*ey+ez*ez);
            ex = ex/norm; ey = ey/norm; ez = ez/norm;
            _i.push_back(i);
            _j.push_back(j);
            _ex.push_back(ex);
            _ey.push_back(ey);
            _ez.push_back(ez);
            _d.push_back(norm);
            // _d.push_back(norm - particles[i]->radius() - particles[j]->radius());
            // std::cout << "---> contact (" << i << "," << j << ") distance = "<< norm << std::endl;
          }
        }
    }

    // auto toc_timer = std::chrono::high_resolution_clock::now();
    // auto time_span = toc_timer - tic_timer;
    // auto duration = time_span.count();
    duration = toc();
    std::cout << "\n-- C++ -- Contacts: CPUTIME (build vectors...) = " << duration << "( "<< _i.size() << " contacts )" << std::endl;

  }

  template<std::size_t dim>
  void Contacts<dim>::print()
  {
    for (std::size_t ic = 0; ic < _i.size(); ++ic)
    {
      std::cout << "---> contact (" << _i[ic] << "," << _j[ic] << ") d = "<< _d[ic];
      std::cout << " e = ("<< _ex[ic] << "," << _ey[ic] <<","<< _ey[ic] << ")"<< std::endl;
    }
  }

  template<std::size_t dim>
  std::vector<int> Contacts<dim>::i()
  {
    return _i;
  }

  template<std::size_t dim>
  std::vector<int> Contacts<dim>::j()
  {
    return _j;
  }

  template<std::size_t dim>
  std::vector<double> Contacts<dim>::ex()
  {
    return _ex;
  }

  template<std::size_t dim>
  std::vector<double> Contacts<dim>::ey()
  {
    return _ey;
  }

  template<std::size_t dim>
  std::vector<double> Contacts<dim>::ez()
  {
    return _ez;
  }

  template<std::size_t dim>
  std::vector<double> Contacts<dim>::d()
  {
    return _d;
  }


  // struct contact
  // {
  //   std::size_t i, j;
  //   double d, ex, ey, ez, res, lam;
  // };
  //
  // SOA_DEFINE_TYPE(contact, i, j, d, ex, ey, ez, res, lam);
  //
  //
  // class KdTree {
  //
  // public:
  //
  //   KdTree(soa::vector<particle> &p) : _p{p}{
  //   }
  //
  //   inline std::size_t kdtree_get_point_count() const{
  //     return _p.size();
  //   }
  //
  //   inline double kdtree_get_pt(std::size_t idx, const std::size_t dim) const{
  //     if (dim == 0)
  //     return _p.x[idx];
  //     else if (dim == 1)
  //     return _p.y[idx];
  //     else
  //     return _p.z[idx];
  //   }
  //
  //   template<class BBOX>
  //   bool kdtree_get_bbox(BBOX & /* bb */) const{
  //     return false;
  //   }
  //
  // private:
  //   soa::vector<particle> &_p;
  //
  // };
  //
  // /// @class Contacts
  // /// @bried This class manages contacts
  // class Contacts
  // {
  // public:
  //
  //   /// @brief Constructor
  //   /// Instantiate np contacts
  //   Contacts(const double radius);
  //
  //   /// @brief Destructor
  //   ~Contacts();
  //
  //   void compute_contacts(Particles& particles, Obstacles& obstacles);
  //
  //   /// @brief Print the contacts
  //   void print();
  //
  //   /// @brief Get contacts list : only for access from python
  //   xt::xtensor<double, 2> get_data() const;
  //
  //   soa::vector<contact> data;
  //
  // private:
  //
  //   const double _radius;
  //
  // };

}

#endif

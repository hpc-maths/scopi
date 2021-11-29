#pragma once 

#include "base.hpp"

namespace scopi
{

    template<std::size_t dim>
    class KdTree
    {
      public:
        KdTree(scopi::scopi_container<dim> &p, std::size_t actptr) : _p{p}, _actptr{actptr} {}
        inline std::size_t kdtree_get_point_count() const
        {
          //std::cout << "KDTREE _p.size() = "<< _p.size() <<std::endl;
          return _p.size();
        }
        inline double kdtree_get_pt(std::size_t idx, const std::size_t d) const
        {
          //std::cout << "KDTREE _p["<< _actptr+idx << "][" << d << "] = " << _p.pos()(_actptr+idx)[d] << std::endl;
          return _p.pos()(_actptr+idx)(d); //_p[idx]->pos()[d];
          // return _p.pos()(idx)[d];
        }
        template<class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const
        {
          return false;
        }
      private:
        scopi::scopi_container<dim> &_p;
        std::size_t _actptr;
    };



    class contact_kdtree: public contact_base<contact_kdtree>
    {
    public:
        using base_type = contact_base<contact_kdtree>;

        contact_kdtree(double dmax, double kdtree_radius): contact_base(dmax), _kdtree_radius(kdtree_radius){};

        template <std::size_t dim>
        std::vector<scopi::neighbor<dim>> run_impl(scopi_container<dim>& particles, std::size_t active_ptr)
        {
            std::cout << "----> CONTACTS : run implementation contact_kdtree" << std::endl;

            std::vector<scopi::neighbor<dim>> contacts;
            // double dmax = 2;

            // utilisation de kdtree pour ne rechercher les contacts que pour les particules proches
            tic();
            using my_kd_tree_t = typename nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<double, KdTree<dim>>, KdTree<dim>, dim >;
            KdTree<dim> kd(particles,active_ptr);
            my_kd_tree_t index(
            dim, kd,
            nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)
            );
            index.buildIndex();
            auto duration = toc();
            std::cout << "----> CPUTIME : build kdtree index = " << duration << std::endl;

            tic();

            #pragma omp parallel for num_threads(8)

            for(std::size_t i = active_ptr; i < particles.size() - 1; ++i)
            {

                // for(std::size_t j = i + 1; j < particles.size(); ++j)
                // {
                //     auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                //     if (neigh.dij < dmax)
                //     {
                //         contacts.emplace_back(std::move(neigh));
                //         contacts.back().i = i;
                //         contacts.back().j = j;
                //     }
                // }

                double query_pt[dim];
                for (std::size_t d=0; d<dim; ++d)
                {
                    query_pt[d] = particles.pos()(i)(d);
                    // query_pt[d] = particles.pos()(i)(d);
                }
                //std::cout << "i = " << i << " query_pt = " << query_pt[0] << " " << query_pt[1] << std::endl;

                std::vector<std::pair<size_t, double>> indices_dists;

                nanoflann::RadiusResultSet<double, std::size_t> resultSet(
                    _kdtree_radius, indices_dists);

                std::vector<std::pair<unsigned long, double>> ret_matches;

                const std::size_t nMatches = index.radiusSearch(query_pt, _kdtree_radius, ret_matches,
                    nanoflann::SearchParams());

                //std::cout << i << " nMatches = " << nMatches << std::endl;

                for (std::size_t ic = 0; ic < nMatches; ++ic) {

                    std::size_t j = ret_matches[ic].first;
                    //double dist = ret_matches[ic].second;
                    if (i < j)  { //&& (j>=active_ptr)
                      auto neigh = scopi::closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                      if (neigh.dij < _dmax) {
                          neigh.i = i;
                          neigh.j = j;
                          #pragma omp critical
                          contacts.emplace_back(std::move(neigh));
                          // contacts.back().i = i;
                          // contacts.back().j = j;
                      }
                    }

                }

            }

            duration = toc();
            std::cout << "----> CPUTIME : compute " << contacts.size() << " contacts = " << duration << std::endl;



            tic();
            std::sort(contacts.begin(), contacts.end(), [](auto& a, auto& b )
            {
                if (a.i < b.i) {
                  return true;
                }
                else {
                  if (a.i == b.i) {
                    return a.j < b.j;
                  }
                }
                return false;
            });
            duration = toc();
            std::cout << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration << std::endl;

            for(std::size_t ic=0; ic<contacts.size(); ++ic)
            {
                std::cout << "----> CONTACTS : i j = " << contacts[ic].i << " " << contacts[ic].j << " d = " <<  contacts[ic].dij << std::endl;
                // std::cout << "----> CONTACTS : contact = " << contacts[ic] << std::endl;
            }

            return contacts;

        }

      private:
        double _kdtree_radius;

    };

}

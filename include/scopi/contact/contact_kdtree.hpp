#pragma once

#include "base.hpp"

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{

    template<std::size_t dim>
    class KdTree
    { 
    public:
        KdTree(scopi_container<dim> &p, std::size_t actptr) : m_p{p}, m_actptr{actptr} {}
        inline std::size_t kdtree_get_point_count() const
        {
          //std::cout << "KDTREE m_p.size() = "<< m_p.size() <<std::endl;
          return m_p.size();
        }
        inline double kdtree_get_pt(std::size_t idx, const std::size_t d) const
        {
          //std::cout << "KDTREE m_p["<< m_actptr+idx << "][" << d << "] = " << m_p.pos()(m_actptr+idx)[d] << std::endl;
          return m_p.pos()(m_actptr+idx)(d); //m_p[idx]->pos()[d];
          // return m_p.pos()(idx)[d];
        }
        template<class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const
        {
          return false;
        }
    private:
        scopi_container<dim> &m_p;
        std::size_t m_actptr;
    };

    class contact_kdtree: public contact_base<contact_kdtree>
    {
    public:
        using base_type = contact_base<contact_kdtree>;

        contact_kdtree(double dmax, double kdtree_radius=10)
        : contact_base(dmax)
        , m_kd_tree_radius(kdtree_radius)
        , m_nMatches(0)
        {};

        std::size_t get_nMatches() const
        {
            return m_nMatches;
        }

        template <std::size_t dim>
        std::vector<neighbor<dim>> run_impl(scopi_container<dim>& particles, std::size_t active_ptr)
        {
            // std::cout << "----> CONTACTS : run implementation contact_kdtree" << std::endl;

            std::vector<neighbor<dim>> contacts;
            // double dmax = 2;

            // utilisation de kdtree pour ne rechercher les contacts que pour les particules proches
            tic();
            using my_kd_tree_t = typename nanoflann::KDTreeSingleIndexAdaptor<
            nanoflann::L2_Simple_Adaptor<double, KdTree<dim>>, KdTree<dim>, dim, std::size_t>;
            KdTree<dim> kd(particles,active_ptr);
            my_kd_tree_t index(
            dim, kd,
            nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)
            );
            index.buildIndex();
            auto duration = toc();
            PLOG_INFO << "----> CPUTIME : build kdtree index = " << duration << std::endl;

            tic();

            m_nMatches = 0;
            #pragma omp parallel for reduction(+:m_nMatches) //num_threads(8)

            for (std::size_t i = active_ptr; i < particles.size() - 1; ++i)
            {
                // for (std::size_t j = i + 1; j < particles.size(); ++j)
                // {
                //     auto neigh = closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                //     if (neigh.dij < dmax)
                //     {
                //         contacts.emplace_back(std::move(neigh));
                //         contacts.back().i = i;
                //         contacts.back().j = j;
                //     }
                // }

                double query_pt[dim];
                for (std::size_t d = 0; d < dim; ++d)
                {
                    query_pt[d] = particles.pos()(i)(d);
                    // query_pt[d] = particles.pos()(i)(d);
                }
                //std::cout << "i = " << i << " query_pt = " << query_pt[0] << " " << query_pt[1] << std::endl;

                std::vector<std::pair<size_t, double>> indices_dists;

                nanoflann::RadiusResultSet<double, std::size_t> resultSet(
                    m_kd_tree_radius, indices_dists);

                std::vector<std::pair<std::size_t, double>> ret_matches;

                auto nMatches_loc = index.radiusSearch(query_pt, m_kd_tree_radius, ret_matches,
                    nanoflann::SearchParams());

                //std::cout << i << " nMatches = " << nMatches << std::endl;

                for (std::size_t ic = 0; ic < nMatches_loc; ++ic) {

                    std::size_t j = ret_matches[ic].first;
                    //double dist = ret_matches[ic].second;
                    if (i < j)  { //&& (j>=active_ptr)
                      auto neigh = closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                      m_nMatches++;
                      std::size_t size = 6;
                      for (std::size_t gi = 0; gi < size; ++gi)
                      {
                          for (std::size_t gj = 0; gj < size; ++gj)
                          {
                              if (neigh[size*gi+gj].dij < m_dmax) {
                                  neigh[size*gi+gj].i = i*size + gi;
                                  neigh[size*gi+gj].j = j*size + gj;
                                  #pragma omp critical
                                  contacts.emplace_back(std::move(neigh[size*gi+gj]));
                                  // contacts.back().i = i;
                                  // contacts.back().j = j;
                              }
                          }
                      }
                    }

                }

            }

            /*
            // obstacles
            for (std::size_t i = 0; i < active_ptr; ++i)
            {
                for (std::size_t j = active_ptr; j < particles.size(); ++j)
                {
                    auto neigh = closest_points_dispatcher<dim>::dispatch(*particles[i], *particles[j]);
                    if (neigh.dij < m_dmax)
                    {
                        neigh.i = i;
                        neigh.j = j;
                        contacts.emplace_back(std::move(neigh));
                    }
                }
            }
            */

            duration = toc();
            PLOG_INFO << "----> CPUTIME : compute " << contacts.size() << " contacts = " << duration << " compute " << m_nMatches << " distances" << std::endl;



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
            PLOG_INFO << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration << std::endl;

            /*
            for (std::size_t ic=0; ic<contacts.size(); ++ic)
            {
                std::cout << "----> CONTACTS : i j = " << contacts[ic].i << " " << contacts[ic].j << " d = " <<  contacts[ic].dij << std::endl;
                // std::cout << "----> CONTACTS : contact = " << contacts[ic] << std::endl;
            }
            */

            return contacts;

        }

    private:
        double m_kd_tree_radius;
        std::size_t m_nMatches;

    };

}

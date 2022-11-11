#pragma once

#include "base.hpp"
#include "../box.hpp"
#include "../utils.hpp"
#include <CLI/CLI.hpp>

#include <cstddef>
#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"

namespace scopi
{

    class contact_kdtree;

    /**
     * @brief Parameters for contact_kdtree.
     *
     * Specialization of ContactsParams.
     */
    template<>
    struct ContactsParams<contact_kdtree>
    {
        /**
         * @brief Default constructor.
         */
        ContactsParams();

        void init_options(CLI::App& app);
        /**
         * @brief Maximum distance between two neighboring particles.
         *
         * Default value: 2.
         * \note \c dmax > 0
         */
        double dmax;
        /**
         * @brief Kd-tree radius.
         *
         * For two particles \c i and \c j, compute the exact distance only if the squared distance between a point in \c i and a point in \c j is less than \c kdtree_radius.
         * For a sphere or a superellipsoid, this point is the center.
         * For a plan, it is the point used to construct it.
         * \note \c kd_tree_radius > 0
         */
        double kd_tree_radius;
    };

    /**
     * @brief
     *
     * \todo Write documentation.
     *
     * @tparam dim Dimension (2 or 3).
     */
    template<std::size_t dim>
    class KdTree
    {
    public:
        /**
         * @brief
         *
         * \todo Write documentation.
         *
         * @param p [in] Array of particles.
         * @param actptr [in] Index of the first active particle.
         */
        KdTree(scopi_container<dim> &p, std::size_t actptr) : m_p{p}, m_actptr{actptr} {}
        inline std::size_t kdtree_get_point_count() const
        {
          //std::cout << "KDTREE m_p.size() = "<< m_p.size() <<std::endl;
          return m_p.pos().size();
          // return m_p.pos().size() - m_actptr;
        }
        /**
         * @brief
         *
         * \todo Write documentation.
         *
         * @param idx
         * @param d
         *
         * @return
         */
        inline double kdtree_get_pt(std::size_t idx, const std::size_t d) const
        {
          //std::cout << "KDTREE m_p["<< m_actptr+idx << "][" << d << "] = " << m_p.pos()(m_actptr+idx)[d] << std::endl;
          return m_p.pos()(m_actptr+idx)(d); //m_p[idx]->pos()[d];
          // return m_p.pos()(idx)[d];
        }
        /**
         * @brief
         *
         * \todo Write documentation.
         *
         * @tparam BBOX
         * @param
         *
         * @return
         */
        template<class BBOX>
        bool kdtree_get_bbox(BBOX & /* bb */) const
        {
          return false;
        }
    private:
        /**
         * @brief Array of particles.
         */
        scopi_container<dim> &m_p;
        /**
         * @brief Index of the first active particle.
         */
        std::size_t m_actptr;
    };

    /**
     * @brief Contacts with Kd-tree.
     *
     * Use a Kd-tree to select particles close enough the compute the exact distance.
     */
    class contact_kdtree: public contact_base<contact_kdtree>
    {
    public:
        /**
         * @brief Alias for the base class contact_base.
         */
        using base_type = contact_base<contact_kdtree>;

        /**
         * @brief Constructor.
         *
         * @param params [in] Parameters.
         */
        contact_kdtree(const ContactsParams<contact_kdtree>& params = ContactsParams<contact_kdtree>());
        /**
         * @brief Get the number of exact distances computed.
         */
        std::size_t get_nMatches() const;

        /**
         * @brief Compute neighboring particles.
         *
         * Compute contacts between particles using a Kd-tree to select particles close enough.
         * Then, compute the exact distance.
         *
         * Only the contact between particles \c i and \c j is computed, not the contact between \c j and \c i, with \c i < \c j.
         *
         * The returned array of neighbors is sorted.
         * See sort_contacts.
         *
         * @tparam dim Dimension (2 or 3).
         * @param particles [in] Array of particles.
         * @param active_ptr [in] Index of the first active particle.
         *
         * @return Array of neighbors.
         */
        template <std::size_t dim>
        std::vector<neighbor<dim>> run_impl(const BoxDomain<dim>& box, scopi_container<dim>& particles, std::size_t active_ptr);

    private:
        /**
         * @brief Number of exact distances computed.
         */
        std::size_t m_nMatches;
    };

    template <std::size_t dim>
    std::vector<neighbor<dim>> contact_kdtree::run_impl(const BoxDomain<dim>& box, scopi_container<dim>& particles, std::size_t active_ptr)
    {
        // std::cout << "----> CONTACTS : run implementation contact_kdtree" << std::endl;

        std::vector<neighbor<dim>> contacts;

        add_objects_from_periodicity(box, particles, this->m_params.dmax);

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
        #pragma omp parallel for reduction(+:m_nMatches) //num_threads(1)

        for (std::size_t i = particles.offset(active_ptr); i < particles.pos().size() - 1; ++i)
        {
            double query_pt[dim];
            for (std::size_t d = 0; d < dim; ++d)
            {
                query_pt[d] = particles.pos()(i)(d);
                // query_pt[d] = particles.pos()(i)(d);
            }
            PLOG_INFO << "i = " << i << " query_pt = " << query_pt[0] << " " << query_pt[1] << std::endl;

            std::vector<std::pair<size_t, double>> indices_dists;

            nanoflann::RadiusResultSet<double, std::size_t> resultSet(
                this->m_params.kd_tree_radius, indices_dists);

            std::vector<std::pair<std::size_t, double>> ret_matches;

            auto nMatches_loc = index.radiusSearch(query_pt, this->m_params.kd_tree_radius, ret_matches,
                nanoflann::SearchParams());

            for (std::size_t ic = 0; ic < nMatches_loc; ++ic) {
                std::size_t j = ret_matches[ic].first;
                if (i < j)
                {
                    compute_exact_distance(box, particles, i, j, contacts, this->m_params.dmax);
                    m_nMatches++;
                }
            }
        }

        // obstacles
        for (std::size_t i = 0; i < active_ptr; ++i)
        {
            for (std::size_t j = active_ptr; j < particles.size(); ++j)
            {
                compute_exact_distance(box, particles, i, j, contacts, this->m_params.dmax);
            }
        }

        duration = toc();
        PLOG_INFO << "----> CPUTIME : compute " << contacts.size() << " contacts = " << duration << " compute " << m_nMatches << " distances" << std::endl;



        tic();
        sort_contacts(contacts);
        duration = toc();
        PLOG_INFO << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration << std::endl;

        /*
        for (std::size_t ic=0; ic<contacts.size(); ++ic)
        {
            std::cout << "----> CONTACTS : i j = " << contacts[ic].i << " " << contacts[ic].j << " d = " <<  contacts[ic].dij << std::endl;
            // std::cout << "----> CONTACTS : contact = " << contacts[ic] << std::endl;
        }
        */

        particles.reset_periodic();

        return contacts;

    }
}

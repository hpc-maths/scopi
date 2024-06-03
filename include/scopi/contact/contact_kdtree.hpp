#pragma once

#include "../box.hpp"
#include "../scopi.hpp"
#include "../utils.hpp"
#include "base.hpp"
#include <CLI/CLI.hpp>

#include <cstddef>
#include <plog/Initializers/RollingFileInitializer.h>
#include <plog/Log.h>

#include <nanoflann.hpp>

namespace scopi
{

    template <class problem_t>
    class contact_kdtree;

    /**
     * @brief Parameters for contact_kdtree.
     *
     * Specialization of ContactsParams.
     */
    template <class problem_t>
    struct ContactsParams<contact_kdtree<problem_t>>
    {
        /**
         * @brief Default constructor.
         */
        ContactsParams() // cppcheck-suppress uninitMemberVar
            : dmax(2.)
            , kd_tree_radius(17.)
        {
        }

        void init_options()
        {
            auto& app = get_app();
            auto opt  = app.add_option_group("KD tree options");
            opt->add_option("--dmax", dmax, "Maximum distance between two neighboring particles")->capture_default_str();
            opt->add_option("--kd-radius", kd_tree_radius, "Kd-tree radius")->capture_default_str();
        }

        /**
         * @brief Maximum distance between two neighboring particles.
         *
         * Default value: 2.
         * \note \c dmax > 0
         */
        double dmax = 2.;
        /**
         * @brief Kd-tree radius.
         *
         * For two particles \c i and \c j, compute the exact distance only if the squared distance between a point in \c i and a point in
         * \c j is less than \c kdtree_radius. For a sphere or a superellipsoid, this point is the center. For a plan, it is the point used
         * to construct it. \note \c kd_tree_radius > 0
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
    template <std::size_t dim>
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
        KdTree(scopi_container<dim>& p, std::size_t actptr)
            : m_p{p}
            , m_actptr{actptr}
        {
        }

        inline std::size_t kdtree_get_point_count() const
        {
            // std::cout << "KDTREE m_p.size() = "<< m_p.size() <<std::endl;
            //  return m_p.pos().size();
            return m_p.pos().size() - m_actptr;
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
            // std::cout << "KDTREE m_p["<< m_actptr+idx << "][" << d << "] = " << m_p.pos()(m_actptr+idx)[d] << std::endl;
            return m_p.pos()(m_actptr + idx)(d); // m_p[idx]->pos()[d];
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
        template <class BBOX>
        bool kdtree_get_bbox(BBOX& /* bb */) const
        {
            return false;
        }

      private:

        /**
         * @brief Array of particles.
         */
        scopi_container<dim>& m_p;
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
    template <class problem_t>
    class contact_kdtree : public contact_base<contact_kdtree<problem_t>>
    {
      public:

        /**
         * @brief Alias for the base class contact_base.
         */
        using base_type = contact_base<contact_kdtree<problem_t>>;

        /**
         * @brief Constructor.
         *
         * @param params [in] Parameters.
         */
        explicit contact_kdtree(const ContactsParams<contact_kdtree<problem_t>>& params = ContactsParams<contact_kdtree<problem_t>>())
            : base_type(params)
            , m_nMatches(0)
        {
        }

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
        auto run_impl(const BoxDomain<dim>& box, scopi_container<dim>& particles, std::size_t active_ptr);

        auto& default_contact_property()
        {
            return m_default_contact_property;
        }

      private:

        /**
         * @brief Number of exact distances computed.
         */
        std::size_t m_nMatches;
        contact_property<problem_t> m_default_contact_property;
    };

    template <class problem_t>
    template <std::size_t dim>
    auto contact_kdtree<problem_t>::run_impl(const BoxDomain<dim>& box, scopi_container<dim>& particles, std::size_t active_ptr)
    {
        // std::cout << "----> CONTACTS : run implementation contact_kdtree" << std::endl;

        std::vector<neighbor<dim, problem_t>> contacts;

        add_objects_from_periodicity(box, particles, this->m_params.dmax);

        // utilisation de kdtree pour ne rechercher les contacts que pour les particules proches
        tic();
        using my_kd_tree_t = typename nanoflann::
            KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, KdTree<dim>>, KdTree<dim>, dim, std::size_t>;
        KdTree<dim> kd(particles, active_ptr);
        my_kd_tree_t index(dim, kd, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
        auto duration = toc();
        PLOG_INFO << "----> CPUTIME : build kdtree index = " << duration << std::endl;

        tic();

        m_nMatches = 0;
#pragma omp parallel for reduction(+ : m_nMatches) // num_threads(1)

        for (std::size_t i = particles.offset(active_ptr); i < particles.pos().size() - 1; ++i)
        {
            double query_pt[dim];
            for (std::size_t d = 0; d < dim; ++d)
            {
                query_pt[d] = particles.pos()(i)(d);
            }
            PLOG_DEBUG << "i = " << i << " query_pt = " << query_pt[0] << " " << query_pt[1] << std::endl;

            std::vector<nanoflann::ResultItem<std::size_t, double>> ret_matches;

            auto nMatches_loc = index.radiusSearch(query_pt, this->m_params.kd_tree_radius, ret_matches, nanoflann::SearchParameters());

            for (std::size_t ic = 0; ic < nMatches_loc; ++ic)
            {
                std::size_t j = ret_matches[ic].first + particles.offset(active_ptr);
                if (i < j)
                {
                    compute_exact_distance<problem_t>(box, particles, contacts, this->m_params.dmax, i, j, m_default_contact_property);
                    m_nMatches++;
                }
            }
        }

        // obstacles
        for (std::size_t i = 0; i < active_ptr; ++i)
        {
            for (std::size_t j = active_ptr; j < particles.pos().size(); ++j)
            {
                compute_exact_distance<problem_t>(box, particles, contacts, this->m_params.dmax, i, j, m_default_contact_property);
            }
        }

        duration = toc();
        PLOG_INFO << "----> CPUTIME : compute " << contacts.size() << " contacts = " << duration << " compute " << m_nMatches
                  << " distances" << std::endl;

        tic();
        sort_contacts(contacts);
        duration = toc();
        PLOG_INFO << "----> CPUTIME : sort " << contacts.size() << " contacts = " << duration << std::endl;

        particles.reset_periodic();

        return contacts;
    }
}

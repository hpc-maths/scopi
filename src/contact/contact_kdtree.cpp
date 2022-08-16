#include "scopi/contact/contact_kdtree.hpp"

namespace scopi
{
    contact_kdtree::contact_kdtree(const ContactsParams<contact_kdtree>& params)
    : contact_base()
    , m_params(params)
    {};

    std::size_t contact_kdtree::get_nMatches() const
    {
        return m_nMatches;
    }

    ContactsParams<contact_kdtree>::ContactsParams()
    : ContactsParamsBase()
    , kd_tree_radius(17.)
    {}

    ContactsParams<contact_kdtree>::ContactsParams(const ContactsParams<contact_kdtree>& params)
    : ContactsParamsBase(params)
    , kd_tree_radius(params.kd_tree_radius)
    {}

}

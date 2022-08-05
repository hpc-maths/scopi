#include "scopi/contact/contact_kdtree.hpp"

namespace scopi
{
    contact_kdtree::contact_kdtree(double dmax, const ContactsParams<contact_kdtree>& params)
    : contact_base(dmax)
    , m_params(params)
    , m_kd_tree_radius(17.)
    {};

    std::size_t contact_kdtree::get_nMatches() const
    {
        return m_nMatches;
    }

    ContactsParams<contact_kdtree>::ContactsParams()
    : ContactsParamsBase()
    {}

    ContactsParams<contact_kdtree>::ContactsParams(const ContactsParams<contact_kdtree>& params)
    : ContactsParamsBase(params)
    {}

}

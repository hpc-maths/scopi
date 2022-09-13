#include "scopi/contact/contact_brute_force.hpp"

namespace scopi
{
    contact_brute_force::contact_brute_force(const ContactsParams<contact_brute_force>& params)
    : contact_base()
    , m_params(params)
    {}

    ContactsParams<contact_brute_force>::ContactsParams()
    : dmax(2.)
    {}

    ContactsParams<contact_brute_force>::ContactsParams(const ContactsParams<contact_brute_force>& params)
    : dmax(params.dmax)
    {}

}

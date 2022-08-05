#include "scopi/contact/contact_brute_force.hpp"

namespace scopi
{
    contact_brute_force::contact_brute_force(double dmax, const ContactsParams<contact_brute_force>& params)
    : contact_base(dmax)
    , m_params(params)
    {};

    ContactsParams<contact_brute_force>::ContactsParams()
    : ContactsParamsBase()
    {}

    ContactsParams<contact_brute_force>::ContactsParams(const ContactsParams<contact_brute_force>& params)
    : ContactsParamsBase(params)
    {}

}

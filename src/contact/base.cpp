#include "scopi/contact/base.hpp"

namespace scopi
{
    ContactsParamsBase::ContactsParamsBase()
    : dmax(2.)
    {}

    ContactsParamsBase::ContactsParamsBase(const ContactsParamsBase& params)
    : dmax(params.dmax)
    {}

}

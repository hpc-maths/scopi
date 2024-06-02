#include "scopi/contact/contact_brute_force.hpp"

namespace scopi
{
    contact_brute_force::contact_brute_force(const ContactsParams<contact_brute_force>& params)
        : contact_base(params)
    {
    }

    ContactsParams<contact_brute_force>::ContactsParams()
        : dmax(2.)
    {
    }

    void ContactsParams<contact_brute_force>::init_options()
    {
        auto& app = get_app();
        auto opt  = app.add_option_group("Brute force contact options");
        opt->add_option("--dmax", dmax, "Maximum distance between two neighboring particles")->capture_default_str();
    }
}

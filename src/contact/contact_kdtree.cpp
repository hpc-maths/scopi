#include "scopi/contact/contact_kdtree.hpp"
#include "scopi/scopi.hpp"

#include <memory>

namespace scopi
{
    contact_kdtree::contact_kdtree(const ContactsParams<contact_kdtree>& params)
        : contact_base(params)
    {
    }

    ContactsParams<contact_kdtree>::ContactsParams()
        : dmax(2.)
        , kd_tree_radius(17.)
    {
    }

    void ContactsParams<contact_kdtree>::init_options()
    {
        auto& app = get_app();
        auto opt  = app.add_option_group("KD tree options");
        opt->add_option("--dmax", dmax, "Maximum distance between two neighboring particles")->capture_default_str();
        opt->add_option("--kd-radius", kd_tree_radius, "Kd-tree radius")->capture_default_str();
    }
}

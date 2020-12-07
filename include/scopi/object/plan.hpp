#pragma once

#include "base.hpp"

namespace scopi
{
    ///////////////////////
    // plan definition //
    ///////////////////////
    template<std::size_t dim, bool owner=true>
    class plan: public object<dim, owner>
    {
    public:

        using base_type = object<dim, owner>;
        using position_type = typename base_type::position_type;

        plan(position_type pos);

        virtual std::unique_ptr<base_constructor<dim>> construct() const override;
        virtual void print() const override;
        virtual std::size_t hash() const override;

        XTL_IMPLEMENT_INDEXABLE_CLASS()

    private:

        void create_hash();

        std::size_t m_hash;
    };

    ///////////////////////////
    // plan implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    plan<dim, owner>::plan(position_type pos)
    : base_type(pos, 1)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    std::unique_ptr<base_constructor<dim>> plan<dim, owner>::construct() const
    {
        return make_object_constructor<plan<dim, false>>();
    }

    template<std::size_t dim, bool owner>
    void plan<dim, owner>::print() const
    {
        std::cout << "plan<" << dim << ">\n";
    }

    template<std::size_t dim, bool owner>
    std::size_t plan<dim, owner>::hash() const
    {
        return m_hash;
    }

    template<std::size_t dim, bool owner>
    void plan<dim, owner>::create_hash()
    {
        std::stringstream ss;
        ss << "plan<" << dim << ")";
        m_hash = std::hash<std::string>{}(ss.str());
    }
}

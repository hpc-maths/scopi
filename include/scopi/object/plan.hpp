#pragma once

#include <array>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

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
        using rotation_type = typename base_type::rotation_type;

        plan(position_type pos);
        plan(position_type pos, rotation_type r);

        auto normal() const;

        virtual std::unique_ptr<base_constructor<dim>> construct() const override;
        virtual void print() const override;
        virtual std::size_t hash() const override;

    private:

        void create_hash();

        std::size_t m_hash;
    };

    ///////////////////////////
    // plan implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    plan<dim, owner>::plan(position_type pos)
    : base_type(pos, {{ { {1, 0}, {0, 1} } }}, 1)
    {
        std::size_t size = 1;
        create_hash();
    }

    template<std::size_t dim, bool owner>
    auto plan<dim, owner>::normal() const
    {// nref = (1,0,0)
        return xt::eval(xt::view(this->R(), xt::all(), 0));
    }

    template<std::size_t dim, bool owner>
    plan<dim, owner>::plan(position_type pos, rotation_type r)
    : base_type(pos, r, 1)
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
        std::cout << "plan<" << dim << ">()\n";
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
        ss << "plan<" << dim << ">()";
        m_hash = std::hash<std::string>{}(ss.str());
    }
}

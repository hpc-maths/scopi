#pragma once

#include "base.hpp"

namespace scopi
{
    ///////////////////////
    // sphere definition //
    ///////////////////////
    template<class position_type, std::size_t dim>
    position_type sphere_construction(const std::array<double, dim>& pos)
    {
        position_type p;
        p.push_back(pos);
        return p;
    }
    
    template<std::size_t dim, bool owner=true>
    class sphere: public object<dim, owner>
    {
    public:

        using base_type = object<dim, owner>;
        using position_type = typename base_type::position_type;

        sphere(const std::array<double, dim>& pos, double radius);
        sphere(std::array<double, dim>* pos, double radius);

        virtual std::shared_ptr<base_constructor<dim>> construct() const override;
        virtual void print() const override;
        virtual std::size_t hash() const override;

    private:

        void create_hash();

        double m_radius;
        std::size_t m_hash;
    };

    ///////////////////////////
    // sphere implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    sphere<dim, owner>::sphere(const std::array<double, dim>& pos, double radius)
    : base_type(sphere_construction<position_type, dim>(pos)), m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    sphere<dim, owner>::sphere(std::array<double, dim>* pos, double radius)
    : base_type(pos, 1), m_radius(radius)
    {
    }

    template<std::size_t dim, bool owner>
    std::shared_ptr<base_constructor<dim>> sphere<dim, owner>::construct() const
    {
        return make_object_constructor<dim, sphere<dim, false>>(m_radius);
    }

    template<std::size_t dim, bool owner>
    void sphere<dim, owner>::print() const
    {
        std::cout << "sphere<" << dim << ">(" << m_radius << ")\n";
    }

    template<std::size_t dim, bool owner>
    std::size_t sphere<dim, owner>::hash() const
    {
        return m_hash;
    }

    template<std::size_t dim, bool owner>
    void sphere<dim, owner>::create_hash()
    {
        std::stringstream ss;
        ss << "sphere<" << dim << ">(" << m_radius << ")";
        m_hash = std::hash<std::string>{}(ss.str());
    }
}
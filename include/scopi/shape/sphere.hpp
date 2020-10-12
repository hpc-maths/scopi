#pragma once

#include "base.hpp"

namespace scopi
{
    ///////////////////////
    // sphere definition //
    ///////////////////////
    template<std::size_t dim, bool owner=true>
    class sphere: public shape<dim, owner>
    {
    public:

        using base_type = shape<dim, owner>;

        template<class T>
        sphere(T&& pos, double radius);

        sphere(std::array<double, dim>& pos, double radius);
        sphere(std::array<double, dim>&& pos, double radius);

        virtual std::function<shape<dim, false>*(std::array<double, dim>*)> construct() const override;
        virtual void print() override;
        virtual std::size_t hash() override;

    private:

        void create_hash();

        double m_radius;
        std::size_t m_hash;
    };

    ///////////////////////////
    // sphere implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    template<class T>
    sphere<dim, owner>::sphere(T&& pos, double radius)
    : base_type(pos), m_radius{radius}
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    sphere<dim, owner>::sphere(std::array<double, dim>& pos, double radius)
    : base_type(pos), m_radius{radius}
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    sphere<dim, owner>::sphere(std::array<double, dim>&& pos, double radius)
    : base_type(pos), m_radius{radius}
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    std::function<shape<dim, false>*(std::array<double, dim>*)> sphere<dim, owner>::construct() const
    {
        return make_shape_constructor<sphere<dim, false>>(m_radius)();
    }

    template<std::size_t dim, bool owner>
    void sphere<dim, owner>::print()
    {
        std::cout << "sphere\n";
    }

    template<std::size_t dim, bool owner>
    std::size_t sphere<dim, owner>::hash()
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
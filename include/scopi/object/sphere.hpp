#pragma once

#include "base.hpp"
#include "../quaternion.hpp"

namespace scopi
{
    ///////////////////////
    // sphere definition //
    ///////////////////////
    template<std::size_t dim, bool owner=true>
    class sphere: public object<dim, owner>
    {
    public:

        using base_type = object<dim, owner>;
        using position_type = typename base_type::position_type;
        using quaternion_type = typename base_type::quaternion_type;

        sphere(position_type pos, double radius);
        sphere(position_type pos, quaternion_type q, double radius);

        // sphere(const sphere&) = default;
        // sphere& operator=(const sphere&) = default;

        double radius() const;
        virtual std::unique_ptr<base_constructor<dim>> construct() const override;
        virtual void print() const override;
        virtual std::size_t hash() const override;
        auto rotation() const;

    private:

        void create_hash();

        double m_radius;
        std::size_t m_hash;
    };

    ///////////////////////////
    // sphere implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    sphere<dim, owner>::sphere(position_type pos, double radius)
    : base_type(pos, {quaternion()}, 1)
    , m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    sphere<dim, owner>::sphere(position_type pos, quaternion_type q, double radius)
    : base_type(pos, q, 1)
    , m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    std::unique_ptr<base_constructor<dim>> sphere<dim, owner>::construct() const
    {
        return make_object_constructor<sphere<dim, false>>(m_radius);
    }

    template<std::size_t dim, bool owner>
    double sphere<dim, owner>::radius() const
    {
        return m_radius;
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

    template<std::size_t dim, bool owner>
    auto sphere<dim, owner>::rotation() const
    {
        return rotation_matrix<dim>(this->q());
    }
}

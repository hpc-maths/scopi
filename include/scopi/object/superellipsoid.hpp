#pragma once

#include "base.hpp"
#include "../quaternion.hpp"
#include "../types.hpp"

namespace scopi
{
    ///////////////////////
    // superellipsoid definition //
    ///////////////////////
    template<std::size_t dim, bool owner=true>
    class superellipsoid: public object<dim, owner>
    {
    public:

        using base_type = object<dim, owner>;
        using position_type = typename base_type::position_type;
        using quaternion_type = typename base_type::quaternion_type;

        superellipsoid(position_type pos, type::position<dim> radius, type::position<dim-1>  squareness);
        superellipsoid(position_type pos, quaternion_type q, type::position<dim> radius, type::position<dim-1>  squareness);

        // superellipsoid(const superellipsoid&) = default;
        // superellipsoid& operator=(const superellipsoid&) = default;

        const auto radius() const;
        virtual std::unique_ptr<base_constructor<dim>> construct() const override;
        virtual void print() const override;
        virtual std::size_t hash() const override;
        auto rotation() const;

    private:

        void create_hash();

        type::position<dim>  m_radius;
        type::position<dim-1>  m_squareness;
        std::size_t m_hash;
    };

    ///////////////////////////////////
    // superellipsoid implementation //
    ///////////////////////////////////
    template<std::size_t dim, bool owner>
    superellipsoid<dim, owner>::superellipsoid(position_type pos, type::position<dim> radius, type::position<dim-1>  squareness)
    : base_type(pos, {quaternion()}, 1)
    , m_radius(radius)
    , m_squareness(squareness)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    superellipsoid<dim, owner>::superellipsoid(position_type pos, quaternion_type q, type::position<dim> radius, type::position<dim-1>  squareness)
    : base_type(pos, q, 1)
    , m_radius(radius)
    , m_squareness(squareness)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    std::unique_ptr<base_constructor<dim>> superellipsoid<dim, owner>::construct() const
    {
        return make_object_constructor<superellipsoid<dim, false>>(m_radius, m_squareness);
    }

    template<std::size_t dim, bool owner>
    const auto superellipsoid<dim, owner>::radius() const
    {
        return m_radius;
    }

    template<std::size_t dim, bool owner>
    void superellipsoid<dim, owner>::print() const
    {
        std::cout << "superellipsoid<" << dim << "> : radius = " << m_radius << " squareness = " << m_squareness << "\n";
    }

    template<std::size_t dim, bool owner>
    std::size_t superellipsoid<dim, owner>::hash() const
    {
        return m_hash;
    }

    template<std::size_t dim, bool owner>
    void superellipsoid<dim, owner>::create_hash()
    {
        std::stringstream ss;
        ss << "superellipsoid<" << dim << "> : radius = " << m_radius << " squareness = " << m_squareness << "\n";
        m_hash = std::hash<std::string>{}(ss.str());
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::rotation() const
    {
        return rotation_matrix<dim>(this->q());
    }

}

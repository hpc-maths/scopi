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
        const auto squareness() const; // e, n
        virtual std::unique_ptr<base_constructor<dim>> construct() const override;
        virtual void print() const override;
        virtual std::size_t hash() const override;
        auto rotation() const;
        auto point(const double b) const; // dim = 2
        auto point(const double a, const double b) const; // dim = 3
        auto normal(const double b) const; // dim = 2
        auto normal(const double a, const double b) const; // dim = 3
        auto tangent(const double b) const; // dim = 2
        auto tangents(const double a, const double b) const; // dim = 3

    private:

        void create_hash();

        type::position<dim>  m_radius;
        type::position<dim-1>  m_squareness; // e, n
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
    const auto superellipsoid<dim, owner>::squareness() const
    {
        return m_squareness;
    }

    template<std::size_t dim, bool owner>
    void superellipsoid<dim, owner>::print() const
    {
        std::cout << "superellipsoid<" << dim << "> : radius = " << m_radius << " squareness = " << m_squareness << "\n"; //<< " q = "<< xt::view(this->q(0),xt::all()) << "\n";
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
        ss << "superellipsoid<" << dim << "> : radius = " << m_radius << " squareness = " << m_squareness << "\n"; //<< " q = "<< xt::view(this->q(0),xt::all()) << "\n";
        m_hash = std::hash<std::string>{}(ss.str());
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::rotation() const
    {
        return rotation_matrix<dim>(this->q());
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::point(const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> pt;
        pt(0) = m_radius(0) * sign(std::cos(b)) * std::pow(std::abs(std::cos(b)), m_squareness(0));
        pt(1) = m_radius(1) * sign(std::sin(b)) * std::pow(std::abs(std::sin(b)), m_squareness(0));
        return xt::flatten(xt::eval(xt::linalg::dot(rotation_matrix<dim>(this->q()),pt) + this->pos()));
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::point(const double a, const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> pt;
        pt(0) = m_radius(0) * sign(std::cos(a)) * std::pow(std::abs(std::cos(a)), m_squareness(1)) * sign(std::cos(b)) * std::pow(std::abs(std::cos(b)), m_squareness(0));
        pt(1) = m_radius(1) * sign(std::cos(a)) * std::pow(std::abs(std::cos(a)), m_squareness(1)) * sign(std::sin(b)) * std::pow(std::abs(std::sin(b)), m_squareness(0));
        pt(2) = m_radius(2) * sign(std::sin(a)) * std::pow(std::abs(std::sin(a)), m_squareness(1));
        return xt::flatten(xt::eval(xt::linalg::dot(rotation_matrix<dim>(this->q()),pt) + this->pos()));
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::normal(const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> n;
        n(0) = m_radius(1) * sign(std::cos(b)) * std::pow(std::abs(std::cos(b)), 2-m_squareness(0));
        n(1) = m_radius(0) * sign(std::sin(b)) * std::pow(std::abs(std::sin(b)), 2-m_squareness(0));
        n = xt::flatten(xt::linalg::dot(rotation_matrix<dim>(this->q()),n));
        n /= xt::linalg::norm(n, 2);
        return n;
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::normal(const double a, const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> n;
        n(0) =  m_radius(1) * m_radius(2) * std::pow(std::abs(std::cos(a)), 2-m_squareness(1)) * sign(std::cos(b)) * std::pow(std::abs(std::cos(b)), 2-m_squareness(0));
        n(1) =  m_radius(0) * m_radius(2) * std::pow(std::abs(std::cos(a)), 2-m_squareness(1)) * sign(std::sin(b)) * std::pow(std::abs(std::sin(b)), 2-m_squareness(0));
        n(2) =  m_radius(0) * m_radius(1) * sign(std::cos(a)) * sign(std::sin(a)) * std::pow(std::abs(std::sin(a)), 2-m_squareness(1));
        n = xt::flatten(xt::linalg::dot(rotation_matrix<dim>(this->q()),n));
        n /= xt::linalg::norm(n, 2);
        return n;
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::tangent(const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> tgt;
        tgt(0) = -m_radius(0) * sign(std::sin(b)) * std::pow(std::abs(std::sin(b)), 2-m_squareness(0));
        tgt(1) =  m_radius(1) * sign(std::cos(b)) * std::pow(std::abs(std::cos(b)), 2-m_squareness(0));
        tgt = xt::flatten(xt::linalg::dot(rotation_matrix<dim>(this->q()),tgt));
        tgt /= xt::linalg::norm(tgt, 2);
        return tgt;
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::tangents(const double a, const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> tgt1;
        tgt1(0) = -m_radius(0) * sign(std::cos(a)) * sign(std::sin(b)) * std::pow(std::abs(std::sin(b)), 2-m_squareness(0));
        tgt1(1) =  m_radius(1) * sign(std::cos(a)) * sign(std::cos(b)) * std::pow(std::abs(std::cos(b)), 2-m_squareness(0));
        tgt1(2) =  0;
        tgt1 = xt::flatten(xt::linalg::dot(rotation_matrix<dim>(this->q()),tgt1));
        tgt1 /= xt::linalg::norm(tgt1, 2);
        xt::xtensor_fixed<double, xt::xshape<dim>> tgt2;
        tgt2(0) = -m_radius(0) * sign(std::sin(a)) * std::pow(std::abs(std::sin(a)), 2-m_squareness(1)) * sign(std::cos(b)) * std::pow(std::abs(std::cos(b)), m_squareness(0));
        tgt2(1) = -m_radius(1) * sign(std::sin(a)) * std::pow(std::abs(std::sin(a)), 2-m_squareness(1)) * sign(std::sin(b)) * std::pow(std::abs(std::sin(b)), m_squareness(0));
        tgt2(2) =  m_radius(2) * sign(std::cos(a)) * std::pow(std::abs(std::cos(a)), 2-m_squareness(1));
        tgt2 = xt::flatten(xt::linalg::dot(rotation_matrix<dim>(this->q()),tgt2));
        tgt2 /= xt::linalg::norm(tgt2, 2);
        return std::make_pair(tgt1,tgt2);
    }

}

#pragma once

#include <array>
#include <vector>

namespace scopi
{
    ////////////////////////
    // globule definition //
    ////////////////////////
    template<class position_type, std::size_t dim>
    position_type globule_construction(const std::array<double, dim>& pos)
    {
        position_type p(6);
        for(std::size_t i = 0; i < 6; ++i)
        {
            p[i] = pos;
        }
        return p;
    }

    template<std::size_t dim, bool owner=true>
    class globule: public object<dim, owner>
    {
    public:

        using base_type = object<dim, owner>;
        using position_type = typename base_type::position_type;

        globule(position_type pos, double radius);

        virtual std::unique_ptr<base_constructor<dim>> construct() const override;
        virtual void print() const override;
        virtual std::size_t hash() const override;

        XTL_IMPLEMENT_INDEXABLE_CLASS()

    private:

        void create_hash();

        double m_radius;
        std::size_t m_hash;
    };

    ////////////////////////////
    // globule implementation //
    ////////////////////////////
    template<std::size_t dim, bool owner>
    globule<dim, owner>::globule(position_type pos, double radius)
    : base_type(pos, 6)
    , m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    std::unique_ptr<base_constructor<dim>> globule<dim, owner>::construct() const
    {
        return make_object_constructor<globule<dim, false>>(m_radius);
    }

    template<std::size_t dim, bool owner>
    void globule<dim, owner>::print() const
    {
        std::cout << "globule<" << dim << ">(" << m_radius << ")\n";
    }

    template<std::size_t dim, bool owner>
    std::size_t globule<dim, owner>::hash() const
    {
        return m_hash;
    }

    template<std::size_t dim, bool owner>
    void globule<dim, owner>::create_hash()
    {
        std::stringstream ss;
        ss << "globule<" << dim << ">(" << m_radius << ")";
        m_hash = std::hash<std::string>{}(ss.str());
    }
}
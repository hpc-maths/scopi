#pragma once

#include "base.hpp"

namespace scopi
{
    ///////////////////////
    // sphere definition //
    ///////////////////////
    template<class position_type, std::size_t dim>
    position_type sphere_position_construction(const std::array<double, dim>& pos)
    {
        position_type p;
        p.push_back(pos);
        return p;
    }
    template<class velocity_type, std::size_t dim>
    velocity_type sphere_velocity_construction(const std::array<double, dim>& v)
    {
        velocity_type p;
        p.push_back(v);
        return p;
    }
    template<class desired_velocity_type, std::size_t dim>
    desired_velocity_type sphere_desired_velocity_construction(const std::array<double, dim>& vd)
    {
        desired_velocity_type p;
        p.push_back(vd);
        return p;
    }
    template<class force_type, std::size_t dim>
    force_type sphere_force_construction(const std::array<double, dim>& f)
    {
        force_type p;
        p.push_back(f);
        return p;
    }

    template<std::size_t dim, bool owner=true>
    class sphere: public object<dim, owner>
    {
    public:

        using base_type = object<dim, owner>;
        using position_type = typename base_type::position_type;
        using velocity_type = typename base_type::velocity_type;
        using desired_velocity_type = typename base_type::desired_velocity_type;
        using force_type = typename base_type::force_type;

        sphere(
          const std::array<double, dim>& pos,
          const std::array<double, dim>& v,
          const std::array<double, dim>& vd,
          const std::array<double, dim>& f,
          double radius
        );
        sphere(
          std::array<double, dim>* pos,
          std::array<double, dim>* v,
          std::array<double, dim>* vd,
          std::array<double, dim>* f,
          double radius
        );

        double radius() const;
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
    sphere<dim, owner>::sphere(
      const std::array<double, dim>& pos,
      const std::array<double, dim>& v,
      const std::array<double, dim>& vd,
      const std::array<double, dim>& f,
      double radius
    )
    : base_type(
      sphere_position_construction<position_type, dim>(pos),
      sphere_velocity_construction<velocity_type, dim>(v),
      sphere_desired_velocity_construction<desired_velocity_type, dim>(vd),
      sphere_force_construction<force_type, dim>(f)
    ), m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    sphere<dim, owner>::sphere(
      std::array<double, dim>* pos,
      std::array<double, dim>* v,
      std::array<double, dim>* vd,
      std::array<double, dim>* f,
      double radius
    )
    : base_type(pos, v, vd, f, 1), m_radius(radius)
    {
    }

    template<std::size_t dim, bool owner>
    std::shared_ptr<base_constructor<dim>> sphere<dim, owner>::construct() const
    {
        return make_object_constructor<dim, sphere<dim, false>>(m_radius);
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
}

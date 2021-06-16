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
        std::vector<double> binit_xy(const int n) const; // dim = 2 et 3
        std::vector<double> ainit_yz(const int n) const; // dim  3
        std::vector<double> ainit_xz(const int n) const; // dim  3
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
      if (dim==2){
        std::cout << "superellipsoid 2D : radius = " << m_radius << " squareness e =" << m_squareness(0) << " q = "<< this->q() << "\n";
      }
      else {
        std::cout << "superellipsoid 3D : radius = " << m_radius << " squareness e =" << m_squareness(0) << " n =" << m_squareness(1) << " q = "<< this->q() << "\n";
      }
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
        ss << "superellipsoid<" << dim << "> : radius = " << m_radius << " squareness = " << m_squareness << " q = "<< this->q() << "\n"; // ?????
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
        n(0) =  m_radius(1) * m_radius(2) * sign(std::cos(a)) * std::pow(std::abs(std::cos(a)), 2-m_squareness(1)) * sign(std::cos(b)) * std::pow(std::abs(std::cos(b)), 2-m_squareness(0));
        n(1) =  m_radius(0) * m_radius(2) * sign(std::cos(a)) * std::pow(std::abs(std::cos(a)), 2-m_squareness(1)) * sign(std::sin(b)) * std::pow(std::abs(std::sin(b)), 2-m_squareness(0));
        n(2) =  m_radius(0) * m_radius(1) * sign(std::sin(a)) * std::pow(std::abs(std::sin(a)), 2-m_squareness(1));
        n = xt::flatten(xt::linalg::dot(rotation_matrix<dim>(this->q()),n));
        n /= xt::linalg::norm(n, 2);
        return n;
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::tangent(const double b) const  //2D
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> tgt;
        tgt(0) = -m_radius(0) * sign(std::sin(b)) * std::pow(std::abs(std::sin(b)), 2-m_squareness(0));
        tgt(1) =  m_radius(1) * sign(std::cos(b)) * std::pow(std::abs(std::cos(b)), 2-m_squareness(0));
        tgt = xt::flatten(xt::linalg::dot(rotation_matrix<dim>(this->q()),tgt));
        tgt /= xt::linalg::norm(tgt, 2);
        return tgt;
    }

    template<std::size_t dim, bool owner>
    auto superellipsoid<dim, owner>::tangents(const double a, const double b) const //3D
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

    // return a regular angle b distribution, used to initialize newton method
    template<std::size_t dim, bool owner>
    std::vector<double> superellipsoid<dim, owner>::binit_xy(const int n) const
    {
      const double pi = 4*std::atan(1);
      std::vector<double> bs;
      // double epsilon = 0.01;
      // auto angles = xt::linspace<double>(epsilon, 0.5*pi-epsilon, n, true);
      auto angles = xt::linspace<double>(0, 0.5*pi, n, true);
      for (std::size_t i = 0; i < angles.size(); i++) {
        double b_ell = angles(i);
        // std::cout << "\nb_ell= " << b_ell << std::endl;
        double x_ell =  m_radius(0)*std::cos(b_ell);
        double y_ell =  m_radius(1)*std::sin(b_ell);
        // std::cout << "x_ell= " << x_ell << " y_ell= " << y_ell<< std::endl;
        // e = m_squareness(0)
        double d = std::pow( std::pow(m_radius(1)*x_ell,2.0/m_squareness(0))+std::pow(m_radius(0)*y_ell,2.0/m_squareness(0)), m_squareness(0)/2.0);
        // std::cout << "std::pow(m_radius(1)*x_ell,2.0/m_squareness(dim-2))= " << std::pow(m_radius(1)*x_ell,2.0/m_squareness(dim-2)) << std::endl;
        // std::cout << "std::pow(m_radius(0)*y_ell,2.0/m_squareness(dim-2))= " << std::pow(m_radius(0)*y_ell,2.0/m_squareness(dim-2)) << std::endl;
        // std::cout << "m_radius(0)= " << m_radius(0) << std::endl;
        // std::cout << "m_radius(1)= " << m_radius(1) << std::endl;
        // std::cout << "m_squareness(0)= " << m_squareness(0) << std::endl;
        // std::cout << "m_squareness(1)= " << m_squareness(1) << std::endl;
        // std::cout << "d= " << d << std::endl;
        double x_supell = x_ell*m_radius(0)*m_radius(1)/d;
        double y_supell = y_ell*m_radius(0)*m_radius(1)/d;
        // std::cout << "x_supell= " << x_supell << " y_supell= " << y_supell<< std::endl;
        double sinb = std::max(-1.0, std::min( 1.0, std::sqrt( std::pow( std::pow(y_supell/m_radius(1),2), 1.0/m_squareness(0)) )) );
        // std::cout << "sinb= " << sinb << std::endl;
        double b = std::asin(sinb);
        // std::cout << "b= " << b << std::endl;
        if (b>pi) {
          b -= 2*pi;
        }
        if (b<-pi) {
          b += 2*pi;
        }
        bs.push_back(b);
        bs.push_back(b+pi/2);
        bs.push_back(b-pi);
        bs.push_back(b-pi/2);
      }
      for(std::size_t i = 0; i < bs.size(); ++i) {
        if (std::abs(bs[i])<1.0e-6) {
          bs[i] = 0.01;
        }
      }
      return bs;
    }
    // return a regular angle b distribution, used to initialize newton method
    template<std::size_t dim, bool owner>
    std::vector<double> superellipsoid<dim, owner>::ainit_yz(const int n) const
    {
      const double pi = 4*std::atan(1);
      std::vector<double> as;
      // double epsilon = 0.01;
      // auto angles = xt::linspace<double>(epsilon, 0.5*pi-epsilon, n, true);
      auto angles = xt::linspace<double>(0, 0.5*pi, n, true);
      for (std::size_t i = 0; i < angles.size(); i++) {
        double a_ell = angles(i);
        // std::cout << "\na_ell= " << a_ell << std::endl;
        double y_ell =  m_radius(1)*std::cos(a_ell);
        double z_ell =  m_radius(2)*std::sin(a_ell);
        // std::cout << "y_ell= " << y_ell << " z_ell= " << z_ell<< std::endl;
        // n = m_squareness(1)
        // double d = std::pow( std::pow(m_radius(2)*y_ell,2/m_squareness(0))+std::pow(m_radius(1)*z_ell,2/m_squareness(0)), m_squareness(0)/2);
        double d = std::pow( std::pow(m_radius(2)*y_ell,2/m_squareness(1))+std::pow(m_radius(1)*z_ell,2/m_squareness(1)), m_squareness(1)/2);
        // std::cout << "d= " << d << std::endl;
        double y_supell = y_ell*m_radius(1)*m_radius(2)/d;
        double z_supell = z_ell*m_radius(1)*m_radius(2)/d;
        // std::cout << "y_supell= " << y_supell << " z_supell= " << z_supell<< std::endl;
        // double sina = std::max(-1.0, std::min( 1.0, std::sqrt( std::pow( std::pow(z_supell/m_radius(2),2), 1/m_squareness(0)) )) );
        double sina = std::max(-1.0, std::min( 1.0, std::sqrt( std::pow( std::pow(z_supell/m_radius(2),2), 1/m_squareness(1)) )) );
        // std::cout << "sina= " << sina << std::endl;
        double a = std::asin(sina);
        // std::cout << "a= " << a << std::endl;
        if (a>pi) {
          a -= 2*pi;
        }
        if (a<-pi) {
          a += 2*pi;
        }
        as.push_back(a);
        as.push_back(a+pi/2);
        as.push_back(a-pi);
        as.push_back(a-pi/2);
        for(std::size_t i = 0; i < as.size(); ++i) {
          if (std::abs(as[i])<1.0e-6) {
            as[i] = 0.01;
          }
        }
      }
      return as;
    }
    // return a regular angle b distribution, used to initialize newton method
    template<std::size_t dim, bool owner>
    std::vector<double> superellipsoid<dim, owner>::ainit_xz(const int n) const
    {
      const double pi = 4*std::atan(1);
      std::vector<double> as;
      // double epsilon = 0.01;
      // auto angles = xt::linspace<double>(epsilon, 0.5*pi-epsilon, n, true);
      auto angles = xt::linspace<double>(0, 0.5*pi, n, true);
      for (std::size_t i = 0; i < angles.size(); i++) {
        double a_ell = angles(i);
        // std::cout << "a_ell= " << a_ell << std::endl;
        double x_ell =  m_radius(0)*std::cos(a_ell);
        double z_ell =  m_radius(2)*std::sin(a_ell);
        // std::cout << "x_ell= " << x_ell << " z_ell= " << z_ell<< std::endl;
        // n = m_squareness(1)
        // double d = std::pow( std::pow(m_radius(2)*x_ell,2/m_squareness(0))+std::pow(m_radius(0)*z_ell,2/m_squareness(0)), m_squareness(0)/2);
        double d = std::pow( std::pow(m_radius(2)*x_ell,2.0/m_squareness(1))+std::pow(m_radius(0)*z_ell,2.0/m_squareness(1)), m_squareness(1)/2.0);
        // std::cout << "d= " << d << std::endl;
        double x_supell = x_ell*m_radius(0)*m_radius(2)/d;
        double z_supell = z_ell*m_radius(0)*m_radius(2)/d;
        // std::cout << "y_supell= " << y_supell << " z_supell= " << z_supell<< std::endl;
        // double sina = std::max(-1.0, std::min( 1.0, std::sqrt( std::pow( std::pow(z_supell/m_radius(2),2), 1/m_squareness(0)) )) );
        double sina = std::max(-1.0, std::min( 1.0, std::sqrt( std::pow( std::pow(z_supell/m_radius(2),2), 1.0/m_squareness(1)) )) );
        // std::cout << "sina= " << sina << std::endl;
        double a = std::asin(sina);
        // std::cout << "a= " << a << std::endl;
        if (a>pi) {
          a -= 2*pi;
        }
        if (a<-pi) {
          a += 2*pi;
        }
        as.push_back(a);
        as.push_back(a+pi/2);
        as.push_back(a-pi);
        as.push_back(a-pi/2);
        for(std::size_t i = 0; i < as.size(); ++i) {
          if (std::abs(as[i])<1.0e-6) {
            as[i] = 0.01;
          }
        }
      }
      return as;
    }
}

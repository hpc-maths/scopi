#pragma once

#include <array>
#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "base.hpp"
#include "../../quaternion.hpp"

namespace scopi
{
    /////////////////////
    // plan definition //
    /////////////////////
    /**
     * @class plan
     * @brief Plane.
     *
     * @tparam dim Dimension (2 or 3).
     * @tparam owner
     */
    template<std::size_t dim, bool owner=true>
    class plan: public object<dim, owner>
    {
    public:

        /**
         * @brief Alias for the base class object.
         */
        using base_type = object<dim, owner>;
        /**
         * @brief Alias for position type.
         */
        using position_type = typename base_type::position_type;
        /**
         * @brief Alias for quaternion type.
         */
        using quaternion_type = typename base_type::quaternion_type;

        /**
         * @brief Constructor with given rotation (2D).
         *
         * @param pos [in] Position of a point in the plane.
         * @param angle [in] Angle of the rotation.
         */
        plan(position_type pos, double angle=0);
        /**
         * @brief Constructor with given rotation.
         *
         * @param pos [in] Position of a point in the plane.
         * @param q [in] Quaternion describing the rotation of the plane.
         */
        plan(position_type pos, quaternion_type q);

        /**
         * @brief Get the rotation matrix of the plane.
         */
        auto rotation() const;
        /**
         * @brief 
         *
         * \todo Write documentation.
         *
         * @param b
         *
         * @return 
         */
        auto point(const double b) const; // dim = 2
        /**
         * @brief 
         *
         * \todo Write documentation.
         *
         * @param a
         * @param b
         *
         * @return 
         */
        auto point(const double a, const double b) const; // dim = 3
        /**
         * @brief Outer normal to the plane.
         */
        auto normal() const;
        /**
         * @brief 
         *
         * \todo Write documentation.
         *
         * @return 
         */
        virtual std::unique_ptr<base_constructor<dim>> construct() const override;
        /**
         * @brief Print the elements of the plane on standard output.
         */
        virtual void print() const override;
        /**
         * @brief Get the hash of the plane.
         */
        virtual std::size_t hash() const override;

    private:

        /**
         * @brief Create the hash of the plane.
         *
         * Two planes with the same dimension have the same hash. 
         */
        void create_hash();

        /**
         * @brief Hash of the plane.
         */
        std::size_t m_hash;
    };

    /////////////////////////
    // plan implementation //
    /////////////////////////
    template<std::size_t dim, bool owner>
    plan<dim, owner>::plan(position_type pos, double angle)
    : base_type(pos, {quaternion(angle)}, 1)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    auto plan<dim, owner>::rotation() const
    {
        return rotation_matrix<dim>(this->q());
    }

    template<std::size_t dim, bool owner>
    plan<dim, owner>::plan(position_type pos, quaternion_type q)
    : base_type(pos, q, 1)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    auto plan<dim, owner>::normal() const
    {
        auto rotation = rotation_matrix<dim>(this->q(0));
        return xt::eval(xt::view(rotation, xt::all(), 0));
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

    template<std::size_t dim, bool owner>
    auto plan<dim, owner>::point(const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> pt;
        pt(0) = 0;
        pt(1) = b;
        return xt::flatten(xt::eval(xt::linalg::dot(rotation_matrix<dim>(this->q()),pt) + this->pos()));
    }

    template<std::size_t dim, bool owner>
    auto plan<dim, owner>::point(const double a, const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> pt;
        pt(0) = 0;
        pt(1) = a;
        pt(2) = b;
        return xt::flatten(xt::eval(xt::linalg::dot(rotation_matrix<dim>(this->q()),pt) + this->pos()));
    }
}

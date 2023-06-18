#pragma once

#include <array>

#include <fmt/format.h>

#include <xtensor/xadapt.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

#include "base.hpp"
#include "../../quaternion.hpp"

namespace scopi
{
    ////////////////////////
    // segment definition //
    ////////////////////////
    /**
     * @class segment
     * @brief segment.
     *
     * @tparam dim Dimension (2 or 3).
     * @tparam owner
     */
    template<std::size_t dim, bool owner=true>
    class segment: public object<dim, owner>
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
         * @param pos1 [in] Position of first point  the segment.
         * @param pos2 [in] Position of second point of the segment.
         */
        segment(const type::position_t<dim>& pos1, const type::position_t<dim>& pos2);
        /**
         * @brief Constructor with given rotation.
         *
         * @param pos [in] Position of a point in the segment.
         * @param q [in] Quaternion describing the rotation of the segment.
         */
        segment(position_type pos, quaternion_type q, double length);

        /**
         * @brief Get the rotation matrix of the segment.
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
         * @brief Outer normal to the segment.
         */
        auto normal() const;
        auto tangent() const;

        auto extrema() const;

        /**
         * @brief
         *
         * \todo Write documentation.
         *
         * @return
         */
        virtual std::unique_ptr<base_constructor<dim>> construct() const override;
        /**
         * @brief Print the elements of the segment on standard output.
         */
        virtual void print() const override;
        /**
         * @brief Get the hash of the segment.
         */
        virtual std::size_t hash() const override;

    private:

        /**
         * @brief Create the hash of the segment.
         *
         * Two segments with the same dimension have the same hash.
         */
        void create_hash();

        /**
         * @brief Hash of the segment.
         */
        std::size_t m_hash;
        double m_length;
    };

    /////////////////////////
    // segment implementation //
    /////////////////////////
    template<std::size_t dim, bool owner>
    segment<dim, owner>::segment(const type::position_t<dim>& pos1, const type::position_t<dim>& pos2)
    : base_type({0.5*(pos1 + pos2)}, {quaternion(std::atan2(pos1[0]-pos2[0], pos2[1]-pos1[1]))}, 1)
    , m_length(xt::linalg::norm(pos2 - pos1))
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    auto segment<dim, owner>::rotation() const
    {
        return rotation_matrix<dim>(this->q());
    }

    template<std::size_t dim, bool owner>
    segment<dim, owner>::segment(position_type pos, quaternion_type q, double length)
    : base_type(pos, q, 1)
    , m_length(length)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    auto segment<dim, owner>::normal() const
    {
        auto rotation = rotation_matrix<dim>(this->q(0));
        return xt::eval(xt::view(rotation, xt::all(), 0));
    }

    template<std::size_t dim, bool owner>
    auto segment<dim, owner>::tangent() const
    {
        auto n = normal();
        return xt::xtensor_fixed<double, xt::xshape<dim>>{-n[1], n[0]};
    }

    template<std::size_t dim, bool owner>
    auto segment<dim, owner>::extrema() const
    {
        std::array<xt::xtensor_fixed<double, xt::xshape<dim>>, 2> pts;
        pts[0] = xt::linalg::dot(rotation_matrix<dim>(this->q(0)), xt::xtensor_fixed<double, xt::xshape<dim>>{0, -0.5*m_length}) + this->pos(0);
        pts[1] = xt::linalg::dot(rotation_matrix<dim>(this->q(0)), xt::xtensor_fixed<double, xt::xshape<dim>>{0, 0.5*m_length}) + this->pos(0);
        return pts;
    }

    template<std::size_t dim, bool owner>
    std::unique_ptr<base_constructor<dim>> segment<dim, owner>::construct() const
    {
        return make_object_constructor<segment<dim, false>>(m_length);
    }

    template<std::size_t dim, bool owner>
    void segment<dim, owner>::print() const
    {
        std::cout << fmt::format("segment<{}, {}>", dim, m_length) << std::endl;
    }

    template<std::size_t dim, bool owner>
    std::size_t segment<dim, owner>::hash() const
    {
        return m_hash;
    }

    template<std::size_t dim, bool owner>
    void segment<dim, owner>::create_hash()
    {
        m_hash = std::hash<std::string>{}(fmt::format("segment<{}, {}>()", dim, m_length));
    }

    template<std::size_t dim, bool owner>
    auto segment<dim, owner>::point(const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> pt;
        pt(0) = 0;
        pt(1) = b;
        return xt::flatten(xt::eval(xt::linalg::dot(rotation_matrix<dim>(this->q()),pt) + this->pos()));
    }

    template<std::size_t dim, bool owner>
    auto segment<dim, owner>::point(const double a, const double b) const
    {
        xt::xtensor_fixed<double, xt::xshape<dim>> pt;
        pt(0) = 0;
        pt(1) = a;
        pt(2) = b;
        return xt::flatten(xt::eval(xt::linalg::dot(rotation_matrix<dim>(this->q()),pt) + this->pos()));
    }
}

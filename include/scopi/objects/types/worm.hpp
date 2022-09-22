#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "base.hpp"
#include "../../quaternion.hpp"
#include "sphere.hpp"

namespace scopi
{
    /**
     * @brief Worm.
     *
     * A worm is a collection of spheres that are in contact (the distance between two successive spheres is 0 instead of being positive).
     * All spheres have the same radius.
     *
     * @tparam dim Dimension (2 or 3).
     * @tparam owner
     */
    template<std::size_t dim, bool owner=true>
    class worm: public object<dim, owner>
    {
    public:

        /**
         * @brief Alias for the base class \c object.
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
         * @brief Constructor with default rotation.
         *
         * Successive spheres should be in contact.
         *
         * @param pos [in] Array of positions of centers of the spheres.
         * @param radius [in] Radius of the spheres. All spheres have the same radius.
         * @param size [in] Number of spheres in the worm.
         */
        worm(position_type pos, double radius, std::size_t size);
        /**
         * @brief Constructor with given rotation for the spheres.
         *
         * Successive spheres should be in contact.
         *
         * @param pos [in] Array of positions of centers of the spheres.
         * @param q [in] Array of quaternions describing the rotation of the spheres.
         * @param radius [in] Radius of the spheres. All spheres have the same radius.
         * @param size [in] Number of spheres in the worm.
         */
        worm(position_type pos, quaternion_type q, double radius, std::size_t size);

        /**
         * @brief 
         *
         * TODO
         *
         * @return 
         */
        virtual std::unique_ptr<base_constructor<dim>> construct() const override;
        /**
         * @brief Print the elements of the worm on standard output.
         */
        virtual void print() const override;
        /**
         * @brief Get the hash of the worm.
         */
        virtual std::size_t hash() const override;

        /**
         * @brief Get the radius of the spheres in the worm.
         */
        double radius() const;
        /**
         * @brief Get a sphere in the worm.
         *
         * @param i [in] Index of the sphere in the worm. 0 <= i < this->size.
         *
         * @return Pointer to the i-th sphere in the worm.
         */
        std::unique_ptr<object<dim, false>> get_sphere(std::size_t i) const;

    private:

        /**
         * @brief Create the hash of the worm.
         *
         * Two worms with the same dimension, same radius and same number of spheres have the same hash. 
         */
        void create_hash();

        /**
         * @brief Radius of the spheres in the worm.
         */
        double m_radius;
        /**
         * @brief Hash of the worm.
         */
        std::size_t m_hash;
    };

    ////////////////////////////
    // worm implementation //
    ////////////////////////////
    template<std::size_t dim, bool owner>
    worm<dim, owner>::worm(position_type pos, double radius, std::size_t size)
    : base_type(pos, {quaternion()}, size)
    , m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    worm<dim, owner>::worm(position_type pos, quaternion_type q, double radius, std::size_t size)
    : base_type(pos, q, size)
    , m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    std::unique_ptr<base_constructor<dim>> worm<dim, owner>::construct() const
    {
        return make_object_constructor<worm<dim, false>>(m_radius, this->size());
    }

    template<std::size_t dim, bool owner>
    void worm<dim, owner>::print() const
    {
        std::cout << "worm<" << dim << ">(" << m_radius << ")\n";
    }

    template<std::size_t dim, bool owner>
    std::size_t worm<dim, owner>::hash() const
    {
        return m_hash;
    }

    template<std::size_t dim, bool owner>
    void worm<dim, owner>::create_hash()
    {
        std::stringstream ss;
        ss << "worm<" << dim << ">(" << m_radius << ", " << this->size() << ")";
        m_hash = std::hash<std::string>{}(ss.str());
    }

    template<std::size_t dim, bool owner>
    double worm<dim, owner>::radius() const
    {
        return m_radius;
    }

    template<std::size_t dim, bool owner>
    std::unique_ptr<object<dim, false>> worm<dim, owner>::get_sphere(std::size_t i) const
    {
        auto object = make_object_constructor<sphere<dim, false>>(m_radius);
        return (*object)(&this->internal_pos()[i], &this->internal_q()[i]);
    }

}

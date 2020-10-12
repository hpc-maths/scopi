#pragma once

#include <array>
#include <vector>

#include "base.hpp"

namespace scopi
{

    namespace detail
    {
        template<class CT, bool owner>
        struct inner_type
        {
            using position_type = typename CT::position_type;
            using force_type = typename CT::force_type;

            template<class T>
            T& get_pointer(T& t)
            {
                return t;
            }

            template<class T>
            const T& get_value(const T& t) const
            {
                return t;
            }

            template<class T>
            T& get_value(T& t) const
            {
                return t;
            }
        };

        template<class CT>
        struct inner_type<CT, false>
        {
            using position_type = typename CT::position_type*;
            using force_type = typename CT::force_type*;

            template<class T>
            T* get_pointer(T& t)
            {
                return &t;
            }

            template<class T>
            const T& get_value(const T* t) const
            {
                return *t;
            }

            template<class T>
            T& get_value(T* t) const
            {
                return *t;
            }
        };
    }

    template<std::size_t dim>
    struct shape_type
    {
        using position_type = std::array<double, dim>;
        using force_type = std::array<double, dim>;
    };

    template<std::size_t dim>
    struct object_type
    {
        using position_type = std::vector<std::array<double, dim>>;
        using force_type = std::vector<std::array<double, dim>>;
    };

    /////////////////////////////////
    // object_container definition //
    /////////////////////////////////
    template<class T, bool owner>
    class object_container: public detail::inner_type<T, owner>
    {
    public:

        using inner_types = detail::inner_type<T, owner>;
        using position_type = typename inner_types::position_type;
        using force_type = typename inner_types::force_type;

        object_container();
        object_container(typename std::remove_pointer<position_type>::type& pos);

        const position_type& pos() const;
        position_type& pos();

        const force_type& force() const;
        force_type& force();

    private:

        position_type m_pos;
        force_type m_force;
    };

    /////////////////////////////////////
    // object_container implementation //
    /////////////////////////////////////
    template<class T, bool owner>
    inline object_container<T, owner>::object_container()
    // : m_pos{nullptr}, m_force{nullptr}
    {}

    template<class T, bool owner>
    inline object_container<T, owner>::object_container(typename std::remove_pointer<position_type>::type& pos)
    : m_pos(this->get_pointer(pos))
    {}

    template<class T, bool owner>
    inline auto object_container<T, owner>::pos() const -> const position_type&
    {
        return this->get_value(m_pos);
    }

    template<class T, bool owner>
    inline auto object_container<T, owner>::pos() -> position_type&
    {
        return this->get_value(m_pos);
    }

    template<class T, bool owner>
    inline auto object_container<T, owner>::force() const -> const force_type&
    {
        return this->get_value(m_force);
    }

    template<class T, bool owner>
    inline auto object_container<T, owner>::force() -> force_type&
    {
        return this->get_value(m_force);
    }

    ///////////////////////
    // object definition //
    ///////////////////////
    template<std::size_t dim, bool owner=true>
    class object: public object_container<object_type<dim>, owner>
    {
    public:
        using base_type = object_container<object_type<dim>, owner>;
        using position_type = typename base_type::position_type;
        using force_type = typename base_type::force_type;

        object() = default;
        object(typename std::remove_pointer<position_type>::type& pos);
        object(typename std::remove_pointer<position_type>::type&& pos);

        virtual std::function<object<dim, false>*(std::vector<std::array<double, dim>>&)> construct() const = 0;
        virtual void print() = 0;
        virtual std::size_t hash() = 0;

    private:

        // std::vector<shape<dim, owner>> m_shapes;
    };

    ///////////////////////////
    // object implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    object<dim, owner>::object(typename std::remove_pointer<position_type>::type& pos)
    : base_type(pos)
    {
        // m_shapes.resize(pos.size());
        // std::copy(pos.cbegin(), pos.cend(), std::back_inserter(pos));
    }

    template<std::size_t dim, bool owner>
    object<dim, owner>::object(typename std::remove_pointer<position_type>::type&& pos)
    : base_type(pos)
    {
        // m_shapes.resize(pos.size());
        // std::copy(pos.cbegin(), pos.cend(), std::back_inserter(pos));
    }    

    ////////////////////////
    // globule definition //
    ////////////////////////

    template<class position_type, std::size_t dim>
    position_type globule_construction(std::array<double, dim>& pos)
    {
        position_type p(6);
        for(std::size_t i = 0; i<6; ++i)
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
        using force_type = typename base_type::force_type;

        globule(typename std::remove_pointer<position_type>::type& pos, double radius);

        globule(std::array<double, dim>& pos, double radius);
        globule(std::array<double, dim>&& pos, double radius);

        virtual std::function<object<dim, false>*(std::vector<std::array<double, dim>>&)> construct() const override;
        virtual void print() override;
        virtual std::size_t hash() override;

    private:

        void create_hash();

        double m_radius;
        std::size_t m_hash;
    };

    ////////////////////////////
    // globule implementation //
    ////////////////////////////
    template<std::size_t dim, bool owner>
    globule<dim, owner>::globule(typename std::remove_pointer<position_type>::type& pos, double radius)
    : base_type(pos), m_radius(radius)
    {}

    template<std::size_t dim, bool owner>
    globule<dim, owner>::globule(std::array<double, dim>& pos, double radius)
    : base_type(globule_construction<position_type, dim>(pos)), m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    globule<dim, owner>::globule(std::array<double, dim>&& pos, double radius)
    : base_type(globule_construction<position_type, dim>(pos)), m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    std::function<object<dim, false>*(std::vector<std::array<double, dim>>&)> globule<dim, owner>::construct() const
    {
        return make_shape_constructor<globule<dim, false>>(m_radius)();
    }

    template<std::size_t dim, bool owner>
    void globule<dim, owner>::print()
    {
        std::cout << "globule\n";
    }

    template<std::size_t dim, bool owner>
    std::size_t globule<dim, owner>::hash()
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
#pragma once

#include <array>
#include <vector>

#include "base.hpp"

namespace scopi
{

    namespace detail
    {
        // template<class T>
        // const auto get_value(std::size_t size, const T& t)
        // {
        //     return xt::adapt(reinterpret_cast<double*>(t.data()), {size, dim});
        // }

        // template<class T>
        // const auto get_value(std::size_t size, const T* t)
        // {
        //     return xt::adapt(reinterpret_cast<double*>(t.data()), {size, dim});
        // }

        // template<class T>
        // const T& get_value(const T* t)
        // {
        //     return *t
        // }

        template<std::size_t dim, bool owner>
        struct inner_type
        {
            using position_type = typename std::vector<std::array<double, dim>>;
            using ref_position_type = typename std::vector<std::array<double, dim>>&;
            using force_type = typename std::vector<std::array<double, dim>>;
            using ref_force_type = typename std::vector<std::array<double, dim>>&;

            template<class T>
            T& get_pointer(T& t)
            {
                return t;
            }

            // template<class T>
            // auto get_value(const T& t, std::size_t size) const
            // {
            //     return xt::adapt(reinterpret_cast<double*>(t.data()), {size, dim});
            // }

            template<class T>
            auto get_value(T& t, std::size_t size) const
            {
                return xt::adapt(reinterpret_cast<double*>(t.data()), {size, dim});
            }

            template<class T>
            const std::array<double, dim>& get_value(const T& t, std::size_t, std::size_t i) const
            {
                return t[i];
            }

            template<class T>
            std::array<double, dim>& get_value(T& t, std::size_t, std::size_t i) const
            {
                return t[i];
            }

        };

        template<std::size_t dim>
        struct inner_type<dim, false>
        {
            using position_type = typename std::array<double, dim>*;
            using ref_position_type = typename std::array<double, dim>&;
            using force_type = typename std::array<double, dim>*;
            using ref_force_type = typename std::array<double, dim>&;

            template<class T>
            T* get_pointer(T* t)
            {
                return t;
            }

            // template<class T>
            // auto get_value(const T* t, std::size_t size) const
            // {
            //     return xt::adapt(reinterpret_cast<double*>(t), {size, dim});
            // }

            template<class T>
            auto get_value(T* t, std::size_t size)
            {
                return xt::adapt(reinterpret_cast<double*>(t), {size, dim});
            }
        };
    }

    /////////////////////////////////
    // object_container definition //
    /////////////////////////////////
    template<std::size_t dim, bool owner>
    class object_container: public detail::inner_type<dim, owner>
    {
    public:

        using inner_types = detail::inner_type<dim, owner>;
        using position_type = typename inner_types::position_type;
        using ref_position_type = typename inner_types::ref_position_type;
        using force_type = typename inner_types::force_type;
        using ref_force_type = typename inner_types::ref_force_type;

        object_container(position_type pos, std::size_t size);

        // const auto pos() const;
        auto pos();
        // const auto pos(std::size_t i) const;
        auto pos(std::size_t i);

        auto force() const;
        auto force();

        std::size_t size() const;

    private:

        position_type m_pos;
        force_type m_force;
        std::size_t m_size;
    };

    /////////////////////////////////////
    // object_container implementation //
    /////////////////////////////////////
    template<std::size_t dim, bool owner>
    inline object_container<dim, owner>::object_container(position_type pos, std::size_t size)
    : m_pos(this->get_pointer(pos)), m_size(size)
    {}

    // template<std::size_t dim, bool owner>
    // inline const auto object_container<dim, owner>::pos() const
    // {
    //     return this->get_value(m_pos, m_size);
    // }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::pos()
    {
        return this->get_value(m_pos, m_size);
    }

    // template<std::size_t dim, bool owner>
    // inline const auto object_container<dim, owner>::pos(std::size_t i) const
    // {
    //     return this->get_value(m_pos, m_size, i);
    // }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::pos(std::size_t i)
    {
        return this->get_value(m_pos, m_size, i);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::force() const
    {
        return this->get_value(m_force, m_size);
    }

    template<std::size_t dim, bool owner>
    inline auto object_container<dim, owner>::force()
    {
        return this->get_value(m_force, m_size);
    }

    template<std::size_t dim, bool owner>
    inline std::size_t object_container<dim, owner>::size() const
    {
        return m_size;
    }

    ///////////////////////////////////
    // object_constructor definition //
    //////////////////////////////////
    template<std::size_t Dim, bool owner>
    class object;

    struct base_constructor
    {
    public:

        virtual std::shared_ptr<object<2, false>> operator()(std::array<double, 2>* pos) const = 0;
    };
    
    template<class T, class... Args>
    class object_constructor: public base_constructor
    {
    public:

        using object_type = T;
        using tuple_type = std::tuple<Args...>;

        template<class... CTA>
        object_constructor(CTA&&... args);

        virtual std::shared_ptr<object<2, false>> operator()(std::array<double, 2>* pos) const override;
        auto extra_args() const;

    private:

        template<std::size_t... I>
        auto lambda_constructor(std::array<double, 2>* pos, std::index_sequence<I...>) const;

        tuple_type m_extra;
    };

    ///////////////////////
    // object definition //
    ///////////////////////
    template<std::size_t Dim, bool owner=true>
    class object: public object_container<Dim, owner>
    {
    public:
        static constexpr std::size_t dim = Dim;
        using base_type = object_container<dim, owner>;
        using position_type = typename base_type::position_type;
        using ref_position_type = typename base_type::ref_position_type;
        using force_type = typename base_type::force_type;

        object() = default;
        object(std::array<double, dim>* pos, std::size_t size);
        object(const std::vector<std::array<double, dim>>& pos);

        virtual std::shared_ptr<base_constructor> construct() const = 0;
        virtual void print() const = 0;
        virtual std::size_t hash() const = 0;

    };

    ///////////////////////////
    // object implementation //
    ///////////////////////////
    template<std::size_t dim, bool owner>
    object<dim, owner>::object(const std::vector<std::array<double, dim>>& pos)
    : base_type(pos, pos.size())
    {
    }

    template<std::size_t dim, bool owner>
    object<dim, owner>::object(std::array<double, dim>* pos, std::size_t size)
    : base_type(pos, size)
    {
    }

    //////////////////////////////////////
    // object_constructor implementation //
    //////////////////////////////////////
    template<class T, class... Args>
    template<class... CTA>
    object_constructor<T, Args...>::object_constructor(CTA&&... args)
    : m_extra(std::forward<CTA>(args)...)
    {}

    template<class T, class... Args>
    std::shared_ptr<object<2, false>> object_constructor<T, Args...>::operator()(std::array<double, 2>* pos) const
    {
        return lambda_constructor(pos, std::make_index_sequence<sizeof...(Args)>{});
    }

    template<class T, class... Args>
    auto object_constructor<T, Args...>::extra_args() const
    {
        return m_extra;
    }

    template<class T, class... Args>
    template<std::size_t... I>
    auto object_constructor<T, Args...>::lambda_constructor(std::array<double, 2>* pos, std::index_sequence<I...>) const
    {
        return std::make_shared<object_type>(pos, std::get<I>(m_extra)...);
    }

    template<class T, class... Args>
    auto make_object_constructor(Args&&... args)
    {
        using constructor_type = object_constructor<T, Args...>;
        return std::make_shared<constructor_type>(args...);
    } 

    ////////////////////////
    // globule definition //
    ////////////////////////

    template<class position_type, std::size_t dim>
    position_type globule_construction(const std::array<double, dim>& pos)
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
        using ref_position_type = typename base_type::ref_position_type;
        using force_type = typename base_type::force_type;

        globule(const std::array<double, dim>& pos, double radius);
        globule(std::array<double, dim>* pos, double radius);

        virtual std::shared_ptr<base_constructor> construct() const override;
        virtual void print() const override;
        virtual std::size_t hash() const override;

    private:

        void create_hash();

        double m_radius;
        std::size_t m_hash;
    };

    ////////////////////////////
    // globule implementation //
    ////////////////////////////
    template<std::size_t dim, bool owner>
    globule<dim, owner>::globule(const std::array<double, dim>& pos, double radius)
    : base_type(globule_construction<position_type, dim>(pos)), m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    globule<dim, owner>::globule(std::array<double, dim>* pos, double radius)
    : base_type(pos, 6), m_radius(radius)
    {
        create_hash();
    }

    template<std::size_t dim, bool owner>
    std::shared_ptr<base_constructor> globule<dim, owner>::construct() const
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

#if defined(__GNUC__) && !defined(__clang__)
    namespace workaround
    {
        // Fixes "undefined symbol" issues
        inline void long_long_allocator()
        {
            std::allocator<long long> a;
            std::allocator<unsigned long long> b;
            std::allocator<double> c;
            std::allocator<std::complex<double>> d;
        }
    }
#endif
}
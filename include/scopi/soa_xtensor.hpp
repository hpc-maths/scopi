#pragma once

#include <algorithm>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <xtensor/xtensor.hpp>
//#include <xtensor/xarray.hpp>


namespace soa
{

    // Specialized for aggregates so soa::vector<T> can be istanciated.
    // Specialization of non-template types can be done with the macro
    // 'SOA_DEFINE_TYPE(type, members...);' in the global namespace.
    template<class Aggregate>
    struct members
    {
    };

    // These proxy types are defined with the macro. They are created when
    // iterating on a soa::vector and mimic the given aggregate members as
    // references.
    template<class Aggregate>
    struct ref_proxy
    {
    };
    template<class Aggregate>
    struct cref_proxy
    {
    };

    // A trait allows to check if the three class above have been defined for
    // the given type.
    template<class Aggregate>
    constexpr bool is_defined_v = !std::is_empty_v<members<Aggregate>> &&
                                  !std::is_empty_v<ref_proxy<Aggregate>> &&
                                  !std::is_empty_v<cref_proxy<Aggregate>>;

    template<class T>
    class vector;

    template<class container_type>
    class vector_span : private container_type {
        template<class>
        friend class vector;

      public:
        using container = container_type;
        using container::begin;
        using container::cbegin;
        using container::cend;
        using container::crbegin;
        using container::crend;
        using container::end;
        using container::fill;
        using container::rbegin;
        using container::rend;
        using container::size;
        using container::operator[];
        using container::operator=;

      private:
        using container::resize;
    };

    namespace detail
    {
        // Aggregate to tuple implementation, only for soa::member<T>.

        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 1>)
        {
            auto & [v1] = agg;
            return std::forward_as_tuple(v1);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 2>)
        {
            auto & [ v1, v2 ] = agg;
            return std::forward_as_tuple(v1, v2);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 3>)
        {
            auto & [ v1, v2, v3 ] = agg;
            return std::forward_as_tuple(v1, v2, v3);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 4>)
        {
            auto & [ v1, v2, v3, v4 ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 5>)
        {
            auto & [ v1, v2, v3, v4, v5 ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 6>)
        {
            auto & [ v1, v2, v3, v4, v5, v6 ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 7>)
        {
            auto & [ v1, v2, v3, v4, v5, v6, v7 ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 8>)
        {
            auto & [ v1, v2, v3, v4, v5, v6, v7, v8 ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 9>)
        {
            auto & [ v1, v2, v3, v4, v5, v6, v7, v8, v9 ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 10>)
        {
            auto & [ v1, v2, v3, v4, v5, v6, v7, v8, v9, v10 ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 11>)
        {
            auto & [ v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11 ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 12>)
        {
            auto & [ v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12 ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11, v12);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 13>)
        {
            auto & [ v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13 ] =
                agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11, v12, v13);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 14>)
        {
            auto & [
                v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14
            ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11, v12, v13, v14);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 15>)
        {
            auto & [
                v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15
            ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11, v12, v13, v14, v15);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 16>)
        {
            auto & [
                v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                v15, v16
            ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11, v12, v13, v14, v15, v16);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 17>)
        {
            auto & [
                v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                v15, v16, v17
            ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11, v12, v13, v14, v15, v16,
                                         v17);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 18>)
        {
            auto & [
                v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                v15, v16, v17, v18
            ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11, v12, v13, v14, v15, v16, v17,
                                         v18);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 19>)
        {
            auto & [
                v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14,
                v15, v16, v17, v18, v19
            ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11, v12, v13, v14, v15, v16, v17,
                                         v18, v19);
        }
        template<class T>
        auto as_tuple(T &agg, std::integral_constant<int, 20>)
        {
            auto & [
                v1,  v2,  v3,  v4,  v5,  v6,  v7,  v8,  v9,  v10,
                v11, v12, v13, v14, v15, v16, v17, v18, v19, v20
            ] = agg;
            return std::forward_as_tuple(v1, v2, v3, v4, v5, v6, v7, v8, v9,
                                         v10, v11, v12, v13, v14, v15, v16, v17,
                                         v18, v19, v20);
        }

        // The arity is the number of members of a well-formed soa::member<T>.
        template<class Members>
        constexpr int
            arity_v = sizeof(Members) /
                      sizeof(vector_span<typename Members::container_base>);

        // Continue the overloads above to increase the max_arity.
        constexpr int max_arity = 20;

        // Converts a well-formed soa::member<T> to a tuple with references on
        // each member of the class.
        template<class T>
        auto as_tuple(members<T> const &agg)
        {
            // std::cout << arity_v<members<T>> << "\n";
            return as_tuple(agg,
                            std::integral_constant<int, arity_v<members<T>>>{});
        }
        template<class T>
        auto as_tuple(members<T> &agg)
        {
            // std::cout << arity_v<members<T>> << "\n";
            return as_tuple(agg,
                            std::integral_constant<int, arity_v<members<T>>>{});
        }

        // Allows to converts any aggregate to a tuple given it's arity.
        template<size_t Arity, class T>
        auto as_tuple(T &&agg)
        {
            return as_tuple(agg, std::integral_constant<int, Arity>{});
        }

        template<class F, size_t... Is, class... Ts>
        void for_each(std::tuple<Ts &...> const &tuple,
                      std::index_sequence<Is...>, F &&f)
        {
            (f(std::get<Is>(tuple)), ...);
        }

        template<class F, size_t... Is, class... Ts1, class... Ts2>
        void for_each(std::tuple<Ts1 &...> const &tuple1,
                      std::tuple<Ts2 &...> const &tuple2,
                      std::index_sequence<Is...>, F &&f)
        {
            static_assert(sizeof...(Ts1) == sizeof...(Ts2));
            (f(std::get<Is>(tuple1), std::get<Is>(tuple2)), ...);
        }

        // template<class T>
        // [[noreturn]] void throw_out_of_range(int index, int size)
        // {
        //     using namespace std::literals;

        //     throw std::out_of_range{detail::concatene(
        //         "Out of bounds access when calling "sv,
        //         detail::type_name<T>(),
        //         "::at("sv, std::to_string(index), ") while size = "sv,
        //         std::to_string(size))};
        // }
    } // namespace detail

    namespace detail
    {
        // Base class of soa::vector<T>.
        // Used to retrieve the size by soa::vector_span<ContainerType> from
        // members<T>.
        template<class T>
        class members_with_size : public members<T> {
            template<class>
            friend class ::soa::vector_span;

          protected:
            std::size_t size_;
        };

        template<class Vector, bool IsConst>
        class proxy_iterator {
            friend Vector;

            using vector_pointer_type =
                std::conditional_t<IsConst, Vector const *, Vector *>;

            vector_pointer_type vec_;
            std::size_t index_;

            proxy_iterator(vector_pointer_type vec, std::size_t index) noexcept
                : vec_{vec}, index_{index}
            {}

          public:
            using iterator_category = std::random_access_iterator_tag;

            using value_type =
                std::conditional_t<IsConst,
                                   typename Vector::const_reference_type,
                                   typename Vector::reference_type>;

            using reference = value_type;
            using pointer = void;
            using difference_type = int;

          private:
            template<size_t... Is>
            value_type make_proxy(std::index_sequence<Is...>) const noexcept
            {
                return {vec_->template get_span<Is>()[index_]...};
            }

          public:
            value_type operator*() const noexcept
            {
                return make_proxy(typename Vector::sequence_type{});
            }

            bool operator==(proxy_iterator const &rhs) const noexcept
            {
                return index_ == rhs.index_;
            }
            bool operator!=(proxy_iterator const &rhs) const noexcept
            {
                return !(*this == rhs);
            }

            bool operator<(proxy_iterator const &rhs) const noexcept
            {
                return index_ < rhs.index_;
            }
            bool operator>(proxy_iterator const &rhs) const noexcept
            {
                return rhs < *this;
            }
            bool operator<=(proxy_iterator const &rhs) const noexcept
            {
                return !(rhs < *this);
            }
            bool operator>=(proxy_iterator const &rhs) const noexcept
            {
                return !(*this < rhs);
            }

            proxy_iterator &operator++() noexcept
            {
                return ++index_, *this;
            }
            proxy_iterator &operator--() noexcept
            {
                return --index_, *this;
            }
            proxy_iterator &operator++(int)noexcept
            {
                const auto old = *this;
                return ++index_, old;
            }
            proxy_iterator &operator--(int)noexcept
            {
                const auto old = *this;
                return --index_, old;
            }

            proxy_iterator &operator+=(int shift) noexcept
            {
                return index_ += shift, *this;
            }
            proxy_iterator &operator-=(int shift) noexcept
            {
                return index_ -= shift, *this;
            }

            proxy_iterator operator+(int shift) const noexcept
            {
                return {vec_, index_ + shift};
            }
            proxy_iterator operator-(int shift) const noexcept
            {
                return {vec_, index_ - shift};
            }

            int operator-(proxy_iterator const &rhs) const noexcept
            {
                return index_ - rhs.index_;
            }
        };
    } // namespace detail

    template<class T>
    class vector : public detail::members_with_size<T> {
      public:
        static_assert(
            is_defined_v<T>,
            "soa::vector<T> can't be instancied because the required types "
            "'soa::members<T>', "
            "'soa::ref_proxy<T>' or 'soa::cref_proxy<T>' haven't been defined. "
            "Did you forget to call the macro SOA_DEFINE_TYPE(T, members...) "
            "?");

        using value_type = T;
        using reference_type = ref_proxy<T>;
        using const_reference_type = cref_proxy<T>;
        using size_type = std::size_t;

        using iterator = detail::proxy_iterator<vector, false>;
        using const_iterator = detail::proxy_iterator<vector, true>;
        using reverse_iterator = std::reverse_iterator<iterator>;
        using const_reverse_iterator = std::reverse_iterator<const_iterator>;

        static constexpr std::size_t components_count =
            detail::arity_v<members<T>>;

        // construct/copy/destroy:
        vector();
        vector(const vector &rhs);
        vector(vector &&rhs);

        vector &operator=(vector &&rhs);
        vector &operator=(vector const &rhs);

        // iterators:
        iterator begin() noexcept;
        const_iterator begin() const noexcept;
        iterator end() noexcept;
        const_iterator end() const noexcept;

        reverse_iterator rbegin() noexcept;
        const_reverse_iterator rbegin() const noexcept;
        reverse_iterator rend() noexcept;
        const_reverse_iterator rend() const noexcept;

        const_iterator cbegin() const noexcept;
        const_iterator cend() const noexcept;
        const_reverse_iterator crbegin() const noexcept;
        const_reverse_iterator crend() const noexcept;

        // capacity:
        size_type size() const noexcept;
        void resize(size_type size);

        // element access:
        reference_type operator[](size_type i) noexcept;
        const_reference_type operator[](size_type i) const noexcept;
        reference_type at(size_type i) noexcept;
        const_reference_type at(size_type i) const noexcept;
        reference_type front() noexcept;
        const_reference_type front() const noexcept;
        reference_type back() noexcept;
        const_reference_type back() const noexcept;

        // modifiers:
        void push_back(T const &value);
        void push_back(T &&value);

      private:
        friend iterator;
        friend const_iterator;

        using sequence_type = std::make_index_sequence<components_count>;

        members<T> &base() noexcept;
        members<T> const &base() const noexcept;

        detail::members_with_size<T> &base_with_size() noexcept;
        detail::members_with_size<T> const &base_with_size() const noexcept;

        template<std::size_t I>
        auto &get_span() noexcept;
        template<std::size_t I>
        auto const &get_span() const noexcept;
    };

    template<class T>
    inline vector<T>::vector() : detail::members_with_size<T>{}
    {
        this->size_ = get_span<0>().size();
    }

    template<class T>
    inline vector<T>::vector(const vector &rhs)
    {
        detail::for_each(detail::as_tuple(base()), detail::as_tuple(rhs.base()),
                         sequence_type{},
                         [](auto &span1, auto &span2) { span1 = span2; });
        this->size_ = get_span<0>().size();
    }

    template<class T>
    inline vector<T>::vector(vector &&rhs)
    {
        detail::for_each(detail::as_tuple(base()), detail::as_tuple(rhs.base()),
                         sequence_type{}, [](auto &span1, auto &span2) {
                             span1 = std::move(span2);
                         });
        rhs.size_ = 0;

        this->size_ = get_span<0>().size();
    }

    template<class T>
    vector<T> &vector<T>::operator=(vector<T> &&rhs)
    {
        detail::for_each(detail::as_tuple(base()), detail::as_tuple(rhs.base()),
                         sequence_type{}, [](auto &span1, auto &span2) {
                             span1 = std::move(span2);
                         });
        rhs.size_ = 0;

        this->size_ = get_span<0>().size();
        return *this;
    }

    template<class T>
    vector<T> &vector<T>::operator=(vector<T> const &rhs)
    {
        detail::for_each(detail::as_tuple(base()), detail::as_tuple(rhs.base()),
                         sequence_type{},
                         [](auto &span1, auto &span2) { span1 = span2; });
        this->size_ = get_span<0>().size();
        return *this;
    }

    template<class T>
    inline auto vector<T>::begin() noexcept -> iterator
    {
        return {this, 0};
    }

    template<class T>
    inline auto vector<T>::begin() const noexcept -> const_iterator
    {
        return {this, 0};
    }

    template<class T>
    inline auto vector<T>::end() noexcept -> iterator
    {
        return {this, size()};
    }

    template<class T>
    inline auto vector<T>::end() const noexcept -> const_iterator
    {
        return {this, size()};
    }

    template<class T>
    inline auto vector<T>::rbegin() noexcept -> reverse_iterator
    {
        return reverse_iterator(end());
    }

    template<class T>
    inline auto vector<T>::rbegin() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(end());
    }

    template<class T>
    inline auto vector<T>::rend() noexcept -> reverse_iterator
    {
        return reverse_iterator(begin());
    }

    template<class T>
    inline auto vector<T>::rend() const noexcept -> const_reverse_iterator
    {
        return const_reverse_iterator(begin());
    }

    template<class T>
    inline auto vector<T>::cbegin() const noexcept -> const_iterator
    {
        return begin();
    }

    template<class T>
    inline auto vector<T>::cend() const noexcept -> const_iterator
    {
        return end();
    }

    template<class T>
    inline auto vector<T>::crbegin() const noexcept -> const_reverse_iterator
    {
        return rbegin();
    }

    template<class T>
    inline auto vector<T>::crend() const noexcept -> const_reverse_iterator
    {
        return rend();
    }

    template<class T>
    inline auto vector<T>::size() const noexcept -> size_type
    {
        return this->size_;
    }

    template<class T>
    void vector<T>::resize(size_type size)
    {
        detail::for_each(detail::as_tuple(base()), sequence_type{},
                         [size](auto &span) { span.resize({size}); });
        this->size_ = get_span<0>().size();
    }

    template<class T>
    inline auto vector<T>::operator[](size_type i) noexcept -> reference_type
    {
        return *(begin() + i);
    }

    template<class T>
    inline auto vector<T>::operator[](size_type i) const noexcept
        -> const_reference_type
    {
        return *(begin() + i);
    }

    template<class T>
    inline auto vector<T>::at(size_type i) noexcept -> reference_type
    {
        // check_at(i);
        return *(begin() + i);
    }

    template<class T>
    inline auto vector<T>::at(size_type i) const noexcept
        -> const_reference_type
    {
        // check_at(i);
        return *(begin() + i);
    }

    template<class T>
    inline auto vector<T>::front() noexcept -> reference_type
    {
        return *begin();
    }

    template<class T>
    inline auto vector<T>::front() const noexcept -> const_reference_type
    {
        return *begin();
    }

    template<class T>
    inline auto vector<T>::back() noexcept -> reference_type
    {
        return *(end() - 1);
    }

    template<class T>
    inline auto vector<T>::back() const noexcept -> const_reference_type
    {
        return *(end() - 1);
    }

    template<class T>
    members<T> &vector<T>::base() noexcept
    {
        return *this;
    }

    template<class T>
    members<T> const &vector<T>::base() const noexcept
    {
        return *this;
    }

    template<class T>
    detail::members_with_size<T> &vector<T>::base_with_size() noexcept
    {
        return *this;
    }

    template<class T>
    detail::members_with_size<T> const &vector<T>::base_with_size() const
        noexcept
    {
        return *this;
    }

    template<class T>
    template<std::size_t I>
    auto &vector<T>::get_span() noexcept
    {
        static_assert(I < components_count);
        return std::get<I>(detail::as_tuple(base()));
    }

    template<class T>
    template<std::size_t I>
    auto const &vector<T>::get_span() const noexcept
    {
        static_assert(I < components_count);
        return std::get<I>(detail::as_tuple(base()));
    }
} // namespace soa

#define SOA_PP_EMPTY
#define SOA_PP_EMPTY_ARGS(...)

#define SOA_PP_EVAL0(...) __VA_ARGS__
#define SOA_PP_EVAL1(...) SOA_PP_EVAL0(SOA_PP_EVAL0(SOA_PP_EVAL0(__VA_ARGS__)))
#define SOA_PP_EVAL2(...) SOA_PP_EVAL1(SOA_PP_EVAL1(SOA_PP_EVAL1(__VA_ARGS__)))
#define SOA_PP_EVAL3(...) SOA_PP_EVAL2(SOA_PP_EVAL2(SOA_PP_EVAL2(__VA_ARGS__)))
#define SOA_PP_EVAL4(...) SOA_PP_EVAL3(SOA_PP_EVAL3(SOA_PP_EVAL3(__VA_ARGS__)))
#define SOA_PP_EVAL(...) SOA_PP_EVAL4(SOA_PP_EVAL4(SOA_PP_EVAL4(__VA_ARGS__)))

#define SOA_PP_MAP_GET_END() 0, SOA_PP_EMPTY_ARGS

#define SOA_PP_MAP_NEXT0(item, next, ...) next SOA_PP_EMPTY
#if defined(_MSC_VER)
#define SOA_PP_MAP_NEXT1(item, next)                                           \
    SOA_PP_EVAL0(SOA_PP_MAP_NEXT0(item, next, 0))
#else
#define SOA_PP_MAP_NEXT1(item, next) SOA_PP_MAP_NEXT0(item, next, 0)
#endif
#define SOA_PP_MAP_NEXT(item, next)                                            \
    SOA_PP_MAP_NEXT1(SOA_PP_MAP_GET_END item, next)

#define SOA_PP_MAP0(f, n, t, x, peek, ...)                                     \
    f(n, t, x)                                                                 \
        SOA_PP_MAP_NEXT(peek, SOA_PP_MAP1)(f, n + 1, t, peek, __VA_ARGS__)
#define SOA_PP_MAP1(f, n, t, x, peek, ...)                                     \
    f(n, t, x)                                                                 \
        SOA_PP_MAP_NEXT(peek, SOA_PP_MAP0)(f, n + 1, t, peek, __VA_ARGS__)
#define SOA_PP_MAP(f, t, ...)                                                  \
    SOA_PP_EVAL(SOA_PP_MAP1(f, 0, t, __VA_ARGS__, (), 0))

#define SOA_PP_MEMBER(nb, type, name)                                          \
    vector_span<xt::xtensor<decltype(std::declval<type>().name), 1>> name;
//vector_span<xt::xarray<decltype(std::declval<type>().name)>> name;

#define SOA_PP_REF(nb, type, name) decltype(std::declval<type>().name) &name;

#define SOA_PP_CREF(nb, type, name)                                            \
    decltype(std::declval<type>().name) const &name;

#define SOA_PP_COPY(nb, type, name) name = rhs.name;

#define SOA_PP_MOVE(nb, type, name) name = std::move(rhs.name);

#define SOA_PP_ENABLE_FOR_COPYABLE(type, alias)                                \
    template<class alias,                                                      \
             class = std::enable_if_t<std::is_same_v<alias, type> &&           \
                                      std::is_copy_constructible_v<type>>>

// Shortcut to specialize soa::member<my_type>, by listing all the members
// in their declaration order. It must be used in the global namespace.
// Usage exemple :
//
// namespace user {
//     struct person {
//         std::string name;
//         int age;
//     };
// }
//
// SOA_DEFINE_TYPE(user::person, name, age);
//
// This is equivalent to typing :
//
// namespace soa {
//     template <>
//     struct members<user::person> {
//         vector_span<0, user::person, std::string> name;
//         vector_span<1, user::person, int> age;
//     };
//     template <>
//     struct ref_proxy<user::person> {
//         std::string & name;
//         int & age;
//
//         operator user::person() const {
//             return { name, age };
//         }
//
//         ref_proxy& operator=(user::person const& rhs) {
//             name = rhs.name;
//             age = rhs.age;
//             return *this;
//         }
//         ref_proxy& operator=(user::person && rhs) noexcept {
//             name = std::move(rhs.name);
//             age = std::move(rhs.age);
//             return *this;
//         }
//     };
//     template <>
//     struct cref_proxy<user::person> {
//         std::string const& name;
//         int const& age;
//
//         operator user::person() const {
//             return { name, age };
//         }
//     };
// }


// template<>                                                             \
// struct members<::type>                                                 \
// {                                                                      \
//     using container_base = xt::xarray<char>;                       \
//     SOA_PP_MAP(SOA_PP_MEMBER, ::type, __VA_ARGS__)                     \
// };

#define SOA_DEFINE_TYPE(type, ...)                                             \
    namespace soa                                                              \
    {                                                                          \
        template<>                                                             \
        struct members<::type>                                                 \
        {                                                                      \
            using container_base = xt::xtensor<char, 1>;                       \
            SOA_PP_MAP(SOA_PP_MEMBER, ::type, __VA_ARGS__)                     \
        };                                                                     \
        template<>                                                             \
        struct ref_proxy<::type>                                               \
        {                                                                      \
            SOA_PP_MAP(SOA_PP_REF, ::type, __VA_ARGS__)                        \
                                                                               \
            ref_proxy &operator=(::type &&rhs) noexcept                        \
            {                                                                  \
                SOA_PP_MAP(SOA_PP_MOVE, ::type, __VA_ARGS__)                   \
                return *this;                                                  \
            }                                                                  \
            SOA_PP_ENABLE_FOR_COPYABLE(::type, _type)                          \
            ref_proxy &operator=(_type const &rhs)                             \
            {                                                                  \
                SOA_PP_MAP(SOA_PP_COPY, ::type, __VA_ARGS__)                   \
                return *this;                                                  \
            }                                                                  \
            SOA_PP_ENABLE_FOR_COPYABLE(::type, _type)                          \
            operator _type() const                                             \
            {                                                                  \
                return {__VA_ARGS__};                                          \
            }                                                                  \
        };                                                                     \
        template<>                                                             \
        struct cref_proxy<::type>                                              \
        {                                                                      \
            SOA_PP_MAP(SOA_PP_CREF, ::type, __VA_ARGS__)                       \
                                                                               \
            SOA_PP_ENABLE_FOR_COPYABLE(::type, _type)                          \
            operator _type() const                                             \
            {                                                                  \
                return {__VA_ARGS__};                                          \
            }                                                                  \
        };                                                                     \
    }                                                                          \
    struct _soa__force_semicolon_

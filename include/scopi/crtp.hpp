#pragma once 

namespace scopi
{
    class Foo{};

    template <class D, class crtpType = Foo>
    class crtp_base
    {
    public:
        using derived_type = D;

        const derived_type& derived_cast() const & noexcept;
        derived_type& derived_cast() & noexcept;
        derived_type derived_cast() && noexcept;

    protected:
        crtp_base() = default;
        ~crtp_base() = default;

        crtp_base(const crtp_base&) = default;
        crtp_base& operator=(const crtp_base&) = default;

        crtp_base(crtp_base&&) = default;
        crtp_base& operator=(crtp_base&&) = default;
    };

    template <class D, class crtpType>
    inline auto crtp_base<D, crtpType>::derived_cast() const & noexcept -> const derived_type&
    {
        return *static_cast<const derived_type*>(this);
    }

    template <class D, class crtpType>
    inline auto crtp_base<D, crtpType>::derived_cast() & noexcept -> derived_type&
    {
        return *static_cast<derived_type*>(this);
    }

    template <class D, class crtpType>
    inline auto crtp_base<D, crtpType>::derived_cast() && noexcept -> derived_type
    {
        return *static_cast<derived_type*>(this);
    }
}

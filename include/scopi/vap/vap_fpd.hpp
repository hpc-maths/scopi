#pragma once

#include "base.hpp"

namespace scopi
{
    class vap_fpd: public vap_base<vap_fpd>
    {
        public:
            using base_type = vap_base<vap_fpd>;
            template <std::size_t dim>
                void aPrioriVelocity_impl(scopi_container<dim>& particles);

            template <std::size_t dim>
                void updateVelocity_impl(scopi_container<dim>& particles, const xt::xtensor<double, 2>& uadapt, const xt::xtensor<double, 2>& wadapt);

            vap_fpd(std::size_t Nactive, std::size_t active_ptr, double dt);

        private:
            template <std::size_t dim>
                auto f_ext(scopi_container<dim>& particles, std::size_t i);
            double t_ext();

            double _mass;
            double _moment;

    };

    vap_fpd::vap_fpd(std::size_t Nactive, std::size_t active_ptr, double dt)
        : base_type(Nactive, active_ptr, dt)
          , _mass(1.)
          , _moment(0.1)
    {}

    template <std::size_t dim>
        void vap_fpd::aPrioriVelocity_impl(scopi_container<dim>& particles)
        {
            // TODO what if f_ext and t_ext depends on the particule ?
            for (std::size_t i=0; i<_Nactive; ++i)
            {
                // auto tmp = f_ext(particles, i);
                // std::cout << tmp << std::endl;
                auto pos = particles.pos()(i + _active_ptr);
                double radius = 0.4;//xt::amax(particles[i]->radius()); // TODO radius for test critical_2d_no_velocity, shoud depend on the particle
                double dist = xt::linalg::norm(pos) + radius;
                auto fExt = - 10.*_mass/(dist*dist)*pos/dist;
                particles.vd()(_active_ptr + i) = particles.v()(_active_ptr + i) + _dt*fExt/_mass; // TODO: add mass into particles
            }
            // TODO should be dt * (R_i * t_i^{ext , n} - omega'_i * (J_i omega'_i)
            particles.desired_omega() = particles.omega() + _dt*t_ext();
        }

    template <std::size_t dim>
        void vap_fpd::updateVelocity_impl(scopi_container<dim>& particles, const xt::xtensor<double, 2>& uadapt, const xt::xtensor<double, 2>& wadapt)
        {
            for (std::size_t i=0; i<_Nactive; ++i)
            {
                for (std::size_t d=0; d<dim; ++d)
                {
                    particles.v()(i + _active_ptr)(d) = uadapt(i, d);
                }
                particles.omega()(i + _active_ptr) = wadapt(i, 2);
            }
        }

    template <std::size_t dim>
        auto vap_fpd::f_ext(scopi_container<dim>& particles, std::size_t i)
        {
            auto pos = particles.pos()(i + _active_ptr);
            double dist = xt::linalg::norm(pos);
            auto res = _mass/(dist*dist)*pos/dist;
            std::cout << res << std::endl;
            return res;
        }

    double vap_fpd::t_ext()
    {
        return 0.;
    }
}

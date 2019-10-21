#include "scopi/particles.hpp"


Particles::~Particles() {
}

Particles::Particles(const std::string &name) : _name(name) {

}

void Particles::add_particles_in_ball(std::size_t n, double rmin, double rmax,
  double xc, double yc, double zc, double rball){

    const double pi = std::acos(-1);

    std::size_t psize = _data.size();
    if (psize==0){
      _data.resize(n);
      xt::xtensor<double, 1> phi = xt::random::rand<double>({n})*2*pi;
      xt::xtensor<double, 1> theta = xt::acos(-1+xt::random::rand<double>({n})*2);
      xt::xtensor<double, 1> rb = rball*xt::random::rand<double>({n});
      _data.x = xc+rb * xt::sin( theta) * xt::cos( phi );
      _data.y = yc+rb * xt::sin( theta) * xt::sin( phi );
      _data.z = zc+rb * xt::cos( theta );
      _data.r = xt::view(rmin+xt::random::rand<double>({n})*(rmax-rmin), xt::range(psize,xt::placeholders::_));
    }
    else {
      // xt::xtensor<double, 1> phi = xt::random::rand<double>({n})*2*pi;
      // xt::xtensor<double, 1> theta = xt::acos(-1+xt::random::rand<double>({n})*2);
      // xt::xtensor<double, 1> rb = rball*xt::random::rand<double>({n});
      // _data.x = xt::concatenate(xt::xtuple(_data.x, xt::xtensor<double, 1>( xc+rb * xt::sin( theta) * xt::cos( phi ) )));
      // _data.y = xt::concatenate(xt::xtuple(_data.y, xt::xtensor<double, 1>( yc+rb * xt::sin( theta) * xt::sin( phi ) )));
      // _data.z = xt::concatenate(xt::xtuple(_data.z, xt::xtensor<double, 1>( zc+rb * xt::cos( theta ) )));
      // _data.r = xt::concatenate(xt::xtuple(_data.r, xt::xtensor<double, 1>( rmin+xt::random::rand<double>({n})*(rmax-rmin) )));

      std::cerr<<"  Particles / add_particles_in_ball : Attention le resize ecrase les données déjà présentes... xtensor does not preserve data!  EXIT"<<std::endl;
      exit(EXIT_FAILURE);
    }
  }

  xt::pyarray<double> Particles::xyzr(){
    return xt::stack(xt::xtuple(xt::flatten(_data.x), xt::flatten(_data.y), xt::flatten(_data.z), xt::flatten(_data.r)),1);
  }

  void Particles::add_particles_in_box(std::size_t n,
    double rmin, double rmax,
    double xmin, double xmax,
    double ymin, double ymax,
    double zmin, double zmax){

      std::size_t psize = _data.size();
      if (psize==0){
        _data.resize(n);
        _data.x = xt::view(xmin+xt::random::rand<double>({n})*(xmax-xmin), xt::range(psize,xt::placeholders::_));
        _data.y = xt::view(ymin+xt::random::rand<double>({n})*(ymax-ymin), xt::range(psize,xt::placeholders::_));
        _data.z = xt::view(zmin+xt::random::rand<double>({n})*(zmax-zmin), xt::range(psize,xt::placeholders::_));
        _data.r = xt::view(rmin+xt::random::rand<double>({n})*(rmax-rmin), xt::range(psize,xt::placeholders::_));
      }
      else {
        // A réécrire mieux avec xtensor... strides/view/push_back...
        xt::xtensor<double, 1> x = xt::random::rand<double>({n});
        xt::xtensor<double, 1> y = xt::random::rand<double>({n});
        xt::xtensor<double, 1> z = xt::random::rand<double>({n});
        xt::xtensor<double, 1> r = xt::random::rand<double>({n});
        _data.resize(psize+n);
        std::cerr<<"  Particles / add_particles_in_box : Attention le resize ecrase les données déjà présentes... EXIT"<<std::endl;
        exit(EXIT_FAILURE);
        // for (std::size_t ip=0; ip<n; ++ip){
        //   _data.x[psize+ip] = x[ip];
        //   _data.y[psize+ip] = x[ip];
        //   _data.z[psize+ip] = x[ip];
        //   _data.r[psize+ip] = x[ip];
        // }
        //??? _data.x(xt::range(psize,xt::placeholders::_)) = xt::view(xt::random::rand<double>({n}), xt::range(psize,xt::placeholders::_));
      }
      //std::cout << typeid(x()).name() << '\n';
    }

    void Particles::print(){
      std::cout<<"  Particles : name = "<<_name<<std::endl;
      auto print_data = [](const auto &p) {  std::cout << "  Particles : " << p.x << " " << p.y  << " " << p.z  << " " << p.r << "\n"; };
      std::for_each(_data.begin(), _data.end(), print_data);
      std::cout << "\n";

    }

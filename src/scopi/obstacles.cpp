#include "scopi/obstacles.hpp"


Obstacles::~Obstacles() {
}

Obstacles::Obstacles() {

}

void Obstacles::add_obstacle(Obstacle obs)
{
  _obstacles.push_back(obs);

}


void Obstacles::print() const{

  std::cout<<"\n-- C++ -- Obstacles : "<<std::endl;
  auto print_obs = [](const auto &p) { p.print(); };
  std::for_each(_obstacles.begin(), _obstacles.end(), print_obs);

}

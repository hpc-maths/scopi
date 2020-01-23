#ifndef DEF_Obstacles
#define DEF_Obstacles

#include <vector>
#include "scopi/obstacle.hpp"

/// @class Obstacles
/// @bried This class manages all the obstacles
class Obstacles
{
public:

  /// @brief Constructor
  /// Instantiate vector of obstacles
  Obstacles();

  /// @brief Destructor
  ~Obstacles();

  /// @brief Add an obstacle
  void add_obstacle(Obstacle obs);

  /// @brief Print the obstacles
  void print() const;

private:

  std::vector<Obstacle> _obstacles;

};

#endif

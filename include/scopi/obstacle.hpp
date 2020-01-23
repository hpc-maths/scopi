#ifndef DEF_Obstacle
#define DEF_Obstacle

#include <iostream>
#include <vector>
#include <string>
#include <math.h>

#include <vtkPolyData.h>
#include <vtkSTLReader.h>
#include <vtkSmartPointer.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRenderWindowInteractor.h>

#include "xtensor/xarray.hpp"
#include "xtensor/xadapt.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xstrided_view.hpp"
#include "xtensor/xoperation.hpp"

//#include <xtensor-io/xhighfive.hpp>

#include "xtensor-python/pytensor.hpp"
#include "xtensor-python/pyarray.hpp"

#include "fm/fm.hpp"

/// @class Obstacle
/// @bried This class manages an obstacle
class Obstacle
{
public:

  /// @brief Constructor
  /// Instantiate an obstacle
  Obstacle(const std::string &model_filename, double dmax, double h);
  
  /// @brief Destructor
  ~Obstacle();

  std::string get_name() const;

  /// @brief Print the obstacle
  void print() const;

private:

  std::string _name;
  std::string _model_filename;
  double _dmax, _h, _xmin, _xmax, _ymin, _ymax, _zmin, _zmax;
  std::size_t _Nx, _Ny, _Nz;
  xt::xtensor<double, 3> _distance;
  xt::xtensor<double, 3> _gradx, _grady, _gradz;

};

#endif

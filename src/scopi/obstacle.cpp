#include "scopi/obstacle.hpp"


Obstacle::~Obstacle() {
}


Obstacle::Obstacle(const std::string &model_filename, double dmax, double h) :
_model_filename(model_filename), _dmax(dmax), _h(h) {

  vtkSmartPointer<vtkSTLReader> reader = vtkSmartPointer<vtkSTLReader>::New();
  reader->SetFileName(_model_filename.c_str());
  reader->Update();
  std::cout << "-- C++ -- Obstacle : vtkSTLReader = " << std::endl;
  reader->Print(std::cout);
  vtkSmartPointer<vtkPolyData> data = reader->GetOutput();
  double * bounds = data->GetBounds();
  _xmin=bounds[0]-_dmax;
  _xmax=bounds[1]+_dmax;
  _ymin=bounds[2]-_dmax;
  _ymax=bounds[3]+_dmax;
  _zmin=bounds[4]-_dmax;
  _zmax=bounds[5]+_dmax;
  _Nx = round((_xmax-_xmin)/_h);
  _Ny = round((_ymax-_ymin)/_h);
  _Nz = round((_zmax-_zmin)/_h);
  xt::xarray<double> img = 1.0e20*xt::ones<double>({_Nx, _Ny, _Nz});

  std::size_t npts = data->GetNumberOfPoints();

  for(std::size_t ip = 0; ip < npts; ip++)
  {
    double p[3];
    data->GetPoint(ip,p);
    // This is identical to: data->GetPoints()->GetPoint(ip,p);
    std::size_t i = round((p[0]-_xmin)/_h);
    std::size_t j = round((p[1]-_ymin)/_h);
    std::size_t k = round((p[2]-_zmin)/_h);
    //std::cout << "-- C++ -- Obstacle : ";
    //std::cout << "point : (" << p[0] << " " << p[1] << " " << p[2] << ")";
    //std::cout << " <=> (" << i << " " << j << " " << k << ") "<< std::endl;
    img(i,j,k) = 0;
  }

  xt::xarray<std::size_t> narrow_band = xt::from_indices( xt::argwhere( xt::equal(img, 0) ) );
  //std::cout <<  "-- C++ -- Obstacle : narrow_band = "<< narrow_band << std::endl;

  // Fast-marching method...
  _distance = xt_compute_distance(_h, 1.0e20, img, narrow_band);
  //std::cout <<"-- C++ -- Obstacle : distance : "<< _distance << std::endl;
  //xt::dump_hdf5("img.h5", "./", _distance);

  // Let us compute the gradient of the distance...
  _gradx = xt::view(_distance, xt::range(2, xt::placeholders::_), xt::range(1, -1), xt::range(1, -1))
  - xt::view(_distance, xt::range(0, -2), xt::range(1, -1), xt::range(1, -1));

  _grady = xt::view(_distance, xt::range(1, -1), xt::range(2, xt::placeholders::_), xt::range(1, -1))
  - xt::view(_distance, xt::range(1, -1), xt::range(0, -2), xt::range(1, -1));

  _gradz = xt::view(_distance, xt::range(1, -1), xt::range(1, -1), xt::range(2, xt::placeholders::_))
  - xt::view(_distance, xt::range(1, -1), xt::range(1, -1), xt::range(0, -2));

  auto norm = xt::sqrt(_gradx*_gradx+_grady*_grady+_gradz*_gradz);
  _gradx = _gradx/norm;
  _grady = _grady/norm;
  _gradz = _gradz/norm;

  // Let us remove boundary points...

  _distance = xt::view(_distance, xt::range(1, -1), xt::range(1, -1), xt::range(1, -1));

  _Nx -= 2;
  _Ny -= 2;
  _Nz -= 2;
  _xmin += h;
  _xmax -= h;
  _ymin += h;
  _ymax -= h;
  _zmin += h;
  _zmax -= h;

  std::cout << "-- C++ -- Obstacle : box size = [" << _xmin << "," << _xmax << "]x[" << _ymin << "," << _ymax << "]x[" << _zmin << "," << _zmax << "]" << std::endl;
  std::cout << "-- C++ -- Obstacle : distance.shape = " << xt::adapt(_distance.shape()) << std::endl;
  std::cout << "-- C++ -- Obstacle : gradx.shape = " << xt::adapt(_gradx.shape()) << std::endl;
  std::cout << "-- C++ -- Obstacle : grady.shape = " << xt::adapt(_grady.shape()) << std::endl;
  std::cout << "-- C++ -- Obstacle : gradz.shape = " << xt::adapt(_gradz.shape()) << std::endl;

  // Visualize
  vtkSmartPointer<vtkPolyDataMapper> mapper =
  vtkSmartPointer<vtkPolyDataMapper>::New();
  mapper->SetInputConnection(reader->GetOutputPort());

  vtkSmartPointer<vtkActor> actor =
  vtkSmartPointer<vtkActor>::New();
  actor->SetMapper(mapper);

  vtkSmartPointer<vtkRenderer> renderer =
  vtkSmartPointer<vtkRenderer>::New();
  vtkSmartPointer<vtkRenderWindow> renderWindow =
  vtkSmartPointer<vtkRenderWindow>::New();
  renderWindow->AddRenderer(renderer);
  vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor =
  vtkSmartPointer<vtkRenderWindowInteractor>::New();
  renderWindowInteractor->SetRenderWindow(renderWindow);
  renderer->AddActor(actor);
  renderer->SetBackground(.3, .6, .3); // Background color green
  renderWindow->Render();
  renderWindowInteractor->Start();

}


std::string Obstacle::get_name() const {
  return _name;
}


void Obstacle::print() const{

  std::cout<<"\n-- C++ -- Obstacle : print "<<std::endl;
  // std::cout<<"\n-- C++ -- Obstacle : name = "<<_name<<std::endl;

}

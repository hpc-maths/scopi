#include "scopi/projection.hpp"
#include <typeinfo>


/// Timer used in tic & toc
auto tic_timer = std::chrono::high_resolution_clock::now();
/// Launching the timer
void tic()
{
  tic_timer = std::chrono::high_resolution_clock::now();
}
/// Stopping the timer and returning the duration in seconds
double toc()
{
  const auto toc_timer = std::chrono::high_resolution_clock::now();
  const std::chrono::duration<double> time_span = toc_timer - tic_timer;
  return time_span.count();
}


Projection::~Projection() {

}

Projection::Projection(
  const std::size_t maxiter,
  const double rho,
  const double dmin,
  const double tol,
  const double dt
) : _maxiter(maxiter), _rho(rho), _dmin(dmin), _tol(tol), _dt(dt) {

}

// xt::pyarray<double> Projection::run(
void Projection::run(
  Particles& particles,
  Contacts& contacts
) {

  xt::xtensor<double, 1> invM = 1 / xt::stack(
    xtuple(particles.data.m, particles.data.m, particles.data.m), 1
  ).reshape({3*particles.data.size(), });
  // std::cout << "-- C++ -- Projection : 1/mass = " << invM << std::endl;

  xt::xtensor<double, 1> V = xt::stack(
    xtuple(particles.data.vapx, particles.data.vapy, particles.data.vapz), 1
  ).reshape({3*particles.data.size(), });
  // std::cout << "-- C++ -- Projection : V = " << V << std::endl;

  xt::xtensor<double, 1> U = xt::stack(
    xtuple(particles.data.vx, particles.data.vy, particles.data.vz), 1
  ).reshape({3*particles.data.size(), });
  // std::cout << "-- C++ -- Projection : U = " << U << std::endl;

  tic();

  _B_coef.resize(0);
  _B_col.resize(0);
  _B_index.resize(0);

  if (contacts.data.size()>0){

    int row_ptr = 0;

    for (std::size_t ic=0 ; ic<contacts.data.size() ; ++ic){
      int i = int(contacts.data.i[ic]);
      int j = int(contacts.data.j[ic]);
      //if (contacts.data.d[ic]<=dmin){ // Si on veut enlever quelques contacts supplementaires...
      _B_index.push_back(row_ptr);
      _B_coef.push_back(contacts.data.ex[ic]); _B_col.push_back(3*i);
      _B_coef.push_back(contacts.data.ey[ic]); _B_col.push_back(3*i+1);
      _B_coef.push_back(contacts.data.ez[ic]); _B_col.push_back(3*i+2);
      row_ptr += 3;
      _B_coef.push_back(-contacts.data.ex[ic]); _B_col.push_back(3*j);
      _B_coef.push_back(-contacts.data.ey[ic]); _B_col.push_back(3*j+1);
      _B_coef.push_back(-contacts.data.ez[ic]); _B_col.push_back(3*j+2);
      row_ptr += 3;
      //}
    }
    _B_index.push_back(row_ptr);

    auto duration = toc();
    std::cout << "-- C++ -- Projection : Number of contacts : " << contacts.data.size() << std::endl;
    std::cout << "-- C++ -- Projection : CPUTIME (vectors used to create B & Bt) : " << duration << " ===> ratio per contact = "<< duration/contacts.data.size() << std::endl;

    tic();

    _descB.type = SPARSE_MATRIX_TYPE_GENERAL;
    _descB.diag = SPARSE_DIAG_NON_UNIT;
    // sparse_status_t mkl_sparse_d_create_csr (sparse_matrix_t *A,
    // sparse_index_base_t indexing,
    // MKL_INT rows,     (Number of rows)
    // MKL_INT cols,     (Number of columns)
    // MKL_INT *rows_start, (This array contains row indices)
    // MKL_INT *rows_end,
    // MKL_INT *col_indx,
    // double *values);
    //int* a = &Uzawa_row_index[0];
    //std::cout<<int(contacts.shape()[0])<<" "<<3*int(xyzr.shape()[0])<<std::endl;
    int np3 = 3*int(particles.data.size());
    _stat_B_csr = mkl_sparse_d_create_csr(
      &_mklB,
      SPARSE_INDEX_BASE_ZERO, int(contacts.data.size()), np3,
      &_B_index[0], &_B_index[0] + 1,
      &_B_col[0], &_B_coef[0]
    );
    std::cout<<"-- C++ -- Projection : _stat_B_csr = "<<_stat_B_csr<<" sparse_status_t::SPARSE_STATUS_SUCCESS = "<<sparse_status_t::SPARSE_STATUS_SUCCESS<<std::endl;
    if (_stat_B_csr==sparse_status_t::SPARSE_STATUS_SUCCESS){
      std::cout<<"-- C++ -- Projection : B_csr well computed..."<<std::endl;
    }
    duration = toc();
    std::cout << "-- C++ -- Projection : CPUTIME (B) : " << duration << " ===> ratio per contact = "<< duration/contacts.data.size() << std::endl;

    tic();
    _descBt.type = SPARSE_MATRIX_TYPE_GENERAL;
    _descBt.diag = SPARSE_DIAG_NON_UNIT;
    _stat_Bt_csr = mkl_sparse_convert_csr(_mklB,
      SPARSE_OPERATION_TRANSPOSE,
      &_mklBt
    );
    std::cout<<"-- C++ -- Projection : _stat_Bt_csr = "<<_stat_Bt_csr<<" sparse_status_t::SPARSE_STATUS_SUCCESS = "<<sparse_status_t::SPARSE_STATUS_SUCCESS<<std::endl;
    if (_stat_Bt_csr==sparse_status_t::SPARSE_STATUS_SUCCESS){
      std::cout<<"-- C++ -- Projection : Bt_csr well computed..."<<std::endl;
    }
    duration = toc();
    std::cout << "-- C++ -- Projection : CPUTIME (Bt) : " << duration << " ===> ratio per contact = "<< duration/contacts.data.size() << std::endl;

    // Uzawa algorithm

    // while (( dt*R.max()>tol*2*people[:,2].min()) and (k<nb_iter_max)):
    //    U[:] = V[:] - dt M^{-1} B.transpose()@L[:]
    //    R[:] = dt B@U[:] - (D[:]-dmin)
    //    L[:] = sp.maximum(L[:] + rho*R[:], 0)
    //    k += 1

    // The mkl_sparse_?_mv routine computes a sparse matrix-vector product defined as
    //              y := alpha*op(A)*x + beta*y
    // sparse_status_t mkl_sparse_d_mv (
    //    sparse_operation_t operation,
    //    double alpha,
    //    const sparse_matrix_t A,
    //    struct matrix_descr descr,
    //    const double *x,
    //    double beta,
    //    double *y
    // );

    tic();

    auto L = xt::zeros_like(contacts.data.d);
    auto R = xt::zeros_like(contacts.data.d);

    std::size_t cc = 0;

    double cmax = -1000.0;

    while ( (cmax<=-_tol)&&(cc <= _maxiter) ){

      // std::cout << "-- C++ -- Projection : Uzawa cc = " << cc << std::endl;

      // SCOPI :  U[:] = V[:] - dt M^{-1} B.transpose()@L[:]
      // BOOK :   U[:] = V[:] - M^{-1} B.transpose()@L[:]
      sparse_status_t infoU = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, _dt, _mklBt,  _descBt, &L[0], 0.0, &U[0]);
      U = V - U*invM; // SCOPI
      // std::cout<<"-- C++ -- Projection : U = "<<U<<std::endl;

      // SCOPI :  R[:] = dt B@U[:] - (D[:]-dmin)
      // BOOK :   R[:] = B@U[:] - (D[:]-dmin)/dt
      sparse_status_t infoR = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, _dt, _mklB,  _descB, &U[0], 0.0, &R[0]);
      R = (contacts.data.d-_dmin)-R; // SCOPI
      // std::cout<<"-- C++ -- Projection : R = "<<R<<std::endl;

      //    L[:] = sp.maximum(L[:] + rho*R[:], 0)
      L = xt::maximum( L-_rho*R, 0);
      // std::cout<<"-- C++ -- Projection : L = "<<std::endl;

      cmax = double((xt::amin(R))(0));
      // std::cout << "-- C++ -- Projection : minimal constraint : " << cmax << " < " << -_tol <<" ? " << std::endl;

      cc += 1;

    }

    duration = toc();
    std::cout << "-- C++ -- Projection : CPUTIME (Uzawa) : " << duration << " ===> ratio per contact = "<< duration/contacts.data.size() << std::endl;
    std::cout << "-- C++ -- Projection : minimal constraint : " << cmax << " < " << -_tol <<" ? " << " | nb of iterations  : "<< cc << std::endl;

    if (cc>=_maxiter){
      std::cout<<"\n-- C++ -- Projection : ********************** WARNING **********************"<<std::endl;
      std::cout<<  "-- C++ -- Projection : *************** Uzawa does not converge ***************"<<std::endl;
      std::cout<<  "-- C++ -- Projection : ********************** WARNING **********************\n"<<std::endl;
      //exit(EXIT_FAILURE);
    }

    // Update particle velocities :
    particles.data.vx = xt::view(U, xt::range(0, xt::placeholders::_, 3));
    particles.data.vy = xt::view(U, xt::range(1, xt::placeholders::_, 3));
    particles.data.vz = xt::view(U, xt::range(2, xt::placeholders::_, 3));

  }

  else {

    particles.data.vx = particles.data.vapx;
    particles.data.vy = particles.data.vapy;
    particles.data.vz = particles.data.vapz;

  }

}


void Projection::print(){

  if (_B_coef.size()>0){
    // Print matrix B
    sparse_index_base_t B_index;
    int B_rows, B_cols;
    double *B_values;
    int *B_pointerB, *B_pointerE, *B_columns;
    sparse_status_t stat_B_export_csr;
    stat_B_export_csr = mkl_sparse_d_export_csr( _mklB, &B_index, &B_rows, &B_cols, &B_pointerB,
      &B_pointerE, &B_columns, &B_values
    );
    if (stat_B_export_csr==sparse_status_t::SPARSE_STATUS_SUCCESS){
      std::cout<<"-- C++ -- Projection : export B_csr ok"<<std::endl;
    }

    std::cout<<"-- C++ -- Projection : B, nb of cols = "<<B_cols<<" nb of rows = "<<B_rows<<std::endl;
    for (int irow = 0; irow < B_rows; ++irow){
      int ncoef = B_pointerB[irow+1]-B_pointerB[irow];
      std::cout<<"-- C++ -- Projection : B, ----- i = "<<irow<<" nb of coefs = "<<ncoef<<std::endl;
      for (int pos = 0; pos < ncoef; ++pos){
        std::cout<<"-- C++ -- Projection : B, i = "<<irow<<" j = "<< B_columns[B_pointerB[irow]+pos]<<" value = "<<B_values[B_pointerB[irow]+pos]<<std::endl;
      }
    }

    // Print matrix Bt (B transpose)
    sparse_index_base_t Bt_index;
    int Bt_rows, Bt_cols;
    double *Bt_values;
    int *Bt_pointerB, *Bt_pointerE, *Bt_columns;
    mkl_sparse_d_export_csr( _mklBt, &Bt_index, &Bt_rows, &Bt_cols, &Bt_pointerB,
      &Bt_pointerE, &Bt_columns, &Bt_values );
      std::cout<<"-- C++ -- Projection : Bt, nb of cols = "<<Bt_cols<<" nb of rows = "<<Bt_rows<<std::endl;
      for (int irow = 0; irow < Bt_rows; ++irow){
        int ncoef = Bt_pointerB[irow+1]-Bt_pointerB[irow];
        std::cout<<"-- C++ -- Projection : Bt, ----- i = "<<irow<<" nb of coefs = "<<ncoef<<std::endl;
        for (int pos = 0; pos < ncoef; ++pos){
          std::cout<<"-- C++ -- Projection : Bt : i = "<<irow<<" j = "<< Bt_columns[Bt_pointerB[irow]+pos]<<" value = "<<Bt_values[Bt_pointerB[irow]+pos]<<std::endl;
        }
      }
      std::cout<<std::endl;
    }

  }

#ifndef DEF_Projection
#define DEF_Projection

#include <iostream>
#include <vector>
#include <typeinfo>

#include "mkl_service.h"
#include "mkl_spblas.h"

#include <scopi/contacts.hpp>

namespace scopi
{

  ///////////////////////////
  // Projection definition //
  ///////////////////////////
  template<std::size_t dim>
  class Projection
  {
    public:
      /// @brief Constructor
      /// Instantiate
      Projection(
        const std::size_t maxiter,
        const double rho,
        const double dmin,
        const double tol,
        const double dt
      );
      /// @brief Destructor
      ~Projection();
      /// @brief Compute new velocities
      void run(
        scopi::scopi_container<dim> &p,
        scopi::Contacts<dim> &c
      );
      /// @brief Print the contacts
      void print();
    private:
      const std::size_t _maxiter;
      const double _rho;
      const double _dmin;
      const double _tol;
      const double _dt;
      std::vector<int> _B_rowptr, _B_col;
      std::vector<double>  _B_coef;
      matrix_descr _descB;
      sparse_matrix_t _mklB;
      sparse_status_t _stat_B_csr;
      matrix_descr _descBt;
      sparse_matrix_t _mklBt;
      sparse_status_t _stat_Bt_csr;
  };

  /////////////////////////////
  // Projection implementation //
  /////////////////////////////

  template<std::size_t dim>
  Projection<dim>::~Projection()
  {
  }

  template<std::size_t dim>
  Projection<dim>::Projection(
    const std::size_t maxiter,
    const double rho,
    const double dmin,
    const double tol,
    const double dt
  ) : _maxiter(maxiter), _rho(rho), _dmin(dmin), _tol(tol), _dt(dt)
  {
  }

  template<std::size_t dim>
  void Projection<dim>::run(
    scopi::scopi_container<dim> &p,
    scopi::Contacts<dim> &c
  )
  {

    _B_rowptr.resize(0); _B_col.resize(0); _B_coef.resize(0);

    if (c.i().size()>0){
      _B_rowptr.reserve(c.i().size()+1);
      _B_col.reserve(2*dim*c.i().size());
      _B_coef.reserve(2*dim*c.i().size());
      int row_ptr = 0;
      for (std::size_t ic = 0; ic < c.i().size(); ++ic) {
        int i = c.i()[ic];
        int j = c.j()[ic];
        _B_rowptr.push_back(row_ptr);
        _B_coef.push_back(c.ex()[ic]); _B_col.push_back(3*i);
        _B_coef.push_back(c.ey()[ic]); _B_col.push_back(3*i+1);
        _B_coef.push_back(c.ez()[ic]); _B_col.push_back(3*i+2);
        row_ptr += 3;
        _B_coef.push_back(-c.ex()[ic]); _B_col.push_back(3*j);
        _B_coef.push_back(-c.ey()[ic]); _B_col.push_back(3*j+1);
        _B_coef.push_back(-c.ez()[ic]); _B_col.push_back(3*j+2);
        row_ptr += 3;
      }
      _B_rowptr.push_back(row_ptr);

      // Matrix B
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
      int np3 = 3*int(p.size());
      _stat_B_csr = mkl_sparse_d_create_csr(
        &_mklB,
        SPARSE_INDEX_BASE_ZERO, int(c.i().size()), np3,
        &_B_rowptr[0], &_B_rowptr[0] + 1,
        &_B_col[0], &_B_coef[0]
      );
      std::cout<<"\n-- C++ -- Projection : _stat_B_csr = "<<_stat_B_csr<<" sparse_status_t::SPARSE_STATUS_SUCCESS = "<<sparse_status_t::SPARSE_STATUS_SUCCESS<<std::endl;
      if (_stat_B_csr==sparse_status_t::SPARSE_STATUS_SUCCESS){
        std::cout<<"-- C++ -- Projection : B_csr well computed..."<<std::endl;
      }
      auto duration = toc();
      std::cout << "-- C++ -- Projection : CPUTIME (B) : " << duration << " ===> ratio per contact = "<< duration/c.i().size() << std::endl;

      // Matrix B transposed
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
      std::cout << "-- C++ -- Projection : CPUTIME (Bt) : " << duration << " ===> ratio per contact = "<< duration/c.i().size() << std::endl;

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
      auto invM = xt::ones<double>({3*p.size(), });
      // Les deux lignes ci-apres sont éventuellement à revoir (copies ? )
      auto U = xt::adapt(reinterpret_cast<double*>(p.v().data()), {3*p.size(), });
      auto V = xt::adapt(reinterpret_cast<double*>(p.vd().data()), {3*p.size(), });
      auto D = xt::adapt(c.d(), {c.d().size()});
      auto L = xt::zeros_like(D);
      auto R = xt::zeros_like(D);

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
        R = (D-_dmin)-R; // SCOPI
        // std::cout<<"-- C++ -- Projection : R = "<<R<<std::endl;
        //    L[:] = sp.maximum(L[:] + rho*R[:], 0)
        L = xt::maximum( L-_rho*R, 0);
        // std::cout<<"-- C++ -- Projection : L = "<<std::endl;
        cmax = double((xt::amin(R))(0));
        // std::cout << "-- C++ -- Projection : minimal constraint : " << cmax << " < " << -_tol <<" ? " << std::endl;
        cc += 1;
      }
      duration = toc();
      std::cout << "-- C++ -- Projection : CPUTIME (Uzawa) : " << duration << " ===> ratio per contact = "<< duration/c.i().size() << std::endl;
      std::cout << "-- C++ -- Projection : minimal constraint : " << cmax << " < " << -_tol <<" ? " << " | nb of iterations  : "<< cc << std::endl;
      if (cc>=_maxiter){
        std::cout<<"\n-- C++ -- Projection : ********************** WARNING **********************"<<std::endl;
        std::cout<<  "-- C++ -- Projection : *************** Uzawa does not converge ***************"<<std::endl;
        std::cout<<  "-- C++ -- Projection : ********************** WARNING **********************\n"<<std::endl;
        //exit(EXIT_FAILURE);
      }
    }
  }

}

#endif

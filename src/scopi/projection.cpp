#include "scopi/projection.hpp"
#include <typeinfo>

Projection::~Projection() {
}

Projection::Projection() {

}

xt::pyarray<double> Projection::run(
  xt::pyarray<double> &xyzr,
  xt::pyarray<double> &contacts,
  xt::pyarray<double> &V,
  xt::pyarray<double> &D,
  xt::pyarray<double> &invM,
  std::size_t maxiter,
  double rho,
  double dmin,
  double tol,
  double dt
) {


  std::cout<<"<-- C++ --> Projection::run()"<<std::endl;

  xt::pyarray<double> U = xt::full_like(invM, 0.);  // Same shape than M^{-1}, filled with 0...

  _nc = 0;

  _B_coef.resize(0);
  _B_col.resize(0);
  _B_index.resize(0);
  int row_ptr = 0;
  for (std::size_t ic=0 ; ic<contacts.shape()[0] ; ++ic){
    int i = int(contacts(ic,0));
    int j = int(contacts(ic,1));
    //std::cout<<"ic = "<<ic<<" i,j = "<<i<<","<<j<<" Dij = "<<contacts(ic,2)<<std::endl;
    //if (contacts(ic,2)<=dmin){ // Si on veut enlever quelques contacts supplementaires...
      _nc += 1;
      _B_index.push_back(row_ptr);
      _B_coef.push_back(contacts(ic,3)); _B_col.push_back(3*i);
      _B_coef.push_back(contacts(ic,4)); _B_col.push_back(3*i+1);
      _B_coef.push_back(contacts(ic,5)); _B_col.push_back(3*i+2);
      row_ptr += 3;
      _B_coef.push_back(-contacts(ic,3)); _B_col.push_back(3*j);
      _B_coef.push_back(-contacts(ic,4)); _B_col.push_back(3*j+1);
      _B_coef.push_back(-contacts(ic,5)); _B_col.push_back(3*j+2);
      row_ptr += 3;
    //}
  }
  _B_index.push_back(row_ptr);
  std::cout << "Nombre de contacts pris en compte dans la projection : " << _nc << " i.e. (Dij<=dmin) " << std::endl;
  // // auto print_data = [](const auto &val) {  std::cout << val << "\n"; };
  // // std::for_each(_B_index.begin(), _B_index.end(), print_data);
  // // std::for_each(Uzawa_coef.begin(), Uzawa_coef.end(), print_data);
  // // std::for_each(Uzawa_col.begin(),  Uzawa_col.end(), print_data);
  // // exit(0);
  //
  if (_nc>0){
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
    int np3 = 3*int(xyzr.shape()[0]);
    _stat_B_csr = mkl_sparse_d_create_csr(
      &_mklB,
      SPARSE_INDEX_BASE_ZERO, int(_nc), np3,
      &_B_index[0], &_B_index[0] + 1,
      &_B_col[0], &_B_coef[0]
    );
    //std::cout<<"_stat_B_csr = "<<_stat_B_csr<<" sparse_status_t::SPARSE_STATUS_SUCCESS = "<<sparse_status_t::SPARSE_STATUS_SUCCESS<<std::endl;
    if (_stat_B_csr==sparse_status_t::SPARSE_STATUS_SUCCESS){
      std::cout<<"B_csr ok"<<std::endl;
    }

    _descBt.type = SPARSE_MATRIX_TYPE_GENERAL;
    _descBt.diag = SPARSE_DIAG_NON_UNIT;
    _stat_Bt_csr = mkl_sparse_convert_csr(_mklB,
      SPARSE_OPERATION_TRANSPOSE,
      &_mklBt
    );
    if (_stat_Bt_csr==sparse_status_t::SPARSE_STATUS_SUCCESS){
      std::cout<<"Bt_csr ok"<<std::endl;
    }


    // algo Uzawa

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

    auto print_data = [](const auto &coef) {  std::cout << coef << "\n"; };

    //xt::pyarray<double> L({}),
    xt::pyarray<double> L = xt::zeros<double>({_nc,});
    // std::cout<<"<-- C++ --> Projection::run()  L = "<<std::endl;
    // std::for_each(L.begin(), L.end(), print_data);


    std::size_t cc = 0;
    auto R = xt::full_like(L, 0.);  // Same shape than L, filled with 0...
    double cmax = -1000.0;

    while ( (cmax<=-tol)&&(cc <= maxiter) ){

      //std::cout << "<-- C++ --> Projection::run()  cc = " << cc << std::endl;

      //    U[:] = V[:] - dt M^{-1} B.transpose()@L[:]
      sparse_status_t infoU = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, dt, _mklBt,  _descBt, &L[0], 0.0, &U[0]);
      //std::cout<<"mkl_sparse_d_mv infoU : "<<infoU<<std::endl;
      for(std::size_t pos=0 ; pos<U.shape()[0] ; ++pos){
        U(pos) = V(pos)-U(pos)*invM(pos);
      }
      // std::cout<<"<-- C++ --> Projection::run()  U = "<<std::endl;
      // std::for_each(U.begin(), U.end(), print_data);

      //    R[:] = dt B@U[:] - (D[:]-dmin)
      sparse_status_t infoR = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, dt, _mklB,  _descB, &U[0], 0.0, &R[0]);
      //std::cout<<"mkl_sparse_d_mv infoR : "<<infoR<<std::endl;
      for(std::size_t pos=0 ; pos<R.shape()[0] ; ++pos){
        R(pos) = (D(pos)-dmin)-R(pos);
      }
      // std::cout<<"<-- C++ --> Projection::run()  R = "<<std::endl;
      // std::for_each(R.begin(), R.end(), print_data);


      //    L[:] = sp.maximum(L[:] + rho*R[:], 0)
      for(std::size_t pos=0 ; pos<L.shape()[0] ; ++pos){
        L(pos) = std::max( L(pos)-rho*R(pos) ,0.0 );
      }
      // std::cout<<"<-- C++ --> Projection::run()  L = "<<std::endl;
      // std::for_each(L.begin(), L.end(), print_data);

      cmax = double((xt::amin(R))(0));
      //std::cout << "contrainte minimal : " << cmax << " < " << -tol <<" ? " << std::endl;

      cc += 1;

    }
    std::cout << "contrainte minimal : " << cmax << " < " << -tol <<" ? " << std::endl;
    std::cout<<"nb iterations  : "<< cc << std::endl;

  }

  return U;//.reshape({xyzr.shape()[0], 3});
}


void Projection::print(){

  if (_nc>0){
    // Affichage du contenu de B
    sparse_index_base_t B_index;
    int B_rows, B_cols;
    double *B_values;
    int *B_pointerB, *B_pointerE, *B_columns;
    sparse_status_t stat_B_export_csr;
    stat_B_export_csr = mkl_sparse_d_export_csr( _mklB, &B_index, &B_rows, &B_cols, &B_pointerB,
      &B_pointerE, &B_columns, &B_values
    );
    if (stat_B_export_csr==sparse_status_t::SPARSE_STATUS_SUCCESS){
      std::cout<<"export B_csr ok"<<std::endl;
    }

    std::cout<<"B : nb cols = "<<B_cols<<" nb rows = "<<B_rows<<std::endl;
    for (int irow = 0; irow < B_rows; ++irow){  // boucle sur les lignes
       int ncoef = B_pointerB[irow+1]-B_pointerB[irow];
       std::cout<<"B : ----- i = "<<irow<<" nb coefs = "<<ncoef<<std::endl;
      for (int pos = 0; pos < ncoef; ++pos){  // pour récupérer les positions des colonnes et valeurs
        std::cout<<"B : i = "<<irow<<" j = "<< B_columns[B_pointerB[irow]+pos]<<" value = "<<B_values[B_pointerB[irow]+pos]<<std::endl;
     }
    }


    // Affichage du contenu de Bt (B transposée)
    sparse_index_base_t Bt_index;
    int Bt_rows, Bt_cols;
    double *Bt_values;
    int *Bt_pointerB, *Bt_pointerE, *Bt_columns;
    mkl_sparse_d_export_csr( _mklBt, &Bt_index, &Bt_rows, &Bt_cols, &Bt_pointerB,
      &Bt_pointerE, &Bt_columns, &Bt_values );
      std::cout<<"Bt : nb cols = "<<Bt_cols<<" nb rows = "<<Bt_rows<<std::endl;
      for (int irow = 0; irow < Bt_rows; ++irow){  // boucle sur les lignes
        int ncoef = Bt_pointerB[irow+1]-Bt_pointerB[irow];
        std::cout<<"Bt : ----- i = "<<irow<<" nb coefs = "<<ncoef<<std::endl;
        for (int pos = 0; pos < ncoef; ++pos){  // pour récupérer les positions des colonnes et valeurs
          std::cout<<"Bt : i = "<<irow<<" j = "<< Bt_columns[Bt_pointerB[irow]+pos]<<" value = "<<Bt_values[Bt_pointerB[irow]+pos]<<std::endl;
        }
      }
      std::cout<<std::endl;
    }

}

#include <chrono>
#include <iostream>
#include <tuple>

#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "nanoflann/nanoflann.hpp"
#include "scopi/soa_xtensor.hpp"

#include "mkl_service.h"
#include "mkl_spblas.h"

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

struct particle
{
  double x, y, z, r;
};

SOA_DEFINE_TYPE(particle, x, y, z, r);

struct contact
{
  std::size_t i, j;
  double d, ex, ey, ez, res, lam;
};

SOA_DEFINE_TYPE(contact, i, j, d, ex, ey, ez, res, lam);


class KdTree {
public:
  KdTree(soa::vector<particle> &particles) : m_particles{particles}
  {}

    inline std::size_t kdtree_get_point_count() const
    {
      return m_particles.size();
    }

    inline double kdtree_get_pt(std::size_t idx, const std::size_t dim) const
    {
      if (dim == 0)
      return m_particles.x[idx];
      else if (dim == 1)
      return m_particles.y[idx];
      else
      return m_particles.z[idx];
    }

    template<class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const
    {
      return false;
    }

  private:
    soa::vector<particle> &m_particles;
};

  int main()
  {

    // constexpr std::size_t npart = 100000;
    //constexpr std::size_t npart = 1000000;
    constexpr std::size_t npart = 5;
    constexpr std::size_t dim = 3;

    using my_kd_tree_t = typename nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, dim
    >;

    soa::vector<particle> particles;
    particles.resize(npart);
    KdTree s(particles);

    tic();
    particles.x = xt::random::rand<double>({npart}) * 0.1;
    particles.y = xt::random::rand<double>({npart}) * 0.1;
    particles.z = xt::random::rand<double>({npart}) * 0.1;
    // particles.x = xt::random::rand<double>({npart}) * 10;
    // particles.y = xt::random::rand<double>({npart}) * 10;
    // particles.z = xt::random::rand<double>({npart}) * 10;
    particles.r.fill(0.1);
    // particles.r.fill(0.01);
    auto duration = toc();
    std::cout << "fill pos and radius: " << duration << "\n";

    tic();
    my_kd_tree_t index(
      3 /*dim*/, s,
      nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */)
    );
      index.buildIndex();
      duration = toc();
      std::cout << "build tree: " << duration << "\n";

      tic();
      // for (auto particle : particles)
      // {
      //     double query_pt[3] = {particle.x, particle.y, particle.z};


      // Contact list and Uzawa Matrix

      // Uzawa matrix : B =
      //
      //                 <---------- dim * particle_number columns ----------->
      //       |                       i                       j
      //   constraint   0 ... ... 0   eij    0 ... ... 0     -eij      0 ... ... 0
      //    number                  <-dim->                 <-dim->
      //     rows       0 ... ... 0 ex ey ez 0 ... ... 0  -ex -ey -ez  0 ... ... 0
      //       |
      //

      // Uzawa matrix : B^t =
      //
      //                 <---------- constraint number columns ----------->
      //       |                   0
      //       |                   :
      //       |                   0      ∧
      //       |             i    eij    dim
      //      dim                  0      ∨
      // *particule_number         :
      //     rows                  0      ∧
      //       |             j   -eij    dim
      //       |                   0      ∨
      //       |                   :
      //       |                   0

      std::size_t cc = 0;
      std::vector<double> Uzawa_coef;
      std::vector<int> Uzawa_col, Uzawa_row_index;

      // Attention le resize de contacts efface le contenu...
      // les tableaux temporaires ci-dessous seront supprimés quand on pourra faire contacts.push_back()
      std::vector<int> contacts_i, contacts_j;
      std::vector<double> contacts_d, contacts_ex, contacts_ey, contacts_ez, contacts_res, contacts_lam;

      int row_ptr = 0;

      for (std::size_t i = 0; i < npart; ++i)
      {
        double query_pt[3] = {particles.x[i], particles.y[i], particles.z[i]};
        // Unsorted radius search:
        // const double radius = .02;
        const double radius = .005;
        std::vector<std::pair<size_t, double>> indices_dists;
        nanoflann::RadiusResultSet<double, std::size_t> resultSet(
          radius, indices_dists);

          std::vector<std::pair<unsigned long, double>> ret_matches;
          const std::size_t nMatches = index.radiusSearch(query_pt, radius, ret_matches,
            nanoflann::SearchParams());

            //std::cout << "pt " << i << " nMatches = " << nMatches << std::endl;

            for (std::size_t ic = 0; ic < nMatches; ++ic)
            {
              std::size_t j = ret_matches[ic].first;
              //double dist = ret_matches[ic].second;
              if (i < j){

                //std::cout<<"contact : i = "<<i<<" j = "<<j<<std::endl;

                double ex = particles.x[j]-particles.x[i];
                double ey = particles.y[j]-particles.y[i];
                double ez = particles.z[j]-particles.z[i];
                double norm = sqrt(ex*ex+ey*ey+ez*ez);

                Uzawa_row_index.push_back(row_ptr);
                Uzawa_coef.push_back(ex/norm); Uzawa_col.push_back(3*i);
                Uzawa_coef.push_back(ey/norm); Uzawa_col.push_back(3*i+1);
                Uzawa_coef.push_back(ez/norm); Uzawa_col.push_back(3*i+2);
                row_ptr += 3;

                Uzawa_coef.push_back(-ex/norm); Uzawa_col.push_back(3*j);
                Uzawa_coef.push_back(-ey/norm); Uzawa_col.push_back(3*j+1);
                Uzawa_coef.push_back(-ez/norm); Uzawa_col.push_back(3*j+2);
                row_ptr += 3;

                // pas de contacts.push_back()
                //std::cout<<i<<","<<j<<std::endl;
                contacts_i.push_back(i);
                contacts_j.push_back(j);
                contacts_d.push_back(norm-particles.r[i]-particles.r[j]);
                contacts_ex.push_back(ex/norm);
                contacts_ey.push_back(ey/norm);
                contacts_ez.push_back(ez/norm);
                contacts_res.push_back(0.0);
                contacts_lam.push_back(0.0);
                //contacts.push_back({i, j, norm-particles.r[i]-particles.r[j], ex/norm, ey/norm, ez/norm, 0.0, 0.0});
                cc += 1;
              }
            }
          }
          Uzawa_row_index.push_back(row_ptr);

          // pas super mais comme on n'a pas de push_back...
          soa::vector<contact> contacts;
          std::cout<<"Nb de contacts : "<<contacts_i.size()<<std::endl;
          contacts.resize(contacts_i.size());
          for (size_t id=0; id<contacts_i.size(); ++id) {
            contacts.i[id] = contacts_i[id];  contacts.j[id] = contacts_j[id];
            contacts.d[id] = contacts_d[id];
            contacts.ex[id] = contacts_ey[id];
            contacts.ey[id] = contacts_ey[id];
            contacts.ez[id] = contacts_ez[id];
            contacts.res[id] = contacts_res[id];
            contacts.lam[id] = contacts_lam[id];
          }


          auto print_contacts = [](const auto &contact) {  std::cout << " contacts : " << contact.i << " " << contact.j  << " " << contact.d  << " " << contact.ex << " " << contact.ey << " " << contact.ez << " " << contact.res << " " << contact.lam << "\n"; };
          std::for_each(contacts.begin(), contacts.end(), print_contacts);
          std::cout << "\n";

          tic();
          matrix_descr descB;
          descB.type = SPARSE_MATRIX_TYPE_GENERAL;
          descB.diag = SPARSE_DIAG_NON_UNIT;
          sparse_matrix_t mklB;
          // sparse_status_t mkl_sparse_d_create_csr (sparse_matrix_t *A,
          // sparse_index_base_t indexing,
          // MKL_INT rows,     (Number of rows)
          // MKL_INT cols,     (Number of columns)
          // MKL_INT *rows_start, (This array contains row indices)
          // MKL_INT *rows_end,
          // MKL_INT *col_indx,
          // double *values);
          //int* a = &Uzawa_row_index[0];
          sparse_status_t stat_B_csr = mkl_sparse_d_create_csr(
            &mklB,
            SPARSE_INDEX_BASE_ZERO, contacts.size(), 3*particles.size(),
            &Uzawa_row_index[0], &Uzawa_row_index[0] + 1,
            &Uzawa_col[0], &Uzawa_coef[0]
          );
          duration = toc();
          std::cout << "create_B_csr: " << duration << "\n";
          // Affichage du contenu de B
          sparse_index_base_t B_index;
          int B_rows, B_cols;
          double *B_values;
          int *B_pointerB, *B_pointerE, *B_columns;
          mkl_sparse_d_export_csr( mklB, &B_index, &B_rows, &B_cols, &B_pointerB,
            &B_pointerE, &B_columns, &B_values
          );
          std::cout<<"B : nb cols = "<<B_cols<<" nb rows = "<<B_rows<<std::endl;
          for (int irow = 0; irow < B_rows; ++irow){  // boucle sur les lignes
            int ncoef = B_pointerB[irow+1]-B_pointerB[irow];
            std::cout<<"B : ----- i = "<<irow<<" nb coefs = "<<ncoef<<std::endl;
            for (int pos = 0; pos < ncoef; ++pos){  // pour récupérer les positions des colonnes et valeurs
              std::cout<<"B : i = "<<irow<<" j = "<< B_columns[B_pointerB[irow]+pos]<<" value = "<<B_values[B_pointerB[irow]+pos]<<std::endl;
            }
          }

          tic();
          matrix_descr descBt;
          descBt.type = SPARSE_MATRIX_TYPE_GENERAL;
          descBt.diag = SPARSE_DIAG_NON_UNIT;
          sparse_matrix_t mklBt;
          sparse_status_t stat_Bt_csr = mkl_sparse_convert_csr(mklB,
            SPARSE_OPERATION_TRANSPOSE,
            &mklBt
          );
          duration = toc();
          std::cout << "create_Bt_csr: " << duration << "\n";
          std::cout<<std::endl;
          // Affichage du contenu de Bt (B transposée)
          sparse_index_base_t Bt_index;
          int Bt_rows, Bt_cols;
          double *Bt_values;
          int *Bt_pointerB, *Bt_pointerE, *Bt_columns;
          mkl_sparse_d_export_csr( mklBt, &Bt_index, &Bt_rows, &Bt_cols, &Bt_pointerB,
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


            //
            //     auto print_particles = [](const auto &particle) { std::cout << " particles : "<< particle.x << " " << particle.y  << " " << particle.z  << " " << particle.r << "\n"; };
            //     std::for_each(particles.begin(), particles.end(), print_particles);
            //     std::cout << "\n";
            //
            //
            //     duration = toc();
            //     std::cout << "find neighbors: " << duration << "\n";
            //
            //
            //
            //     // inefficace... MKL_Set_Num_Threads_Local(1);
            //     // inefficace... MKL_Set_Num_Threads(1);
            //
            //     std::vector<double> X(3*particles.size(), 0.0);
            //     std::vector<double> Y(contacts.size(), 0.0);
            //
            //
            //     tic();
            //     // Computes a sparse matrix-vector product :   y := alpha*op(A)*x + beta*y
            //     // sparse_status_t mkl_sparse_d_mv (sparse_operation_t operation,
            //     // double alpha,
            //     // const sparse_matrix_t A,
            //     // struct matrix_descr descr,
            //     // const double *x,
            //     // double beta,
            //     // double *y);
            //     //mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1, mklA,  SPARSE_MATRIX_TYPE_GENERAL, X, 0, Y);
            //     matrix_descr descB;
            //     descB.type = SPARSE_MATRIX_TYPE_GENERAL;
            //     descB.diag = SPARSE_DIAG_NON_UNIT;
            //     sparse_matrix_t mklB;
            //     // sparse_status_t mkl_sparse_d_create_csr (sparse_matrix_t *A,
            //     // sparse_index_base_t indexing,
            //     // MKL_INT rows,     (Number of rows)
            //     // MKL_INT cols,     (Number of columns)
            //     // MKL_INT *rows_start, (This array contains row indices)
            //     // MKL_INT *rows_end,
            //     // MKL_INT *col_indx,
            //     // double *values);
            //     //int* a = &Uzawa_row_index[0];
            //     sparse_status_t stat_B_csr = mkl_sparse_d_create_csr(
            //         &mklB,
            //         SPARSE_INDEX_BASE_ZERO, contacts.size(), 3*particles.size(),
            //         &Uzawa_row_index[0], &Uzawa_row_index[0] + 1,
            //         &Uzawa_col[0], &Uzawa_coef[0]);
            //     duration = toc();
            //     std::cout << "create_B_csr: " << duration << "\n";
            //
            //
            //     tic();
            //     matrix_descr descBt;
            //     descBt.type = SPARSE_MATRIX_TYPE_GENERAL;
            //     descBt.diag = SPARSE_DIAG_NON_UNIT;
            //     sparse_matrix_t mklBt;
            //     sparse_status_t stat_Bt_csr = mkl_sparse_convert_csr(mklB,
            //       SPARSE_OPERATION_TRANSPOSE,
            //       &mklBt
            //     );
            //     duration = toc();
            //     std::cout << "create_Bt_csr: " << duration << "\n";
            //
            //
            //     // // Affichage du contenu de B
            //     // sparse_index_base_t B_index;
            //     // int B_rows, B_cols;
            //     // double *B_values;
            //     // int *B_pointerB, *B_pointerE, *B_columns;
            //     // mkl_sparse_d_export_csr( mklB, &B_index, &B_rows, &B_cols, &B_pointerB,
            //     //   &B_pointerE, &B_columns, &B_values );
            //     // std::cout<<"B : nb cols = "<<B_cols<<" nb rows = "<<B_rows<<std::endl;
            //     // for (int irow = 0; irow < B_rows; ++irow){  // boucle sur les lignes
            //     //   int ncoef = B_pointerB[irow+1]-B_pointerB[irow];
            //     //   std::cout<<"B : ----- i = "<<irow<<" nb coefs = "<<ncoef<<std::endl;
            //     //   for (int pos = 0; pos < ncoef; ++pos){  // pour récupérer les positions des colonnes et valeurs
            //     //     std::cout<<"B : i = "<<irow<<" j = "<< B_columns[B_pointerB[irow]+pos]<<" value = "<<B_values[B_pointerB[irow]+pos]<<std::endl;
            //     //   }
            //     // }
            //
            //     // std::cout<<std::endl;
            //     // // Affichage du contenu de Bt (B transposée)
            //     // sparse_index_base_t Bt_index;
            //     // int Bt_rows, Bt_cols;
            //     // double *Bt_values;
            //     // int *Bt_pointerB, *Bt_pointerE, *Bt_columns;
            //     // mkl_sparse_d_export_csr( mklBt, &Bt_index, &Bt_rows, &Bt_cols, &Bt_pointerB,
            //     //   &Bt_pointerE, &Bt_columns, &Bt_values );
            //     // std::cout<<"Bt : nb cols = "<<Bt_cols<<" nb rows = "<<Bt_rows<<std::endl;
            //     // for (int irow = 0; irow < Bt_rows; ++irow){  // boucle sur les lignes
            //     //   int ncoef = Bt_pointerB[irow+1]-Bt_pointerB[irow];
            //     //   std::cout<<"Bt : ----- i = "<<irow<<" nb coefs = "<<ncoef<<std::endl;
            //     //   for (int pos = 0; pos < ncoef; ++pos){  // pour récupérer les positions des colonnes et valeurs
            //     //     std::cout<<"Bt : i = "<<irow<<" j = "<< Bt_columns[Bt_pointerB[irow]+pos]<<" value = "<<Bt_values[Bt_pointerB[irow]+pos]<<std::endl;
            //     //   }
            //     // }
            //     // std::cout<<std::endl;
            //
            //     // Construction de la matrice csr à partir de la matrice coo : LENT !
            //     // tic();
            //     // matrix_descr descB_coo;
            //     // descB_coo.type = SPARSE_MATRIX_TYPE_GENERAL;
            //     // descB_coo.diag = SPARSE_DIAG_NON_UNIT;
            //     // sparse_matrix_t mklB_coo;
            //     // // sparse_status_t mkl_sparse_d_create_coo (sparse_matrix_t *A,
            //     // // sparse_index_base_t indexing,
            //     // // MKL_INT rows,     (Number of rows)
            //     // // MKL_INT cols,     (Number of columns)
            //     // // MKL_INT nnz,      (Number of non-zero elements )
            //     // // MKL_INT *row_indx, (This array contains row indices)
            //     // // MKL_INT *rows_end,
            //     // // MKL_INT *col_indx,
            //     // // double *values);
            //     // //int* a = &Uzawa_row_index[0];
            //     // sparse_status_t stat_B_coo = mkl_sparse_d_create_coo(
            //     //     &mklB_coo,
            //     //     SPARSE_INDEX_BASE_ZERO, contacts.size(), 3*particles.size(),
            //     //     2*3*contacts.size(),
            //     //     &Uzawa_row[0], &Uzawa_col[0], &Uzawa_coef[0]);
            //     // duration = toc();
            //     // std::cout << "create_B_coo: " << duration << "\n";
            //     // tic();
            //     // sparse_status_t stat_B_coo_to_csr = mkl_sparse_convert_csr(mklB_coo,
            //     //   SPARSE_OPERATION_NON_TRANSPOSE, &mklB_coo);
            //     // duration = toc();
            //     // std::cout << "create_B_coo_to_csr: " << duration << "\n";
            //
            //
            //     // Test performance produit matrice vecteur
            //     // tic();
            //     // int Nmv = 10;
            //     // for (int it=0;it<Nmv;it++){
            //     //   sparse_status_t info = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, mklB,  descB, &X[0], 0.0, &Y[0]);
            //     //   //std::cout<< info << std::endl;
            //     // }
            //     // duration = toc();
            //     // std::cout << ""<< Nmv <<" matrix vector products: " << duration << "\n";
            //     // std::cout << "One matrix vector product: " << duration/Nmv << "\n";
            //
            //     // algo Uzawa
            //
            //     // while (( dt*R.max()>tol*2*people[:,2].min()) and (k<nb_iter_max)):
            //     //           U[:] = V[:] - B.transpose()@L[:]
            //     //           R[:] = B@U[:] - (D[:]-dmin)/dt
            //     //           L[:] = sp.maximum(L[:] + rho*R[:], 0)
            //     //           k += 1
            //
            //     std::vector<double> V(3*npart,0);
            //     std::vector<double> U(3*npart,0);
            //
            //     int nb_iter_max = 10;
            //     double tol = 0.001;
            //     double dt = 0.01;
            //     double dmin = 0.0;
            //
            //     double rmax = 0;
            //     double rmin = 1.0e99;
            //     for (auto it = particles.r.begin(); it != particles.r.end(); ++it){
            //       if (rmax < *it) rmax = *it;
            //       if (rmin > *it) rmin = *it;
            //     }
            //     std::cout<<"rmax = "<<rmax<<" rmin = "<<rmin<<std::endl;
            //
            //     int iter = 0;
            //     double resmax = 1.0e99;
            //     while ((dt*resmax>tol*2*rmin) && (iter < nb_iter_max)) {
            //       std::cout<<"iter = "<<iter<<std::endl;
            //       // U = V - B^t@L
            //       //
            //       // The mkl_sparse_?_mv routine computes a sparse matrix-vector product defined as
            //       //              y := alpha*op(A)*x + beta*y
            //       // sparse_status_t mkl_sparse_d_mv (
            //       //    sparse_operation_t operation,
            //       //    double alpha,
            //       //    const sparse_matrix_t A,
            //       //    struct matrix_descr descr,
            //       //    const double *x,
            //       //    double beta,
            //       //    double *y
            //       //);
            //       U = V;
            //       sparse_status_t info = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
            //         -1.0,
            //         mklBt,
            //         descBt,
            //         &contacts.lam[0],
            //         1.0,
            //         &U[0]);
            //       // R = B@U - (D-dmin)/dt
            //       //
            //
            //       info = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
            //         1.0,
            //         mklB,
            //         descB,
            //         &U[0],
            //         0.0,
            //         &contacts.res[0]);
            //       //contacts.res = contacts.res - (contacts.d - dmin)/dt;
            //
            //
            //       iter +=1;
            //     }
            //
            //     // https://xtensor.readthedocs.io/en/latest/numpy.html
            //     //rmax = xt::amax(particles.r);
            //
            //     // auto result = std::max_element(particles.r.begin(), particles.r.end());
            //     // double rmax = std::distance(particles.r.begin(), result);
            //     // std::cout<<"result = "<<result<<" rmax = "<<rmax<<std::endl;
            //
            //     // auto print = [](const double& value) { std::cout << " " << value; };
            //     //
            //     // std::cout << "Y:";
            //     // std::for_each(Y.begin(), Y.end(), print);
            //     // std::cout << '\n';
            //


          }

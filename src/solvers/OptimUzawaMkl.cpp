#include "scopi/solvers/OptimUzawaMkl.hpp"

#ifdef SCOPI_USE_MKL

namespace scopi
{
    OptimUzawaMkl::~OptimUzawaMkl()
    {
        m_status = mkl_sparse_destroy ( m_A );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix A: " << m_status;
        m_status = mkl_sparse_destroy ( m_inv_P );
        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_destroy for matrix P^-1: " << m_status;
    }

    void OptimUzawaMkl::print_csr_matrix(const sparse_matrix_t A)
    {
        MKL_INT* csr_row_begin_ptr = NULL;
        MKL_INT* csr_row_end_ptr = NULL;
        MKL_INT* csr_col_ptr = NULL;
        double* csr_val_ptr = NULL;
        sparse_index_base_t indexing;
        MKL_INT nbRows;
        MKL_INT nbCols;
        m_status = mkl_sparse_d_export_csr(A,
                                           &indexing,
                                           &nbRows,
                                           &nbCols,
                                           &csr_row_begin_ptr,
                                           &csr_row_end_ptr,
                                           &csr_col_ptr,
                                           &csr_val_ptr);

        PLOG_ERROR_IF(m_status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_export_csr: " << m_status;

        std::cout << "\nMatrix with " << nbRows << " rows and " << nbCols << " columns\n";
        std::cout << "RESULTANT MATRIX:\nrow# : (column, value) (column, value)\n";
        int ii = 0;
        for( int i = 0; i < nbRows; i++ )
        {
            std::cout << "row#" << i << ": ";
            for(MKL_INT j = csr_row_begin_ptr[i]; j < csr_row_end_ptr[i]; j++ )
            {
                std::cout << " (" << csr_col_ptr[ii] << ", " << csr_val_ptr[ii] << ")";
                ii++;
            }
            std::cout << std::endl;
        }
        std::cout << "_____________________________________________________________________  \n" ;
    }

    void OptimUzawaMkl::set_moment_matrix_impl(std::size_t nparts,
                           std::vector<MKL_INT>& invP_csr_row,
                           std::vector<MKL_INT>& invP_csr_col,
                           std::vector<double>& invP_csr_val,
                           const scopi_container<2>& particles)
    {
        for (std::size_t i = 0; i < nparts; ++i)
        {
            invP_csr_row.push_back(3*nparts + 3*i + 2);
            invP_csr_col.push_back(3*nparts + 3*i + 2);
            invP_csr_val.push_back(1./particles.j()(active_offset + i));
        }
    }

    void OptimUzawaMkl::set_moment_matrix_impl(std::size_t nparts,
                           std::vector<MKL_INT>& invP_csr_row,
                           std::vector<MKL_INT>& invP_csr_col,
                           std::vector<double>& invP_csr_val,
                           const scopi_container<3>& particles)
    {
        for (std::size_t i = 0; i < nparts; ++i)
        {
            for (std::size_t d = 0; d < 3; ++d)
            {
                invP_csr_row.push_back(3*nparts + 3*i + d);
                invP_csr_col.push_back(3*nparts + 3*i + d);
                invP_csr_val.push_back(1./particles.j()(active_offset + i)(d));
            }
        }
    }
}
#endif

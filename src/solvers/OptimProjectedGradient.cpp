#include "scopi/solvers/OptimProjectedGradient.hpp"

#ifdef SCOPI_USE_MKL
namespace scopi
{
    void print_csr_matrix(const sparse_matrix_t A)
    {
        MKL_INT* csr_row_begin = NULL;
        MKL_INT* csr_row_end   = NULL;
        MKL_INT* csr_col       = NULL;
        double* csr_val        = NULL;
        sparse_index_base_t indexing;
        MKL_INT nbRows;
        MKL_INT nbCols;
        auto status = mkl_sparse_d_export_csr(A, &indexing, &nbRows, &nbCols, &csr_row_begin, &csr_row_end, &csr_col, &csr_val);

        PLOG_ERROR_IF(status != SPARSE_STATUS_SUCCESS) << "Error in mkl_sparse_d_export_csr: " << status;

        std::cout << "\nMatrix with " << nbRows << " rows and " << nbCols << " columns\n";
        std::cout << "RESULTANT MATRIX:\nrow# : (column, value) (column, value)\n";
        int ii = 0;
        for (int i = 0; i < nbRows; i++)
        {
            std::cout << "row#" << i << ": ";
            for (MKL_INT j = csr_row_begin[i]; j < csr_row_end[i]; j++)
            {
                std::cout << " (" << csr_col[ii] << ", " << csr_val[ii] << ")";
                ii++;
            }
            std::cout << std::endl;
        }
        std::cout << "_____________________________________________________________________  \n";
    }
}
#endif

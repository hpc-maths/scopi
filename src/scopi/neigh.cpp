#include "scopi/neigh.hpp"

using namespace tbb;

// void Neigh::operator() ( const blocked_range<std::size_t>& r ) const {
//   for ( std::size_t i = r.begin(); i != r.end(); ++i ) {
//     std::size_t max_size = 0, max_pos = 0;
//     for (std::size_t j = 0; j < str.size(); ++j)
//     if (j != i) {
//       size_t limit = str.size()-std::max(i,j);
//       for (size_t k = 0; k < limit; ++k) {
//         if (str[i + k] != str[j + k]) break;
//         if (k > max_size) {
//           max_size = k;
//           max_pos = j;
//         }
//       }
//     }
//     max_array[i] = max_size;
//     pos_array[i] = max_pos;
//   }
// }
//
// Neigh::Neigh(std::string &s, std::size_t *m, std::size_t *p) :
// str(s), max_array(m), pos_array(p)
// {
//
// }


Neigh::Neigh(std::vector<std::size_t> &_val, concurrent_vector< std::pair< std::size_t, std::size_t > > &_cv ) : cv( _cv ), val( _val ) {}

void Neigh::operator( )( const blocked_range<std::size_t>& r ) const {

  //printf("Decoupage : %08d - %08d\n",r.begin(),r.end());

  for ( std::size_t i=r.begin(); i!=r.end( ); ++i ) {
    for ( std::size_t j=i+1; j<N; ++j ){
      if (val[j]-val[i]<6) {
        // printf("val : %08d - %08d\n",val[i],val[j]);
        cv.push_back(std::make_pair(i,j));
      }
    }
  }
}

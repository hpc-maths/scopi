#ifndef DEF_Neigh
#define DEF_Neigh

#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#include <tbb/task_scheduler_init.h>
#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>
#include <tbb/concurrent_vector.h>
#include <tbb/blocked_range.h>

using namespace tbb;

static const std::size_t N = 200000;

// class Neigh {
//   const std::string str;
//   std::size_t *max_array;
//   std::size_t *pos_array;
// public:
//   void operator() ( const blocked_range<std::size_t>& r ) const;
//   Neigh(std::string &s, std::size_t *m, std::size_t *p);
// };

class Neigh {
    private:
        concurrent_vector< std::pair< std::size_t, std::size_t > > &cv;
        std::vector<std::size_t> &val;
    public:
        Neigh(std::vector<std::size_t> &_val, concurrent_vector< std::pair< std::size_t, std::size_t > > &_cv );
        void operator( )( const blocked_range<std::size_t>& r ) const;
};

#endif

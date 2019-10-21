#include <chrono>
#include <iostream>
#include <tuple>

#include <xtensor/xio.hpp>
#include <xtensor/xrandom.hpp>

#include "nanoflann/nanoflann.hpp"
#include "scopi/soa_xtensor.hpp"

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
    double x, y, z;
    double radius;
};

SOA_DEFINE_TYPE(particle, x, y, z, radius);

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
    using my_kd_tree_t = typename nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, KdTree>, KdTree, 3 /* dim */
        >;

    constexpr std::size_t npart = 1000000;
    constexpr std::size_t dim = 3;
    soa::vector<particle> particles;
    particles.resize(npart);
    KdTree s(particles);

    tic();
    particles.x = xt::random::rand<double>({npart}) * 10;
    particles.y = xt::random::rand<double>({npart}) * 10;
    particles.z = xt::random::rand<double>({npart}) * 10;
    particles.radius.fill(0.1);
    auto duration = toc();
    std::cout << "fill pos and radius: " << duration << "\n";

    tic();
    my_kd_tree_t index(
        3 /*dim*/, s,
        nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();
    duration = toc();
    std::cout << "build tree: " << duration << "\n";

    tic();
    // for (auto particle : particles)
    // {
    //     double query_pt[3] = {particle.x, particle.y, particle.z};

    for (std::size_t i = 0; i < npart; ++i)
    {
        double query_pt[3] = {particles.x[i], particles.y[i], particles.z[i]};
        // Unsorted radius search:
        const double radius = .02;
        std::vector<std::pair<size_t, double>> indices_dists;
        nanoflann::RadiusResultSet<double, std::size_t> resultSet(
            radius, indices_dists);

        // const int nMatches = index.findNeighbors(resultSet, query_pt,
        // nanoflann::SearchParams());
        std::vector<std::pair<unsigned long, double>> ret_matches;
        const int nMatches = index.radiusSearch(query_pt, radius, ret_matches,
                                                nanoflann::SearchParams());

        //std::cout << "pt " << i << " nMatches = " << nMatches << std::endl;

        // index.findNeighbors(resultSet, query_pt,
        // nanoflann::SearchParams());

        // // Get worst (furthest) point, without sorting:
        // std::pair<size_t,double> worst_pair = resultSet.worst_item();
        // // std::cout << "Worst pair: idx=" << worst_pair.first << "
        // dist=" << worst_pair.second << std::endl;
    }
    duration = toc();
    std::cout << "find neighbors: " << duration << "\n";
}

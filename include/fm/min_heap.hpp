#include <iostream>
#include <vector>
#include <iterator>
#include <cmath>

using namespace std;

struct addist
{
  double dist;
  std::size_t  i, j, k;
};

class Heap
{
public:
  Heap();
  ~Heap();
  void insert(double dist, std::size_t i, std::size_t j, std::size_t k);
  addist deletemin();
  void print();
  std::size_t size()
  {
    return heap.size();
  }

private:
  std::size_t left(std::size_t parent);
  std::size_t right(std::size_t parent);
  std::size_t parent(std::size_t child);
  void heapifyup  (std::size_t index);
  void heapifydown(std::size_t index);
  vector<addist> heap;
};

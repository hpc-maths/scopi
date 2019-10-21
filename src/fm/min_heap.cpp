#include <iostream>
#include <vector>
#include <iterator>
#include <cmath>

#include "fm/min_heap.hpp"

using namespace std;

const std::size_t Infty = std::numeric_limits<std::size_t>::max();

Heap::Heap() { }

Heap::~Heap() { }

void Heap::insert(double dist, std::size_t i, std::size_t j, std::size_t k)
{
  addist rec;
  rec.dist = dist;
  rec.i = i;
  rec.j = j;
  rec.k = k;
  heap.push_back(rec);
  heapifyup(heap.size() - 1);
}


addist Heap::deletemin()
{
  addist min = heap.front();
  heap[0] = heap.at(heap.size() - 1);
  heap.pop_back();
  heapifydown(0);
  return min;
}

void Heap::print()
{
  vector<addist>::iterator pos = heap.begin();
  cout << "-- C++ -- Heap = \n";
  while ( pos != heap.end() )
  {
    cout << (*pos).dist << ": ("<< (*pos).i << "," << (*pos).j << "," << (*pos).k << ") \n"; ++pos;
  }
  cout << endl;
}

void Heap::heapifyup(std::size_t index)
{
  while ( (index != Infty) && ( index > 0 ) && ( parent(index) >= 0 ) && ( heap[parent(index)].dist > heap[index].dist ) )
  {
    addist tmp = heap[parent(index)];
    heap[parent(index)] = heap[index];
    heap[index] = tmp;
    index = parent(index);
  }
}

void Heap::heapifydown(std::size_t index)
{
  std::size_t child = left(index);

  if ( ( child > 0 ) && (index != Infty) && ( right(index) > 0 ) && ( heap[child].dist > heap[right(index)].dist ) )
  {
    child = right(index);
  }
  //if ( child > 0 )
  if ( (child >0) && (index < Infty))
  {
    // On prend le fils qui a la valeur la plus grande et on permutte avec le pere
    addist tmp = heap[index];
    heap[index] = heap[child];
    heap[child] = tmp;
    heapifydown(child);
  }
}

std::size_t Heap::left(std::size_t parent)
{
  std::size_t i = ( parent << 1 ) + 1; // 2 * parent + 1
  //return ( i < heap.size() ) ? i : -1;
  return ( i < heap.size() ) ? i : Infty;
}

std::size_t Heap::right(std::size_t parent)
{
  std::size_t i = ( parent << 1 ) + 2; // 2 * parent + 2
  //return ( i < heap.size() ) ? i : -1;
  return ( i < heap.size() ) ? i : Infty;
}

std::size_t Heap::parent(std::size_t child)
{
  if (child != 0)
  {
    std::size_t i = (child - 1) >> 1;
    return i;
  }
  //return -1;
  return Infty;
}

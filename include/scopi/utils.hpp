/////////////////////////////
// Functions for the timer //
/////////////////////////////

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

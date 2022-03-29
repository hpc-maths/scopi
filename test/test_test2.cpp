#include "doctest/doctest.h"

int factorial_2(int number) { return number <= 1 ? number : factorial_2(number - 1) * number; }

TEST_CASE("testing the factorial function") {
    CHECK(factorial_2(1) == 0);
    CHECK(factorial_2(2) == 2);
    CHECK(factorial_2(3) == 6);
    CHECK(factorial_2(10) == 3628800);
}


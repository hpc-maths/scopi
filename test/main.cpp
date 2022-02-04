#include <plog/Log.h>
#include "plog/Initializers/RollingFileInitializer.h"
#include <gtest/gtest.h>

int main(int argc, char* argv[])
{
    ::testing::InitGoogleTest(&argc, argv);
    plog::init(plog::none);
    return RUN_ALL_TESTS();
}

#define DOCTEST_CONFIG_IMPLEMENT
#include "doctest/doctest.h"

#include "plog/Initializers/RollingFileInitializer.h"
#include <plog/Log.h>

int main(int argc, char** argv)
{
    plog::init(plog::none, "tests.log");

    doctest::Context context;

    context.applyCommandLine(argc, argv);

    // overrides
    context.setOption("no-breaks", true); // don't break in the debugger when assertions fail

    int res = context.run(); // run

    if (context.shouldExit())
    {               // important - query flags (and --exit) rely on the user doing this
        return res; // propagate the result of the tests
    }

    int client_stuff_return_code = 0;
    // your program - if the testing framework is integrated in your production code

    return res + client_stuff_return_code; // the result from doctest is propagated here as well
}

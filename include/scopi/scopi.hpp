#pragma once

#include <CLI/CLI.hpp>

#define SCOPI_PARSE(argc, argv)       \
    try                               \
    {                                 \
        auto& app = scopi::get_app(); \
        app.parse(argc, argv);        \
    }                                 \
    catch (const CLI::ParseError& e)  \
    {                                 \
        auto& app = scopi::get_app(); \
        return app.exit(e);           \
    }

namespace scopi
{

    inline auto& get_app()
    {
        static CLI::App app;
        app.set_help_all_flag("--help-all", "Expand all help");
        return app;
    }

    inline auto& initialize(std::string description = "")
    {
        auto& app = get_app();
        app.description(description);
        return app;
    }
}

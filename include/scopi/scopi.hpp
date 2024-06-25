#pragma once

#include <CLI/CLI.hpp>

#include <plog/Log.h>

static plog::Severity log_level = plog::none;

#define SCOPI_PARSE(argc, argv)                                                     \
    try                                                                             \
    {                                                                               \
        auto& app = scopi::get_app();                                               \
        app.parse(argc, argv);                                                      \
        plog::init(log_level, fmt::format("{}.log", app.get_description()).data()); \
    }                                                                               \
    catch (const CLI::ParseError& e)                                                \
    {                                                                               \
        auto& app = scopi::get_app();                                               \
        return app.exit(e);                                                         \
    }

namespace scopi
{

    inline auto& get_app()
    {
        static CLI::App app;
        app.set_help_all_flag("--help-all", "Expand all help");
        return app;
    }

    inline auto& initialize(const std::string& description = "")
    {
        auto& app = get_app();

        std::map<std::string, plog::Severity> map{
            {"none",  plog::none },
            {"info",  plog::info },
            {"debug", plog::debug}
        };
        app.add_option("--log-level", log_level, "Log level")->capture_default_str()->transform(CLI::CheckedTransformer(map, CLI::ignore_case));

        if (description.empty())
        {
            app.description("scopi");
        }
        else
        {
            app.description(description);
        }

        return app;
    }
}

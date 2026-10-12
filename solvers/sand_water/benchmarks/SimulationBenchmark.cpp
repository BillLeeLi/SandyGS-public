#include "scene/Scene.h"
#include "simulation/Simulation.h"

#include <cxxopts.hpp>

namespace
{
    using namespace Pivot;

    /// 多步、多次重复的累计结果；所有阶段时间均来自 Simulation 内部计时。
    struct BenchmarkSummary
    {
        SimulationStepTimings timings;
        double outerWallSeconds{0.0};
        std::uint64_t measuredSteps{0};

        void Add(SimulationStepTimings const &step)
        {
            timings.sourcesAndSetup += step.sourcesAndSetup;
            timings.water += step.water;
            timings.preSandDiagnostics += step.preSandDiagnostics;
            timings.sandDynamics += step.sandDynamics;
            timings.layerCorrection += step.layerCorrection;
            timings.sandFinalize += step.sandFinalize;
            timings.validation += step.validation;
            timings.diagnostics += step.diagnostics;
            timings.total += step.total;
            ++measuredSteps;
        }
    };

    void PrintStage(
        std::string_view name,
        double seconds,
        BenchmarkSummary const &summary)
    {
        double const percent = summary.timings.total > 0.0
            ? 100.0 * seconds / summary.timings.total
            : 0.0;
        double const millisecondsPerStep =
            1000.0 * seconds /
            static_cast<double>(summary.measuredSteps);
        fmt::print(
            "  {:>21}: {:9.4f} s  {:7.3f} ms/step  {:6.2f}%\n",
            name,
            seconds,
            millisecondsPerStep,
            percent);
    }
}

int main(int argc, char **argv)
{
    using namespace Pivot;

    try
    {
        cxxopts::Options options(
            "benchmark",
            "M7 solver-only benchmark with per-stage wall-clock timings");
        options.add_options()
            ("scene", "YAML scene file",
             cxxopts::value<std::string>())
            ("steps", "measured steps per repetition",
             cxxopts::value<int>()->default_value("200"))
            ("warmup", "unmeasured warm-up steps per repetition",
             cxxopts::value<int>()->default_value("20"))
            ("repetitions", "number of independent repetitions",
             cxxopts::value<int>()->default_value("3"))
            ("h,help", "print help");

        auto const arguments = options.parse(argc, argv);
        if (arguments.count("help") || !arguments.count("scene"))
        {
            std::cout << options.help() << '\n';
            return arguments.count("help") ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        std::filesystem::path const scenePath =
            std::filesystem::absolute(
                arguments["scene"].as<std::string>());
        int const steps = arguments["steps"].as<int>();
        int const warmup = arguments["warmup"].as<int>();
        int const repetitions = arguments["repetitions"].as<int>();
        if (steps <= 0 || warmup < 0 || repetitions <= 0)
        {
            throw std::invalid_argument(
                "steps/repetitions must be positive and warmup non-negative");
        }

        BenchmarkSummary summary;
        for (int repetition = 0; repetition < repetitions; ++repetition)
        {
            // SceneDescription 是 move-only，因此每次重复都重新解析 YAML，
            // 但计时从 warm-up 后才开始，不把加载或初始化算入求解耗时。
            SceneDescription scene =
                SceneLoader::FromYaml(scenePath);
            Simulation simulation(scene.config);
            simulation.Initialize(std::move(scene));
            for (int step = 0; step < warmup; ++step)
            {
                simulation.Step();
            }

            auto const begin = std::chrono::steady_clock::now();
            for (int step = 0; step < steps; ++step)
            {
                simulation.Step();
                summary.Add(simulation.GetLastStepTimings());
            }
            auto const end = std::chrono::steady_clock::now();
            summary.outerWallSeconds +=
                std::chrono::duration<double>(end - begin).count();
        }

        fmt::print(
            "scene={} grid benchmark, {} repetitions x {} steps "
            "(warmup {})\n",
            scenePath.string(),
            repetitions,
            steps,
            warmup);
        fmt::print(
            "TBB default concurrency: {}\n",
            tbb::info::default_concurrency());
        fmt::print(
            "outer wall: {:.4f} s, {:.3f} ms/step, {:.2f} steps/s\n",
            summary.outerWallSeconds,
            1000.0 * summary.outerWallSeconds /
                static_cast<double>(summary.measuredSteps),
            static_cast<double>(summary.measuredSteps) /
                summary.outerWallSeconds);
        PrintStage(
            "sources/setup",
            summary.timings.sourcesAndSetup,
            summary);
        PrintStage("water", summary.timings.water, summary);
        PrintStage(
            "pre-sand diagnostics",
            summary.timings.preSandDiagnostics,
            summary);
        PrintStage(
            "sand dynamics",
            summary.timings.sandDynamics,
            summary);
        PrintStage(
            "layer correction",
            summary.timings.layerCorrection,
            summary);
        PrintStage(
            "sand finalize/F",
            summary.timings.sandFinalize,
            summary);
        PrintStage(
            "validation",
            summary.timings.validation,
            summary);
        PrintStage(
            "final diagnostics",
            summary.timings.diagnostics,
            summary);
        PrintStage(
            "instrumented total",
            summary.timings.total,
            summary);
        return EXIT_SUCCESS;
    }
    catch (std::exception const &error)
    {
        fmt::print(stderr, "benchmark failed: {}\n", error.what());
        return EXIT_FAILURE;
    }
}

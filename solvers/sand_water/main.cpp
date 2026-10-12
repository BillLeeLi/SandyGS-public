#include "output/Output.h"
#include "scene/Scene.h"
#include "simulation/Simulation.h"

#include <cxxopts.hpp>

int main(int argc, char **argv)
{
    using namespace Pivot;

    try
    {
        cxxopts::Options options(
            "demo",
            "M6.5 YAML scene runner and long-horizon output exporter");
        options.add_options()
            ("scene", "M6 YAML scene file",
             cxxopts::value<std::string>()->default_value(""))
            ("o,output", "run output directory; empty uses results/<scene>",
             cxxopts::value<std::string>()->default_value(""))
            ("steps", "override scene step count; -1 keeps scene/default",
             cxxopts::value<int>()->default_value("-1"))
            ("duration", "run to this physical time; -1 keeps step mode",
             cxxopts::value<double>()->default_value("-1"))
            ("frame-stride", "override output stride; -1 keeps scene/default",
             cxxopts::value<int>()->default_value("-1"))
            ("snapshot", "override snapshot level: render, basic or diagnostic",
             cxxopts::value<std::string>()->default_value(""))
            ("write-obj", "export three OBJ surfaces for every saved frame",
             cxxopts::value<bool>()
                 ->default_value("false")
                 ->implicit_value("true"))
            ("no-obj", "disable scene-default OBJ export for long runs",
             cxxopts::value<bool>()
                 ->default_value("false")
                 ->implicit_value("true"))

            // 没有 --scene 时保留 M4/M5 参数化 pile 入口，方便已有命令继续使用。
            ("nx", "legacy cell count along x",
             cxxopts::value<int>()->default_value("72"))
            ("ny", "legacy cell count along y",
             cxxopts::value<int>()->default_value("72"))
            ("spacing", "legacy grid spacing",
             cxxopts::value<double>()->default_value("0.025"))
            ("dt", "legacy maximum outer water time step",
             cxxopts::value<double>()->default_value("0.003"))
            ("sand-substeps", "legacy sand steps per outer step",
             cxxopts::value<int>()->default_value("3"))
            ("gravity", "legacy gravity acceleration",
             cxxopts::value<double>()->default_value("9.81"))
            ("friction", "legacy ground friction",
             cxxopts::value<double>()->default_value("0.42"))
            ("sand-terrain-contact-tolerance", "legacy sand contact height tolerance",
             cxxopts::value<double>()->default_value("0"))
            ("exchange-rate", "legacy momentum exchange rate c_e",
             cxxopts::value<double>()->default_value("1"))
            ("exchange-scheme", "legacy exchange: paper or implicit",
             cxxopts::value<std::string>()->default_value("paper"))
            ("initial-sand-velocity-x", "legacy initial sand vx",
             cxxopts::value<double>()->default_value("0"))
            ("initial-mixed-velocity-x", "legacy initial mixed-water vx",
             cxxopts::value<double>()->default_value("0"))
            ("young-modulus", "legacy Young's modulus",
             cxxopts::value<double>()->default_value("75000"))
            ("poisson", "legacy Poisson ratio",
             cxxopts::value<double>()->default_value("0.25"))
            ("drucker-friction", "legacy Drucker-Prager friction c_f",
             cxxopts::value<double>()->default_value("0.5"))
            ("cohesion-scale", "legacy cohesion multiplier",
             cxxopts::value<double>()->default_value("1"))
            ("internal-force", "legacy elastoplastic internal force",
             cxxopts::value<bool>()->default_value("true"))
            ("internal-height-floor", "legacy absolute internal-force height floor",
             cxxopts::value<double>()->default_value("0"))
            ("internal-height-fraction", "legacy dx-relative internal-force floor",
             cxxopts::value<double>()->default_value("0.0004"))
            ("pile-shape", "legacy pile: gaussian or cylinder",
             cxxopts::value<std::string>()->default_value("gaussian"))
            ("pile-height", "legacy pile height",
             cxxopts::value<double>()->default_value("0.34"))
            ("pile-width", "legacy Gaussian sigma or cylinder radius",
             cxxopts::value<double>()->default_value("0.12"))
            ("pile-edge-width", "legacy cylinder smoothing width",
             cxxopts::value<double>()->default_value("0"))
            ("saturation", "legacy initial water/(water+sand)",
             cxxopts::value<double>()->default_value("0"))
            ("h,help", "print help");

        auto const arguments = options.parse(argc, argv);
        if (arguments.count("help"))
        {
            std::cout << options.help() << '\n';
            return EXIT_SUCCESS;
        }

        std::string const sceneArgument =
            arguments["scene"].as<std::string>();
        SceneDescription scene;
        std::optional<std::filesystem::path> sourceScenePath;
        double initialSandVelocityX = 0.0;
        double initialMixedVelocityX = 0.0;

        if (!sceneArgument.empty())
        {
            sourceScenePath =
                std::filesystem::absolute(sceneArgument);
            scene = SceneLoader::FromYaml(*sourceScenePath);
        }
        else
        {
            // 旧入口也构造成 SceneDescription，后面的运行/输出路径因此只有一份。
            scene.run.name = "legacy_pile";
            scene.run.steps = 260;
            scene.run.frameStride = 10;
            scene.config.grid.resolution = {
                arguments["nx"].as<int>(),
                arguments["ny"].as<int>()};
            scene.config.grid.spacing =
                arguments["spacing"].as<double>();
            scene.config.time.waterDt =
                arguments["dt"].as<double>();
            scene.config.time.sandSubsteps =
                arguments["sand-substeps"].as<int>();
            scene.config.time.internalForceHeightFloor =
                arguments["internal-height-floor"].as<double>();
            scene.config.time.internalForceHeightFraction =
                arguments["internal-height-fraction"].as<double>();
            scene.config.time.useAdaptiveDt = true;
            scene.config.physics.gravity =
                arguments["gravity"].as<double>();
            scene.config.physics.groundFriction =
                arguments["friction"].as<double>();
            scene.config.physics.sandTerrainContactTolerance =
                arguments["sand-terrain-contact-tolerance"].as<double>();
            scene.config.physics.momentumExchangeRate =
                arguments["exchange-rate"].as<double>();
            scene.config.physics.enableSandInternalForce =
                arguments["internal-force"].as<bool>();
            scene.config.physics.diffusionRate = 0.0;

            std::string const exchangeScheme =
                arguments["exchange-scheme"].as<std::string>();
            if (exchangeScheme != "paper" &&
                exchangeScheme != "implicit")
            {
                throw std::invalid_argument(
                    "exchange-scheme must be paper or implicit");
            }
            scene.config.exchangeScheme =
                exchangeScheme == "paper"
                ? SandWaterExchangeScheme::PaperSequential
                : SandWaterExchangeScheme::AnalyticImplicit;

            double const youngModulus =
                arguments["young-modulus"].as<double>();
            double const poisson =
                arguments["poisson"].as<double>();
            if (!(youngModulus > 0.0) ||
                !(poisson >= 0.0 && poisson < 0.5))
            {
                throw std::invalid_argument(
                    "invalid Young's modulus or Poisson ratio");
            }
            scene.config.physics.lameMu =
                youngModulus / (2.0 * (1.0 + poisson));
            scene.config.physics.lameLambda =
                youngModulus * poisson /
                ((1.0 + poisson) *
                 (1.0 - 2.0 * poisson));
            scene.config.physics.druckerPragerFriction =
                arguments["drucker-friction"].as<double>();
            double const cohesionScale =
                arguments["cohesion-scale"].as<double>();
            scene.config.physics.cohesion.c0 *= cohesionScale;
            scene.config.physics.cohesion.c1 *= cohesionScale;
            scene.config.physics.cohesion.c2 *= cohesionScale;

            std::string const pileShape =
                arguments["pile-shape"].as<std::string>();
            double const pileHeight =
                arguments["pile-height"].as<double>();
            double const pileWidth =
                arguments["pile-width"].as<double>();
            double const edgeWidth =
                arguments["pile-edge-width"].as<double>();
            double const saturation =
                arguments["saturation"].as<double>();
            if ((pileShape != "gaussian" &&
                 pileShape != "cylinder") ||
                !(pileHeight > 0.0) ||
                !(pileWidth > 0.0) ||
                edgeWidth < 0.0 ||
                !(saturation >= 0.0 && saturation < 1.0))
            {
                throw std::invalid_argument(
                    "invalid legacy pile arguments");
            }

            scene.terrain =
                [](Vector2d const &) { return 0.0; };
            scene.sandHeight =
                [=](Vector2d const &position)
            {
                if (pileShape == "gaussian")
                {
                    return pileHeight * std::exp(
                        -position.squaredNorm() /
                        (2.0 * pileWidth * pileWidth));
                }
                if (edgeWidth == 0.0)
                {
                    return position.norm() <= pileWidth
                        ? pileHeight
                        : 0.0;
                }
                double const coverage = std::clamp(
                    (pileWidth + 0.5 * edgeWidth -
                     position.norm()) /
                        edgeWidth,
                    0.0,
                    1.0);
                return pileHeight * coverage * coverage *
                    (3.0 - 2.0 * coverage);
            };
            HeightFunction const sandFunction =
                scene.sandHeight;
            scene.totalWaterHeight =
                [sandFunction, saturation](
                    Vector2d const &position)
            {
                return saturation / (1.0 - saturation) *
                    sandFunction(position);
            };
            initialSandVelocityX =
                arguments["initial-sand-velocity-x"].as<double>();
            initialMixedVelocityX =
                arguments["initial-mixed-velocity-x"].as<double>();
        }

        int const stepOverride =
            arguments["steps"].as<int>();
        double const durationOverride =
            arguments["duration"].as<double>();
        int const strideOverride =
            arguments["frame-stride"].as<int>();
        if (stepOverride < -1 ||
            !std::isfinite(durationOverride) ||
            (durationOverride < 0.0 &&
             durationOverride != -1.0))
        {
            throw std::invalid_argument(
                "steps and duration overrides must be -1 or non-negative");
        }
        if (stepOverride >= 0 && durationOverride >= 0.0)
        {
            throw std::invalid_argument(
                "steps and duration cannot both be overridden");
        }
        if (stepOverride >= 0)
        {
            scene.run.steps = stepOverride;
            // 显式 --steps 应覆盖 YAML 中可能存在的 duration。
            scene.run.duration = -1.0;
        }
        if (durationOverride >= 0.0)
        {
            scene.run.duration = durationOverride;
        }
        if (strideOverride >= 1)
        {
            scene.run.frameStride = strideOverride;
        }
        if (scene.run.steps < 0 ||
            !std::isfinite(scene.run.duration) ||
            (scene.run.duration < 0.0 &&
             scene.run.duration != -1.0) ||
            scene.run.frameStride < 1)
        {
            throw std::invalid_argument(
                "steps must be non-negative, duration must be -1 or "
                "non-negative, and frame-stride must be positive");
        }

        std::string const snapshotOverride =
            arguments["snapshot"].as<std::string>();
        SnapshotLevel snapshotLevel =
            scene.run.diagnosticSnapshots
            ? SnapshotLevel::Diagnostic
            : SnapshotLevel::Basic;
        if (!snapshotOverride.empty())
        {
            if (snapshotOverride != "render" &&
                snapshotOverride != "basic" &&
                snapshotOverride != "diagnostic")
            {
                throw std::invalid_argument(
                    "snapshot must be render, basic or diagnostic");
            }
            scene.run.diagnosticSnapshots =
                snapshotOverride == "diagnostic";
            snapshotLevel =
                snapshotOverride == "render"
                ? SnapshotLevel::Render
                : (snapshotOverride == "diagnostic"
                    ? SnapshotLevel::Diagnostic
                    : SnapshotLevel::Basic);
        }
        bool const forceObj =
            arguments["write-obj"].as<bool>();
        bool const disableObj =
            arguments["no-obj"].as<bool>();
        if (forceObj && disableObj)
        {
            throw std::invalid_argument(
                "write-obj and no-obj cannot be used together");
        }
        // --no-obj 是长程 Render 模式的显式磁盘保护；未指定时仍尊重 YAML。
        scene.run.writeObj =
            disableObj ? false : (scene.run.writeObj || forceObj);

        SimulationConfig const config = scene.config;
        SceneRunSettings const run = scene.run;
        Simulation simulation(config);
        simulation.Initialize(std::move(scene));

        // 旧 M5 入口允许全场初速度；YAML 场景改用 VelocityRegion 源项表达。
        if (sourceScenePath == std::nullopt)
        {
            simulation.GetState().sand.velocity.SetConstant(
                Vector2d(initialSandVelocityX, 0.0));
            simulation.GetState().mixedWater.velocity.SetConstant(
                Vector2d(initialMixedVelocityX, 0.0));
            BoundaryConditions initializationBoundary(
                simulation.GetGrid());
            initializationBoundary.ApplyVelocityBoundary(
                simulation.GetState().sand.velocity);
            initializationBoundary.ApplyVelocityBoundary(
                simulation.GetState().mixedWater.velocity);
            simulation.Validate("M6 legacy initial velocities");
        }

        std::string outputArgument =
            arguments["output"].as<std::string>();
        if (outputArgument.empty())
        {
            outputArgument =
                (std::filesystem::path("results") /
                 run.name).string();
        }
        std::filesystem::path const runDirectory =
            std::filesystem::absolute(outputArgument);

        FrameExporter exporter;
        exporter.WriteMetadata(
            config, runDirectory, run.name, snapshotLevel);
        if (sourceScenePath)
        {
            // 保存一份实际运行的 YAML，使结果目录不依赖原文件后来是否被修改。
            std::filesystem::copy_file(
                *sourceScenePath,
                runDirectory / "scene.yaml",
                std::filesystem::copy_options::overwrite_existing);
        }

        auto const writeCurrentFrame = [&]
        {
            ConservationReport const report =
                simulation.GetState().stepIndex == 0
                ? simulation.Measure()
                : simulation.GetLastReport();
            SimulationSnapshot const snapshot(
                simulation.GetState(),
                simulation.GetScratch());
            exporter.WriteFrame(
                snapshot,
                report,
                config,
                snapshotLevel,
                runDirectory,
                false);
            if (run.writeObj)
            {
                exporter.WriteObjMeshes(
                    snapshot,
                    config,
                    runDirectory / "meshes" /
                        fmt::format(
                            "{:06d}",
                            snapshot.stepIndex));
            }
        };

        auto const writeCurrentMetrics = [&]
        {
            ConservationReport const report =
                simulation.GetState().stepIndex == 0
                ? simulation.Measure()
                : simulation.GetLastReport();
            exporter.WriteMetrics(
                SimulationSnapshot(simulation.GetState()),
                report,
                runDirectory);
        };

        // 标量历史每步记录，体积较小且能覆盖短暂的稳定性极值；大型网格数组
        // 仍只在 frameStride 命中时保存。
        writeCurrentMetrics();
        writeCurrentFrame();
        ConservationReport const initialReport =
            simulation.Measure();
        int executedSteps = 0;
        auto const reachedEnd = [&]
        {
            if (run.duration < 0.0)
            {
                return executedSteps >= run.steps;
            }
            double const tolerance =
                16.0 * std::numeric_limits<double>::epsilon() *
                std::max(1.0, std::abs(run.duration));
            return run.duration - simulation.GetState().time <=
                tolerance;
        };
        while (!reachedEnd())
        {
            if (run.duration >= 0.0)
            {
                // Step(maximumDt) 保留全部稳定性限制，并在最后一步把 dt 缩短到
                // remaining，因此结果时间标签可以精确用于跨配置比较。
                double const remaining =
                    run.duration - simulation.GetState().time;
                simulation.Step(remaining);
            }
            else
            {
                simulation.Step();
            }
            ++executedSteps;
            writeCurrentMetrics();
            if (simulation.GetState().stepIndex %
                        static_cast<std::uint64_t>(
                            run.frameStride) ==
                    0 ||
                reachedEnd())
            {
                writeCurrentFrame();
            }
        }

        ConservationReport const finalReport =
            simulation.GetState().stepIndex == 0
            ? simulation.Measure()
            : simulation.GetLastReport();
        spdlog::info(
            "M6.5 scene '{}' completed: {} steps, t={:.6g}",
            run.name,
            simulation.GetState().stepIndex,
            simulation.GetState().time);
        spdlog::info(
            "output={}, stop={}, frames every {}, OBJ={}, snapshots={}",
            runDirectory.string(),
            run.duration >= 0.0
                ? fmt::format("duration {:.6g}", run.duration)
                : fmt::format("{} steps", run.steps),
            run.frameStride,
            run.writeObj,
            snapshotLevel == SnapshotLevel::Render
                ? "render"
                : (snapshotLevel == SnapshotLevel::Diagnostic
                    ? "diagnostic"
                    : "basic"));
        spdlog::info(
            "volumes sand {:.9g}->{:.9g}, water {:.9g}->{:.9g}; "
            "min det(F)={:.6g}, max yield residual={:.3e}",
            initialReport.sandVolume,
            finalReport.sandVolume,
            initialReport.totalWaterVolume,
            finalReport.totalWaterVolume,
            finalReport.minDetF,
            finalReport.maxYieldResidual);
        return EXIT_SUCCESS;
    }
    catch (std::exception const &error)
    {
        spdlog::critical("{}", error.what());
        return EXIT_FAILURE;
    }
}

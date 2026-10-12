#include "scene/Scene.h"

#include <npy.hpp>

namespace Pivot
{
    namespace
    {
        double ActiveDuration(
            double stepBegin,
            double stepEnd,
            double sourceBegin,
            double sourceEnd)
        {
            return std::max(
                0.0,
                std::min(stepEnd, sourceEnd) -
                    std::max(stepBegin, sourceBegin));
        }

        void ValidateSourceParameters(
            SourceRegion const &region,
            Vector2d const &velocity,
            double beginTime,
            double endTime)
        {
            region.Validate();
            if (!velocity.array().isFinite().all())
            {
                throw std::invalid_argument(
                    "source velocity must be finite");
            }
            if (!std::isfinite(beginTime) ||
                !(endTime > beginTime))
            {
                throw std::invalid_argument(
                    "source interval must be finite and non-empty");
            }
        }

        void PrescribeVelocity(
            SGridData<double> &field,
            SourceRegion const &region,
            Vector2d const &velocity)
        {
            ForEach(field.GetGrids(), [&](int axis, Vector2i const &face)
            {
                Vector2d const position =
                    field[axis].GetGrid().PositionOf(face);
                if (region.Contains(position))
                {
                    field[axis][face] = velocity[axis];
                }
            });
        }

        Vector2d ReadVector2d(
            YAML::Node const &node,
            std::string_view context)
        {
            if (!node || !node.IsSequence() || node.size() != 2)
            {
                throw std::runtime_error(fmt::format(
                    "{} must be a two-element sequence", context));
            }
            Vector2d const value(
                node[0].as<double>(),
                node[1].as<double>());
            if (!value.array().isFinite().all())
            {
                throw std::runtime_error(fmt::format(
                    "{} must contain finite values", context));
            }
            return value;
        }

        Vector2i ReadVector2i(
            YAML::Node const &node,
            std::string_view context)
        {
            if (!node || !node.IsSequence() || node.size() != 2)
            {
                throw std::runtime_error(fmt::format(
                    "{} must be a two-element sequence", context));
            }
            return {
                node[0].as<int>(),
                node[1].as<int>()};
        }

        template <typename Type>
        void ReadOptional(
            YAML::Node const &node,
            char const *key,
            Type &destination)
        {
            if (node && node[key])
            {
                destination = node[key].as<Type>();
            }
        }

        HeightFunction BuildPrimitive(
            YAML::Node const &node,
            SimulationConfig const &config,
            std::filesystem::path const &baseDirectory)
        {
            if (!node || !node.IsMap() || !node["type"])
            {
                throw std::runtime_error(
                    "height primitive must be a map containing 'type'");
            }
            std::string const type = node["type"].as<std::string>();
            if (type == "constant")
            {
                double const height = node["height"].as<double>();
                return [height](Vector2d const &) { return height; };
            }
            if (type == "plane")
            {
                double const height = node["height"]
                    ? node["height"].as<double>()
                    : 0.0;
                Vector2d const slope = node["slope"]
                    ? ReadVector2d(node["slope"], "plane.slope")
                    : Vector2d::Zero();
                Vector2d const origin = node["origin"]
                    ? ReadVector2d(node["origin"], "plane.origin")
                    : Vector2d::Zero();
                return [height, slope, origin](Vector2d const &position)
                {
                    return height + slope.dot(position - origin);
                };
            }
            if (type == "box")
            {
                Vector2d const minimum =
                    ReadVector2d(node["min"], "box.min");
                Vector2d const maximum =
                    ReadVector2d(node["max"], "box.max");
                double const height = node["height"].as<double>();
                double const edgeWidth = node["edge_width"]
                    ? node["edge_width"].as<double>()
                    : 0.0;
                if ((maximum.array() <= minimum.array()).any() ||
                    edgeWidth < 0.0)
                {
                    throw std::runtime_error(
                        "box.max must be greater than box.min and "
                        "edge_width non-negative");
                }
                return [minimum, maximum, height, edgeWidth](
                    Vector2d const &position)
                {
                    if (edgeWidth == 0.0)
                    {
                        return
                            (position.array() >= minimum.array()).all() &&
                            (position.array() <= maximum.array()).all()
                            ? height
                            : 0.0;
                    }

                    // M7.1.5 平滑初值消融：过渡区以原 box 边界为中心，
                    // 使用 C1 连续 smoothstep。两个轴的 coverage 相乘，
                    // 因而远离边缘时保持原高度，且不改变未指定
                    // edge_width 的既有场景语义。
                    double coverage = 1.0;
                    auto const smoothstep = [](double value)
                    {
                        return value * value *
                            (3.0 - 2.0 * value);
                    };
                    for (int axis = 0; axis < 2; ++axis)
                    {
                        double const lowerCoverage = std::clamp(
                            (position[axis] - minimum[axis]) /
                                edgeWidth +
                                0.5,
                            0.0,
                            1.0);
                        double const upperCoverage = std::clamp(
                            (maximum[axis] - position[axis]) /
                                edgeWidth +
                                0.5,
                            0.0,
                            1.0);
                        coverage *=
                            smoothstep(lowerCoverage) *
                            smoothstep(upperCoverage);
                    }
                    return height * coverage;
                };
            }
            if (type == "cylinder")
            {
                Vector2d const center = node["center"]
                    ? ReadVector2d(node["center"], "cylinder.center")
                    : Vector2d::Zero();
                double const radius = node["radius"].as<double>();
                double const height = node["height"].as<double>();
                double const edgeWidth = node["edge_width"]
                    ? node["edge_width"].as<double>()
                    : 0.0;
                if (!(radius > 0.0) || edgeWidth < 0.0)
                {
                    throw std::runtime_error(
                        "cylinder radius must be positive and edge_width non-negative");
                }
                return [center, radius, height, edgeWidth](
                    Vector2d const &position)
                {
                    double const distance = (position - center).norm();
                    if (edgeWidth == 0.0)
                    {
                        return distance <= radius ? height : 0.0;
                    }
                    double const coverage = std::clamp(
                        (radius + 0.5 * edgeWidth - distance) /
                            edgeWidth,
                        0.0,
                        1.0);
                    return height * coverage * coverage *
                        (3.0 - 2.0 * coverage);
                };
            }
            if (type == "gaussian")
            {
                Vector2d const center = node["center"]
                    ? ReadVector2d(node["center"], "gaussian.center")
                    : Vector2d::Zero();
                double const sigma = node["sigma"].as<double>();
                double const height = node["height"].as<double>();
                if (!(sigma > 0.0))
                {
                    throw std::runtime_error(
                        "gaussian sigma must be positive");
                }
                return [center, sigma, height](
                    Vector2d const &position)
                {
                    return height * std::exp(
                        -(position - center).squaredNorm() /
                        (2.0 * sigma * sigma));
                };
            }
            if (type == "npy")
            {
                std::filesystem::path file = node["file"].as<std::string>();
                if (file.is_relative())
                {
                    file = baseDirectory / file;
                }
                std::vector<unsigned long> shape;
                std::vector<double> data;
                npy::LoadArrayFromNumpy<double>(
                    file.string(), shape, data);
                if (shape.size() != 2 ||
                    shape[0] < 2 || shape[1] < 2)
                {
                    throw std::runtime_error(
                        "NPY height map must be a 2D double array of at least 2x2");
                }
                Vector2i const size(
                    static_cast<int>(shape[0]),
                    static_cast<int>(shape[1]));
                double const spacing = node["spacing"]
                    ? node["spacing"].as<double>()
                    : config.grid.spacing;
                Vector2d const defaultOrigin =
                    config.grid.center -
                    0.5 * config.grid.resolution.cast<double>() *
                        config.grid.spacing +
                    Vector2d::Constant(0.5 * config.grid.spacing);
                Vector2d const origin = node["origin"]
                    ? ReadVector2d(node["origin"], "npy.origin")
                    : defaultOrigin;
                double const scale = node["scale"]
                    ? node["scale"].as<double>()
                    : 1.0;
                double const offset = node["offset"]
                    ? node["offset"].as<double>()
                    : 0.0;
                if (!(spacing > 0.0))
                {
                    throw std::runtime_error(
                        "NPY height-map spacing must be positive");
                }

                // 数据沿用项目 NPY 的 [x,y] 存储顺序。双线性采样允许同一
                // 高度图在不同网格分辨率下复用，域外使用最近边界值。
                return [data = std::move(data), size, spacing,
                        origin, scale, offset](
                    Vector2d const &position)
                {
                    Vector2d const index = (
                        (position - origin) / spacing)
                        .cwiseMax(0.0)
                        .cwiseMin(
                            (size - Vector2i::Ones())
                                .cast<double>());
                    Vector2i lower =
                        index.array().floor().cast<int>().matrix();
                    lower = lower.cwiseMin(
                        size - Vector2i::Constant(2));
                    Vector2d const fraction =
                        index - lower.cast<double>();
                    auto const at = [&](int i, int j)
                    {
                        Vector2i const cell =
                            Vector2i(i, j).cwiseMax(0).cwiseMin(
                                size - Vector2i::Ones());
                        return data[
                            static_cast<std::size_t>(
                                cell.y() + size.y() * cell.x())];
                    };
                    double const x0 =
                        (1.0 - fraction.y()) *
                            at(lower.x(), lower.y()) +
                        fraction.y() *
                            at(lower.x(), lower.y() + 1);
                    double const x1 =
                        (1.0 - fraction.y()) *
                            at(lower.x() + 1, lower.y()) +
                        fraction.y() *
                            at(lower.x() + 1, lower.y() + 1);
                    return offset + scale * (
                        (1.0 - fraction.x()) * x0 +
                        fraction.x() * x1);
                };
            }
            throw std::runtime_error(fmt::format(
                "unsupported height primitive type '{}'", type));
        }

        HeightFunction BuildHeightFunction(
            YAML::Node const &node,
            SimulationConfig const &config,
            std::filesystem::path const &baseDirectory)
        {
            if (!node)
            {
                return [](Vector2d const &) { return 0.0; };
            }
            if (node.IsScalar())
            {
                double const value = node.as<double>();
                return [value](Vector2d const &) { return value; };
            }
            if (node["type"])
            {
                return BuildPrimitive(node, config, baseDirectory);
            }
            YAML::Node const shapes = node["shapes"];
            if (!shapes || !shapes.IsSequence() || shapes.size() == 0)
            {
                throw std::runtime_error(
                    "composite height field requires a non-empty shapes sequence");
            }
            std::string const combine = node["combine"]
                ? node["combine"].as<std::string>()
                : "sum";
            if (combine != "sum" && combine != "max")
            {
                throw std::runtime_error(
                    "height-field combine must be 'sum' or 'max'");
            }
            std::vector<HeightFunction> functions;
            functions.reserve(shapes.size());
            for (YAML::Node const &shape : shapes)
            {
                functions.push_back(
                    BuildPrimitive(shape, config, baseDirectory));
            }
            return [functions = std::move(functions), combine](
                Vector2d const &position)
            {
                double value = 0.0;
                for (HeightFunction const &function : functions)
                {
                    double const sample = function(position);
                    value = combine == "max"
                        ? std::max(value, sample)
                        : value + sample;
                }
                return value;
            };
        }

        SourceRegion ReadRegion(YAML::Node const &node)
        {
            if (!node || !node["type"])
            {
                throw std::runtime_error(
                    "source region requires a type");
            }
            SourceRegion region;
            std::string const type = node["type"].as<std::string>();
            if (type == "box")
            {
                region.shape = SourceRegion::Shape::Box;
                region.minimum =
                    ReadVector2d(node["min"], "region.min");
                region.maximum =
                    ReadVector2d(node["max"], "region.max");
            }
            else if (type == "circle")
            {
                region.shape = SourceRegion::Shape::Circle;
                region.center =
                    ReadVector2d(node["center"], "region.center");
                region.radius = node["radius"].as<double>();
            }
            else
            {
                throw std::runtime_error(fmt::format(
                    "unsupported source region type '{}'", type));
            }
            region.Validate();
            return region;
        }

        SourceMaterial ReadMaterial(std::string const &name)
        {
            if (name == "sand") return SourceMaterial::Sand;
            if (name == "pure_water") return SourceMaterial::PureWater;
            if (name == "mixed_water") return SourceMaterial::MixedWater;
            if (name == "all_water") return SourceMaterial::AllWater;
            if (name == "all") return SourceMaterial::All;
            throw std::runtime_error(fmt::format(
                "unsupported source material '{}'", name));
        }

        void ReadConfig(
            YAML::Node const &root,
            SimulationConfig &config)
        {
            YAML::Node const grid = root["grid"];
            if (grid)
            {
                if (grid["resolution"])
                    config.grid.resolution =
                        ReadVector2i(grid["resolution"], "grid.resolution");
                ReadOptional(grid, "spacing", config.grid.spacing);
                ReadOptional(
                    grid, "boundary_width",
                    config.grid.boundaryWidth);
                if (grid["center"])
                    config.grid.center =
                        ReadVector2d(grid["center"], "grid.center");
            }

            YAML::Node const physics = root["physics"];
            if (physics)
            {
                if (physics["saturation_mode"])
                {
                    std::string const mode =
                        physics["saturation_mode"].as<std::string>();
                    if (mode == "dynamic")
                    {
                        config.physics.saturationMode =
                            SaturationEvolutionMode::Dynamic;
                    }
                    else if (mode == "frozen_bound")
                    {
                        config.physics.saturationMode =
                            SaturationEvolutionMode::FrozenBound;
                    }
                    else
                    {
                        throw std::runtime_error(fmt::format(
                            "unsupported physics.saturation_mode '{}'", mode));
                    }
                }
                ReadOptional(physics, "gravity", config.physics.gravity);
                ReadOptional(
                    physics, "sand_density",
                    config.physics.sandDensity);
                ReadOptional(
                    physics, "water_density",
                    config.physics.waterDensity);
                ReadOptional(
                    physics, "diffusion_rate",
                    config.physics.diffusionRate);
                ReadOptional(
                    physics, "momentum_exchange_rate",
                    config.physics.momentumExchangeRate);
                ReadOptional(
                    physics, "ground_friction",
                    config.physics.groundFriction);
                ReadOptional(
                    physics, "sand_terrain_contact_tolerance",
                    config.physics.sandTerrainContactTolerance);
                ReadOptional(
                    physics, "drucker_prager_friction",
                    config.physics.druckerPragerFriction);
                ReadOptional(
                    physics, "enable_sand_internal_force",
                    config.physics.enableSandInternalForce);
                ReadOptional(
                    physics, "project_deformation_after_advection",
                    config.physics.projectDeformationAfterAdvection);
                ReadOptional(
                    physics, "constitutive_saturation_lowpass",
                    config.physics.constitutiveSaturationLowpass);

                if (physics["young_modulus"])
                {
                    double const young =
                        physics["young_modulus"].as<double>();
                    double const poisson = physics["poisson"]
                        ? physics["poisson"].as<double>()
                        : 0.25;
                    config.physics.lameMu =
                        young / (2.0 * (1.0 + poisson));
                    config.physics.lameLambda =
                        young * poisson /
                        ((1.0 + poisson) *
                         (1.0 - 2.0 * poisson));
                }
                else
                {
                    ReadOptional(
                        physics, "lame_mu",
                        config.physics.lameMu);
                    ReadOptional(
                        physics, "lame_lambda",
                        config.physics.lameLambda);
                }

                YAML::Node const cohesion = physics["cohesion"];
                if (cohesion)
                {
                    ReadOptional(
                        cohesion, "c0",
                        config.physics.cohesion.c0);
                    ReadOptional(
                        cohesion, "c1",
                        config.physics.cohesion.c1);
                    ReadOptional(
                        cohesion, "c2",
                        config.physics.cohesion.c2);
                    ReadOptional(
                        cohesion, "phi1",
                        config.physics.cohesion.phi1);
                    ReadOptional(
                        cohesion, "phi2",
                        config.physics.cohesion.phi2);
                    ReadOptional(
                        cohesion, "phi3",
                        config.physics.cohesion.phi3);
                }
            }

            YAML::Node const time = root["time"];
            if (time)
            {
                ReadOptional(time, "water_dt", config.time.waterDt);
                ReadOptional(
                    time, "sand_substeps",
                    config.time.sandSubsteps);
                ReadOptional(
                    time, "height_epsilon",
                    config.time.heightEpsilon);
                ReadOptional(
                    time, "internal_force_height_floor",
                    config.time.internalForceHeightFloor);
                ReadOptional(
                    time, "internal_force_height_fraction",
                    config.time.internalForceHeightFraction);
                ReadOptional(
                    time, "determinant_epsilon",
                    config.time.determinantEpsilon);
                ReadOptional(
                    time, "target_cfl",
                    config.time.targetCfl);
                ReadOptional(
                    time, "diffusion_safety",
                    config.time.diffusionSafety);
                ReadOptional(
                    time, "max_velocity",
                    config.time.maxVelocity);
                ReadOptional(
                    time, "adaptive",
                    config.time.useAdaptiveDt);
            }
            if (root["exchange_scheme"])
            {
                std::string const scheme =
                    root["exchange_scheme"].as<std::string>();
                if (scheme == "paper" ||
                    scheme == "paper_sequential")
                {
                    config.exchangeScheme =
                        SandWaterExchangeScheme::PaperSequential;
                }
                else if (scheme == "implicit" ||
                         scheme == "analytic_implicit")
                {
                    config.exchangeScheme =
                        SandWaterExchangeScheme::AnalyticImplicit;
                }
                else
                {
                    throw std::runtime_error(fmt::format(
                        "unsupported exchange_scheme '{}'", scheme));
                }
            }
        }
    }

    bool SourceRegion::Contains(Vector2d const &position) const
    {
        if (shape == Shape::Box)
        {
            return (position.array() >= minimum.array()).all() &&
                (position.array() <= maximum.array()).all();
        }
        return (position - center).squaredNorm() <= radius * radius;
    }

    void SourceRegion::Validate() const
    {
        if (shape == Shape::Box)
        {
            if (!minimum.array().isFinite().all() ||
                !maximum.array().isFinite().all() ||
                (maximum.array() <= minimum.array()).any())
            {
                throw std::invalid_argument(
                    "box source region must have finite min < max");
            }
        }
        else if (!center.array().isFinite().all() ||
                 !(radius > 0.0) || !std::isfinite(radius))
        {
            throw std::invalid_argument(
                "circle source region must have finite center and positive radius");
        }
    }

    WaterInlet::WaterInlet(
        SourceRegion region,
        double heightRate,
        Vector2d velocity,
        double beginTime,
        double endTime)
        : region_(std::move(region)),
          heightRate_(heightRate),
          velocity_(velocity),
          beginTime_(beginTime),
          endTime_(endTime)
    {
        ValidateSourceParameters(
            region_, velocity_, beginTime_, endTime_);
        if (!(heightRate_ >= 0.0) || !std::isfinite(heightRate_))
        {
            throw std::invalid_argument(
                "water inlet height rate must be finite and non-negative");
        }
    }

    void WaterInlet::Apply(
        SimulationState &state,
        double beginTime,
        double endTime) const
    {
        double const duration = ActiveDuration(
            beginTime, endTime, beginTime_, endTime_);
        if (duration == 0.0 || heightRate_ == 0.0)
        {
            return;
        }
        double const addedHeight = heightRate_ * duration;
        ForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
        {
            Vector2d const position =
                state.terrain.GetGrid().PositionOf(cell);
            if (!region_.Contains(position))
            {
                return;
            }
            double const total =
                state.pureWater.height[cell] +
                state.mixedWater.height[cell] +
                addedHeight;
            state.mixedWater.height[cell] =
                std::min(total, state.sand.height[cell]);
            state.pureWater.height[cell] =
                total - state.mixedWater.height[cell];
        });
        // 入口速度是外部边界条件而非守恒混合；层划分后的两个水相都使用它。
        PrescribeVelocity(
            state.pureWater.velocity, region_, velocity_);
        PrescribeVelocity(
            state.mixedWater.velocity, region_, velocity_);
    }

    SandEmitter::SandEmitter(
        SourceRegion region,
        double heightRate,
        Vector2d velocity,
        double beginTime,
        double endTime)
        : region_(std::move(region)),
          heightRate_(heightRate),
          velocity_(velocity),
          beginTime_(beginTime),
          endTime_(endTime)
    {
        ValidateSourceParameters(
            region_, velocity_, beginTime_, endTime_);
        if (!(heightRate_ >= 0.0) || !std::isfinite(heightRate_))
        {
            throw std::invalid_argument(
                "sand emitter height rate must be finite and non-negative");
        }
    }

    void SandEmitter::Apply(
        SimulationState &state,
        double beginTime,
        double endTime) const
    {
        double const duration = ActiveDuration(
            beginTime, endTime, beginTime_, endTime_);
        if (duration == 0.0 || heightRate_ == 0.0)
        {
            return;
        }
        double const addedHeight = heightRate_ * duration;
        ForEach(state.terrain.GetGrid(), [&](Vector2i const &cell)
        {
            Vector2d const position =
                state.terrain.GetGrid().PositionOf(cell);
            if (!region_.Contains(position))
            {
                return;
            }
            bool const wasDry = state.sand.height[cell] <= 0.0;
            state.sand.height[cell] += addedHeight;
            if (wasDry)
            {
                // 新生成的材料不能继承该网格单元之前遗留的弹塑性历史。
                state.sand.deformationGradient[cell] =
                    Matrix2d::Identity();
                state.sand.rotationAngle[cell] = 0.0;
                state.sand.stretch[cell] = Matrix2d::Identity();
            }
        });
        PrescribeVelocity(state.sand.velocity, region_, velocity_);
    }

    VelocityRegion::VelocityRegion(
        SourceRegion region,
        SourceMaterial material,
        Vector2d velocity,
        double beginTime,
        double endTime)
        : region_(std::move(region)),
          material_(material),
          velocity_(velocity),
          beginTime_(beginTime),
          endTime_(endTime)
    {
        ValidateSourceParameters(
            region_, velocity_, beginTime_, endTime_);
    }

    void VelocityRegion::Apply(
        SimulationState &state,
        double beginTime,
        double endTime) const
    {
        if (ActiveDuration(
                beginTime, endTime, beginTime_, endTime_) == 0.0)
        {
            return;
        }
        if (material_ == SourceMaterial::Sand ||
            material_ == SourceMaterial::All)
        {
            PrescribeVelocity(
                state.sand.velocity, region_, velocity_);
        }
        if (material_ == SourceMaterial::PureWater ||
            material_ == SourceMaterial::AllWater ||
            material_ == SourceMaterial::All)
        {
            PrescribeVelocity(
                state.pureWater.velocity, region_, velocity_);
        }
        if (material_ == SourceMaterial::MixedWater ||
            material_ == SourceMaterial::AllWater ||
            material_ == SourceMaterial::All)
        {
            PrescribeVelocity(
                state.mixedWater.velocity, region_, velocity_);
        }
    }

    SceneDescription::SceneDescription()
        : terrain([](Vector2d const &) { return 0.0; }),
          sandHeight([](Vector2d const &) { return 0.0; }),
          totalWaterHeight([](Vector2d const &) { return 0.0; })
    {
    }

    SceneDescription SceneLoader::FromYaml(
        std::filesystem::path const &path)
    {
        std::filesystem::path const absolutePath =
            std::filesystem::absolute(path);
        YAML::Node const root =
            YAML::LoadFile(absolutePath.string());
        SceneDescription scene;
        ReadConfig(root, scene.config);
        scene.config.Validate();

        YAML::Node const run = root["run"];
        scene.run.name = root["name"]
            ? root["name"].as<std::string>()
            : absolutePath.stem().string();
        if (run)
        {
            ReadOptional(run, "steps", scene.run.steps);
            ReadOptional(run, "duration", scene.run.duration);
            ReadOptional(
                run, "frame_stride",
                scene.run.frameStride);
            ReadOptional(
                run, "diagnostic_snapshots",
                scene.run.diagnosticSnapshots);
            ReadOptional(
                run, "write_obj",
                scene.run.writeObj);
        }
        if (scene.run.steps < 0 ||
            !std::isfinite(scene.run.duration) ||
            (scene.run.duration < 0.0 &&
             scene.run.duration != -1.0) ||
            scene.run.frameStride < 1)
        {
            throw std::runtime_error(
                "run.steps must be non-negative, duration must be -1 or "
                "non-negative, and frame_stride must be positive");
        }

        YAML::Node const initial = root["initial"];
        std::filesystem::path const baseDirectory =
            absolutePath.parent_path();
        scene.terrain = BuildHeightFunction(
            initial ? initial["terrain"] : YAML::Node(),
            scene.config,
            baseDirectory);
        scene.sandHeight = BuildHeightFunction(
            initial ? initial["sand"] : YAML::Node(),
            scene.config,
            baseDirectory);
        scene.totalWaterHeight = BuildHeightFunction(
            initial ? initial["total_water"] : YAML::Node(),
            scene.config,
            baseDirectory);

        YAML::Node const sources = root["sources"];
        if (sources)
        {
            if (!sources.IsSequence())
            {
                throw std::runtime_error(
                    "sources must be a sequence");
            }
            for (YAML::Node const &source : sources)
            {
                std::string const type =
                    source["type"].as<std::string>();
                SourceRegion const region =
                    ReadRegion(source["region"]);
                Vector2d const velocity = source["velocity"]
                    ? ReadVector2d(
                        source["velocity"], "source.velocity")
                    : Vector2d::Zero();
                double const begin = source["begin"]
                    ? source["begin"].as<double>()
                    : 0.0;
                double const end = source["end"]
                    ? source["end"].as<double>()
                    : std::numeric_limits<double>::max();
                if (type == "water_inlet")
                {
                    scene.sources.push_back(
                        std::make_unique<WaterInlet>(
                            region,
                            source["height_rate"].as<double>(),
                            velocity,
                            begin,
                            end));
                }
                else if (type == "sand_emitter")
                {
                    scene.sources.push_back(
                        std::make_unique<SandEmitter>(
                            region,
                            source["height_rate"].as<double>(),
                            velocity,
                            begin,
                            end));
                }
                else if (type == "velocity_region")
                {
                    scene.sources.push_back(
                        std::make_unique<VelocityRegion>(
                            region,
                            ReadMaterial(
                                source["material"].as<std::string>()),
                            velocity,
                            begin,
                            end));
                }
                else
                {
                    throw std::runtime_error(fmt::format(
                        "unsupported source type '{}'", type));
                }
            }
        }
        return scene;
    }
}

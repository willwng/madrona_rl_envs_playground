#pragma once

#include <madrona/taskgraph.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace Cartpole {

    struct RendererInitStub {};

    // 3D Position & Quaternion Rotation
    // These classes are defined in madrona/components.hpp
    using madrona::base::Position;
    using madrona::base::Rotation;

    class Engine;

    struct WorldReset {
        int32_t resetNow;
    };

    struct Action {
        int32_t choice; // Binary Action
    };

    struct State {
        float x;
        float x_dot;
        float theta;
        float theta_dot;
    };

    struct Reward {
        float rew;
    };

    struct Agent : public madrona::Archetype<WorldReset, Action, State, Reward> {};

    struct Config {};

    struct Sim : public madrona::WorldBase {
        static void registerTypes(madrona::ECSRegistry &registry, const Config &cfg);

        static void setupTasks(madrona::TaskGraph::Builder &builder, const Config &cfg);

        Sim(Engine &ctx, const Config& cfg, const WorldInit &init);

        EpisodeManager *episodeMgr;
        RNG rng;

        madrona::Entity *agents;
    };

    class Engine : public ::madrona::CustomContext<Engine, Sim> {
        using CustomContext::CustomContext;
    };

}

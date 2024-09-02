#pragma once

#include <madrona/taskgraph_builder.hpp>
#include <madrona/math.hpp>
#include <madrona/custom_context.hpp>
#include <madrona/components.hpp>

#include "init.hpp"
#include "rng.hpp"

namespace Acrobat {

  class Engine;

  enum class ExportID : uint32_t {
    WorldReset,
    Action,
    State,
    Reward,
    WorldID,
    NumExports,
  };

  struct WorldReset {
    int32_t resetNow;
  };

  // | Num | Action                                | Unit         |
  // |-----|---------------------------------------|--------------|
  // | 0   | apply -1 torque to the actuated joint | torque (N m) |
  // | 1   | apply 0 torque to the actuated joint  | torque (N m) |
  // | 2   | apply 1 torque to the actuated joint  | torque (N m) |
  struct Action {
    int32_t choice; // 3 Actions: (torque -1, 0, 1)
  };

  // Observation Space:
  // | Num | Observation                  | Min                 | Max               |
  // |-----|------------------------------|---------------------|-------------------|
  // | 0   | Cosine of `theta1`           | -1                  | 1                 |
  // | 1   | Sine of `theta1`             | -1                  | 1                 |
  // | 2   | Cosine of `theta2`           | -1                  | 1                 |
  // | 3   | Sine of `theta2`             | -1                  | 1                 |
  // | 4   | Angular velocity of `theta1` | ~ -12.567 (-4 * pi) | ~ 12.567 (4 * pi) |
  // | 5   | Angular velocity of `theta2` | ~ -28.274 (-9 * pi) | ~ 28.274 (9 * pi) |
  struct State {
    float theta1;
    float theta2;
    float omega1;
    float omega2;
  };

  struct Reward {
    float rew;
  };

  struct Agent : public madrona::Archetype<WorldReset, Action, State, Reward> {};

  struct Config {};

  struct Sim : public madrona::WorldBase {
    static void registerTypes(madrona::ECSRegistry &registry, const Config &cfg);

    static void setupTasks(madrona::TaskGraphBuilder &builder, const Config &cfg);

    Sim(Engine &ctx, const Config& cfg, const WorldInit &init);

    EpisodeManager *episodeMgr;
    RNG rng;

    madrona::Entity *agents;
  };

  class Engine : public ::madrona::CustomContext<Engine, Sim> {
    using CustomContext::CustomContext;
  };

}

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

    // Differs from observation space
    // | Num | State                        | Unit  |
    // |-----|------------------------------|-------|
    // | 0   | theta1                       | rad   |
    // | 1   | theta2                       | rad   |
    // | 2   | Angular velocity of `theta1` | rad/s |
    // | 3   | Angular velocity of `theta2` | rad/s |
  struct State {
    float theta1;
    float theta2;
    float omega1;
    float omega2;
  };

  // The goal is to have the free end reach a designated target height in as few steps as possibl        e,
  //  and as such all steps that do not reach the goal incur a reward of -1.
  //  Achieving the target height results in termination with a reward of 0. The reward threshold is -100.
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

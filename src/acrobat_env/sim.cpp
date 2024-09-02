#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

#include <cmath>

using namespace madrona;
using namespace madrona::math;

namespace Acrobat
{

  void Sim::registerTypes(ECSRegistry &registry, const Config &)
  {
    base::registerTypes(registry);

    registry.registerComponent<WorldReset>();
    registry.registerComponent<Action>();
    registry.registerComponent<State>();
    registry.registerComponent<Reward>();

    // Export tensors for pytorch
    registry.registerArchetype<Agent>();
    registry.exportColumn<Agent, WorldReset>((uint32_t)ExportID::WorldReset);
    registry.exportColumn<Agent, Action>((uint32_t)ExportID::Action);
    registry.exportColumn<Agent, State>((uint32_t)ExportID::State);
    registry.exportColumn<Agent, Reward>((uint32_t)ExportID::Reward);
    registry.exportColumn<Agent, WorldID>((uint32_t)ExportID::WorldID);
  }

  static void resetWorld(Engine &ctx)
  {
    // Update the RNG seed for a new episode
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    uint32_t episode_idx = episode_mgr.curEpisode.fetch_add_relaxed(1);
    ctx.data().rng = RNG::make(episode_idx);

    Entity agent = ctx.data().agents[0];

    // Each parameter in the underlying state (theta1, theta2, and the two angular velocities)
    //   is initialized uniformly between -0.1 and 0.1. This means both links are pointing
    //   downwards with some initial stochasticity.
    const math::Vector2 bounds{-0.1f, 0.1f};
    float bounds_diff = bounds.y - bounds.x;
    ctx.get<State>(agent) = {
        bounds.x + ctx.data().rng.rand() * bounds_diff,
        bounds.x + ctx.data().rng.rand() * bounds_diff,
        bounds.x + ctx.data().rng.rand() * bounds_diff,
        bounds.x + ctx.data().rng.rand() * bounds_diff};
  }

  inline void actionSystem(Engine &, Action &action, State &state, Reward &reward)
  {

    constexpr float LINK_LENGTH_1 = 1.0;  // [m]
    constexpr float LINK_LENGTH_2 = 1.0;  // [m]
    constexpr float LINK_MASS_1 = 1.0;    // [kg] mass of link 1
    constexpr float LINK_MASS_2 = 1.0;    // [kg] mass of link 2
    constexpr float LINK_COM_POS_1 = 0.5; // [m] position of the center of mass of link 1
    constexpr float LINK_COM_POS_2 = 0.5; // [m] position of the center of mass of link 2
    constexpr float LINK_MOI = 1.0;       // moments of inertia for both links

    constexpr float MAX_VEL_1 = 4 * pi;
    constexpr float MAX_VEL_2 = 9 * pi;

    constexpr Vector3 AVAIL_TORQUE = {-1.0, 0.0, +1};

    float roque_noise_max = 0.0;
    reward.rew = 1.f;
  }

  inline void checkDone(Engine &ctx, WorldReset &reset, State &state)
  {
    // The episode ends if one of the following occurs:
    //  1. Termination: The free end reaches the target height, which is constructed as: -cos(theta1) - cos(theta2 + theta1) > 1.0
    //  2. Truncation: Episode length is greater than 500 (200 for v0)

    float theta1 = state.theta1;
    float theta2 = state.theta2;
    bool termination = -cosf(theta1) - cosf(theta2 + theta1) > 1.0;
    bool truncation = false;
    reset.resetNow = termination || truncation;

    if (reset.resetNow)
    {
      resetWorld(ctx);
    }
  }

  void Sim::setupTasks(TaskGraphBuilder &builder, const Config &)
  {
    auto action_sys = builder.addToGraph<ParallelForNode<Engine, actionSystem,
                                                         Action, State, Reward>>({});

    auto terminate_sys = builder.addToGraph<ParallelForNode<Engine, checkDone, WorldReset, State>>({action_sys});
  }

  Sim::Sim(Engine &ctx, const Config &, const WorldInit &init)
      : WorldBase(ctx),
        episodeMgr(init.episodeMgr)
  {
    // Make a buffer that will last the duration of simulation for storing
    // agent entity IDs
    agents = (Entity *)rawAlloc(sizeof(Entity));

    agents[0] = ctx.makeEntity<Agent>();

    ctx.get<Action>(agents[0]).choice = 0;
    ctx.get<State>(agents[0]).theta1 = 0.f;
    ctx.get<State>(agents[0]).theta2 = 0.f;
    ctx.get<State>(agents[0]).omega1 = 0.f;
    ctx.get<State>(agents[0]).omega2 = 0.f;
    ctx.get<Reward>(agents[0]).rew = 0.f;

    // Initial reset
    resetWorld(ctx);
    ctx.get<WorldReset>(agents[0]).resetNow = false;
  }

  MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}

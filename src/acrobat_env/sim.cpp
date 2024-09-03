#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

#include <cmath>

using namespace madrona;
using namespace madrona::math;

namespace Acrobat
{

  constexpr float dt = 0.2; // RK4 integration timestep

  constexpr float LINK_LENGTH_1 = 1.0;  // [m]
  constexpr float LINK_LENGTH_2 = 1.0;  // [m]
  constexpr float LINK_MASS_1 = 1.0;    // [kg] mass of link 1
  constexpr float LINK_MASS_2 = 1.0;    // [kg] mass of link 2
  constexpr float LINK_COM_POS_1 = 0.5; // [m] position of the center of mass of link 1
  constexpr float LINK_COM_POS_2 = 0.5; // [m] position of the center of mass of link 2
  constexpr float LINK_MOI = 1.0;       // moments of inertia for both links

  constexpr float MAX_VEL_1 = 4 * pi;
  constexpr float MAX_VEL_2 = 9 * pi;

  using FloatArray5 = std::array<float, 5>;

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
    // Reset episode length
    episode_mgr.episodeLength = 0;

    // Reset agent state
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

  FloatArray5 ds_dt(FloatArray5 s_augmented)
  {
    float m1 = LINK_MASS_1;
    float m2 = LINK_MASS_2;
    float l1 = LINK_LENGTH_1;
    float lc1 = LINK_COM_POS_1;
    float lc2 = LINK_COM_POS_2;
    float I1 = LINK_MOI;
    float I2 = LINK_MOI;
    float g = 9.8;

    float theta1 = s_augmented[0];
    float theta2 = s_augmented[1];
    float dtheta1 = s_augmented[2];
    float dtheta2 = s_augmented[3];
    float a = s_augmented[4];

    float d1 = (m1 * powf(lc1, 2) + m2 * (powf(l1, 2) + powf(lc2, 2) + 2 * l1 * lc2 * cosf(theta2)) + I1 + I2);
    float d2 = m2 * (powf(lc2, 2) + l1 * lc2 * cos(theta2)) + I2;
    float phi2 = m2 * lc2 * g * cosf(theta1 + theta2 - pi / 2.0);
    float phi1 = (-m2 * l1 * lc2 * powf(dtheta2, 2) * sinf(theta2) - 2 * m2 * l1 * lc2 * dtheta2 * dtheta1 * sinf(theta2) + (m1 * lc1 + m2 * l1) * g * cosf(theta1 - pi / 2) + phi2);
    // use dynamics from the book
    float ddtheta2 = (a + d2 / d1 * phi1 - m2 * l1 * lc2 * powf(dtheta1, 2) * sinf(theta2) - phi2) / (m2 * powf(lc2, 2) + I2 - powf(d2, 2) / d1);
    float ddtheta1 = -(d2 * ddtheta2 + phi1) / d1;

    return {dtheta1, dtheta2, ddtheta1, ddtheta2, 0.f};
  }

  inline FloatArray5 mul_const(FloatArray5 x, float c)
  {
    FloatArray5 out = x; // clone
    for (float &v : out)
    {
      v *= c;
    }
    return out;
  }

  inline FloatArray5 add(FloatArray5 x1, FloatArray5 x2)
  {
    FloatArray5 out = x1; // clone
    for (int i = 0; i < 5; i++)
    {
      out[i] += x2[i];
    }
    return out;
  }
  // Integrate system using 4-th order Runge-Kutta.
  inline Vector4 rk4(std::function<FloatArray5(FloatArray5)> derivs, FloatArray5 y0, std::vector<float> t)
  {
    // Allocate the output vector
    FloatArray5 yout[t.size()]; // state for each timestep
    // set first timestep to y0
    for (int i = 0; i < 5; i++)
    {
      yout[0][i] = y0[i];
    }

    // integrate
    for (int i = 0; i < t.size() - 1; i++)
    {
      float tis = t.at(i);
      float dt = t.at(i + 1) - tis;
      float dt2 = dt / 2.0;
      float dt6 = dt / 6.0;
      FloatArray5 y0 = yout[i];

      FloatArray5 k1 = derivs(y0);
      FloatArray5 k2 = derivs(add(y0, mul_const(k1, dt2)));
      FloatArray5 k3 = derivs(add(y0, mul_const(k2, dt2)));
      FloatArray5 k4 = derivs(add(y0, mul_const(k3, dt)));
      // yout[i + 1] = y0 + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
      yout[i + 1] = add(y0, mul_const(add(k1, add(mul_const(k2, 2), add(mul_const(k3, 2), k4))), dt6));
    }
    // Return final timestep, removing final action term
    FloatArray5 yl = yout[t.size() - 1];
    return {yl[0], yl[1], yl[2], yl[3]};
  }

  inline float wrap(float x, float m, float M)
  {
    float diff = M - m;
    while (x > M)
    {
      x -= diff;
    }
    while (x < m)
    {
      x += diff;
    }
    return x;
  }

  inline float bound(float x, float m, float M)
  {
    return std::min(std::max(x, m), M);
  }

  inline void actionSystem(Engine &ctx, Action &action, State &state, Reward &reward)
  {
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    episode_mgr.episodeLength++;

    constexpr Vector3 AVAIL_TORQUE = {-1.0, 0.0, +1};

    float torque_noise_max = 0.0; // ignore noise for now
    float torque = AVAIL_TORQUE[action.choice];
    FloatArray5 s_augmented = {state.theta1, state.theta2, state.omega1, state.omega2, torque};
    std::vector<float> time = {0.0, dt};
    Vector4 ns = rk4(&ds_dt, s_augmented, time);

    // Update state, with wrapping and bounding
    state.theta1 = wrap(ns[0], -pi, pi);
    state.theta2 = wrap(ns[1], -pi, pi);
    state.omega1 = bound(ns[2], -MAX_VEL_1, MAX_VEL_1);
    state.omega2 = bound(ns[3], -MAX_VEL_2, MAX_VEL_2);

    // only reward of 0 when terminated
    reward.rew = -1.f;
  }

  inline void checkDone(Engine &ctx, WorldReset &reset, State &state)
  {
    // The episode ends if one of the following occurs:
    //  1. Termination: The free end reaches the target height, which is constructed as: -cos(theta1) - cos(theta2 + theta1) > 1.0
    float theta1 = state.theta1;
    float theta2 = state.theta2;
    bool termination = -cosf(theta1) - cosf(theta2 + theta1) > 1.0;

    //  2. Truncation: Episode length is greater than 500 (200 for v0)
    EpisodeManager &episode_mgr = *ctx.data().episodeMgr;
    bool truncation = episode_mgr.episodeLength > 500;
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

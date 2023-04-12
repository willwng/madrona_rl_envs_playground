#include "sim.hpp"
#include <madrona/mw_gpu_entry.hpp>

#include<cmath>

using namespace madrona;
using namespace madrona::math;


namespace Overcooked {

    
void Sim::registerTypes(ECSRegistry &registry, const Config &cfg)
{
    base::registerTypes(registry);

    registry.registerSingleton<WorldReset>();
    registry.registerSingleton<WorldState>();
    
    registry.registerComponent<Action>();
    // registry.registerSingleton<Observation>();
    registry.registerComponent<PlayerState>();
    registry.registerComponent<AgentID>();
    registry.registerComponent<ActionMask>();
    registry.registerComponent<ActiveAgent>();
    registry.registerComponent<Reward>();

    registry.registerComponent<LocationObservation>();
    registry.registerComponent<LocationID>();

    registry.registerFixedSizeArchetype<Agent>(cfg.num_players);
    registry.registerFixedSizeArchetype<LocationType>(cfg.width * cfg.height);

    // Export tensors for pytorch
    registry.exportSingleton<WorldReset>(0);
    registry.exportColumn<Agent, ActiveAgent>(1);
    registry.exportColumn<Agent, Action>(2);
    registry.exportColumn<Agent, ActionMask>(4);
    registry.exportColumn<Agent, Reward>(5);
    registry.exportColumn<Agent, WorldID>(6);
    registry.exportColumn<Agent, AgentID>(7);

    registry.exportColumn<LocationType, LocationObservation>(3);
    registry.exportColumn<LocationType, WorldID>(8);
    registry.exportColumn<LocationType, LocationID>(9);
}
    inline TerrainT get_terrain(WorldState &ws, int32_t pos)
    {
        return ws.terrain[pos];
    }

    inline int32_t get_time(WorldState &ws, Object &soup)
    {
        return ws.recipe_times[soup.get_recipe()];
    }

    inline bool is_cooking(WorldState &ws, Object &soup)
    {
        return soup.cooking_tick >= 0 && soup.cooking_tick < get_time(ws, soup);
    }

    inline bool is_ready(WorldState &ws, Object &soup)
    {
        return soup.cooking_tick >= 0 && soup.cooking_tick >= get_time(ws, soup);
    }

    inline void observationSystem(Engine &ctx, LocationID&id, LocationObservation& obs)
{
    WorldState &ws = ctx.getSingleton<WorldState>();

    int32_t shift = 5 * ws.num_players;

    int pos = id.id;

    Object &obj = ws.objects[pos];

    if (ws.horizon - ws.timestep < 40) {
        obs.x[shift + 15] = 1;
    } else {
        obs.x[shift + 15] = 0;
    }

    for (int to_reset = shift - shift; to_reset < shift; to_reset++) {
        obs.x[to_reset] = 0;
    }
        
    obs.x[shift + 6] = 0;
    obs.x[shift + 7] = 0;
    obs.x[shift + 8] = 0;
    obs.x[shift + 9] = 0;
    obs.x[shift + 10] = 0;
    obs.x[shift + 11] = 0;
    obs.x[shift + 12] = 0;
    obs.x[shift + 13] = 0;
    obs.x[shift + 14] = 0;

    if (obj.name == ObjectT::SOUP) {
        if (ws.terrain[pos] == TerrainT::POT) {
            if (obj.cooking_tick < 0) {
                obs.x[shift + 6] = obj.num_onions;
                obs.x[shift + 7] = obj.num_tomatoes;
            } else {
                obs.x[shift + 8] = obj.num_onions;
                obs.x[shift + 9] = obj.num_tomatoes;
                obs.x[shift + 10] = get_time(ws, obj) - obj.cooking_tick;
                if (is_ready(ws, obj)) {
                    obs.x[shift + 11] = 1;
                }
            }
        } else {
            obs.x[shift + 8] = obj.num_onions;
            obs.x[shift + 9] = obj.num_tomatoes;
            obs.x[shift + 10] = 0;
            obs.x[shift + 11] = 1;
        }
    } else if (obj.name == ObjectT::DISH) {
        obs.x[shift + 12] = 1;
    } else if (obj.name == ObjectT::ONION) {
        obs.x[shift + 13] = 1;
    } else if (obj.name == ObjectT::TOMATO) {
        obs.x[shift + 14] = 1;
    }

    int other_i = 1;
    for (int i = 0; i < ws.num_players; i++) {
        PlayerState &ps = ctx.getUnsafe<PlayerState>(ctx.data().agents[i]);

        int32_t pos2 = ps.position;
        if (pos2 != pos) {
            continue;
        }
        
        if (i == 0) {
            obs.x[0] = 1;
            obs.x[ws.num_players + ps.orientation] = 1;
        } else {
            obs.x[other_i] = 1;
            obs.x[ws.num_players + 4 * other_i + ps.orientation] = 1;
            other_i++;
        }

        if (ps.has_object()) {
            Object &obj2 = ps.get_object();
            if (obj2.name == ObjectT::SOUP) {
                obs.x[shift + 8] = obj2.num_onions;
                obs.x[shift + 9] = obj2.num_tomatoes;
                obs.x[shift + 10] = 0;
                obs.x[shift + 11] = 1;
            } else if (obj2.name == ObjectT::DISH) {
                obs.x[shift + 12] = 1;
            } else if (obj2.name == ObjectT::ONION) {
                obs.x[shift + 13] = 1;
            } else if (obj2.name == ObjectT::TOMATO) {
                obs.x[shift + 14] = 1;
            }
        }
    }
}

    inline bool is_dish_pickup_useful(Engine &ctx, WorldState &ws, int32_t non_empty_pots)
    {
        if (ws.num_players != 2) {
            return false;
        }

        int32_t num_player_dishes = 0;
        for (int i = 0; i < ws.num_players; i++) {
            PlayerState &ps = ctx.getUnsafe<PlayerState>(ctx.data().agents[i]);
            if (ps.has_object() && ps.get_object().name == ObjectT::DISH) {
                num_player_dishes++;
            }
        }

        // OPTIMIZE: Iterate over COUNTER
        for (int i = 0; i < ws.num_counters; i++) {
            int pos = ws.counter_locs[i];
            Object &obj = ws.objects[pos];
            if (obj.name == ObjectT::DISH) {
                return false;
            }
        }
        return num_player_dishes < non_empty_pots;
    }

    inline int32_t get_pot_states(WorldState &ws)
    {
        int32_t non_empty_pots = 0;

        // OPTIMIZE: Iterate over POT
        for (int i = 0; i < ws.num_pots; i++) {
            int pos = ws.pot_locs[i];
            if (ws.objects[pos].name != ObjectT::NONE) {
                Object &soup = ws.objects[pos];
                if (soup.cooking_tick >= 0 || soup.num_ingredients() < MAX_NUM_INGREDIENTS) {
                    non_empty_pots++;
                }
            } 
        }
        return non_empty_pots;
    }

    inline int32_t deliver_soup(WorldState &ws, PlayerState &ps, Object &soup)
    {
        ps.remove_object();
        return ws.recipe_values[soup.get_recipe()];
    }

    inline bool soup_to_be_cooked_at_location(WorldState &ws, int32_t pos)
    {
        Object &obj = ws.objects[pos];
        return obj.name == ObjectT::SOUP && !is_cooking(ws, obj) && !is_ready(ws, obj) && obj.num_ingredients() > 0;
    }

    inline bool soup_ready_at_location(WorldState &ws, int32_t pos)
    {
        return ws.objects[pos].name == ObjectT::SOUP && is_ready(ws, ws.objects[pos]);
    }

    inline int32_t move_in_direction(int32_t point, int32_t direction, int64_t width)
    {
        if (direction == ActionT::NORTH) {
            return point - width;
        } else if (direction == ActionT::SOUTH) {
            return point + width;
        } else if (direction == ActionT::EAST) {
            return point + 1;
        } else if (direction == ActionT::WEST) {
            return point - 1;
        }
        return point;
    }

    inline void resolve_interacts(Engine &ctx, WorldState &ws)
    {
        int32_t pot_states = get_pot_states(ws);

        for (int i = 0; i < ws.num_players; i++) {
            Reward &reward = ctx.getUnsafe<Reward>(ctx.data().agents[i]);
            PlayerState &player = ctx.getUnsafe<PlayerState>(ctx.data().agents[i]);
            Action &action = ctx.getUnsafe<Action>(ctx.data().agents[i]);

            reward.rew = 0;

            if (action.choice != ActionT::INTERACT) {
                continue;
            }

            int32_t pos = player.position;
            int32_t o = player.orientation;

            int32_t i_pos = move_in_direction(pos, o, ws.width);
            TerrainT terrain_type = ws.terrain[i_pos];

            if (terrain_type == TerrainT::COUNTER) {
                if (player.has_object() && ws.objects[i_pos].name == ObjectT::NONE) {
                    ws.objects[i_pos] = player.remove_object();
                } else if (!player.has_object() && ws.objects[i_pos].name != ObjectT::NONE) {
                    player.set_object(ws.objects[i_pos]);
                    ws.objects[i_pos] = { .name = ObjectT::NONE };
                }
            } else if (terrain_type == TerrainT::ONION_SOURCE) {
                if (player.held_object.name == ObjectT::NONE) {
                    player.held_object = { .name = ObjectT::ONION };
                }
            } else if (terrain_type == TerrainT::TOMATO_SOURCE) {
                if (player.held_object.name == ObjectT::NONE) {
                    player.held_object = { .name = ObjectT::TOMATO };
                }
            } else if (terrain_type == TerrainT::DISH_SOURCE) {
                if (player.held_object.name == ObjectT::NONE) {
                    if (is_dish_pickup_useful(ctx, ws, pot_states)) {
                        reward.rew += ws.dish_pickup_rew;
                    }
                    player.held_object = { .name = ObjectT::DISH };
                }
            } else if (terrain_type == TerrainT::POT) {
                if (!player.has_object()) {
                    if (soup_to_be_cooked_at_location(ws, i_pos)) {
                        ws.objects[i_pos].cooking_tick = 0;
                    }
                } else {
                    if (player.get_object().name == ObjectT::DISH && soup_ready_at_location(ws, i_pos)) {
                        player.set_object(ws.objects[i_pos]);
                        ws.objects[i_pos] = { .name = ObjectT::NONE };
                        reward.rew += ws.soup_pickup_rew;
                    } else if (player.get_object().name == ObjectT::ONION || player.get_object().name == ObjectT::TOMATO) {
                        if (ws.objects[i_pos].name == ObjectT::NONE) {
                            ws.objects[i_pos] = { .name = ObjectT::SOUP };
                        }

                        Object &soup = ws.objects[i_pos];
                        if (!(soup.cooking_tick >= 0 || soup.num_ingredients() == MAX_NUM_INGREDIENTS)) {
                            Object obj = player.remove_object();
                            if (obj.name == ObjectT::ONION) {
                                soup.num_onions++;
                            } else {
                                soup.num_tomatoes++;
                            }
                            reward.rew += ws.placement_in_pot_rew;
                        }
                    }
                }
            } else if (terrain_type == TerrainT::SERVING) {
                if (player.has_object()) {
                    Object obj = player.get_object();
                    if (obj.name == ObjectT::SOUP) {
                        reward.rew += deliver_soup(ws, player, obj);
                    }
                }
            }

        }
    }

    inline void _move_if_direction(Engine &ctx, PlayerState &ps, Action &action)
    {
        if (action.choice == ActionT::INTERACT) {
            ps.propose_pos_and_or(ps.position, ps.orientation);
        } else {
            WorldState &ws = ctx.getSingleton<WorldState>();
            
            int32_t new_pos = move_in_direction(ps.position, action.choice, ws.width);

            int32_t new_orientation = (action.choice == ActionT::STAY ? ps.orientation : (int32_t) action.choice);
            ps.propose_pos_and_or((ws.terrain[new_pos] != TerrainT::AIR ? ps.position : new_pos), new_orientation);
        }
    }

    inline void _handle_collisions(Engine &ctx, WorldState &ws)
    {
        for (int idx0 = 0; idx0 < ws.num_players; idx0++) {
            for (int idx1 = idx0+1; idx1 < ws.num_players; idx1++) {
                PlayerState &ps0 = ctx.getUnsafe<PlayerState>(ctx.data().agents[idx0]);
                PlayerState &ps1 = ctx.getUnsafe<PlayerState>(ctx.data().agents[idx1]);

                if (ps0.proposed_position == ps1.proposed_position ||
                    (ps0.proposed_position == ps1.position && ps1.proposed_position == ps0.position)) {
                    for (int i = 0; i < ws.num_players; i++) {
                        ctx.getUnsafe<PlayerState>(ctx.data().agents[i]).update_or();
                    }
                    return;
                }
            }
        }

        for (int i = 0; i < ws.num_players; i++) {
            ctx.getUnsafe<PlayerState>(ctx.data().agents[i]).update_pos_and_or();
        }
    }

    inline void step_environment_effects(Engine &ctx, WorldState &ws)
    {
        ws.timestep += 1;

        // OPTIMIZE: ONLY Iterate over POT
        for (int i = 0; i < ws.num_pots; i++) {
            int pos = ws.pot_locs[i];
            Object &obj = ws.objects[pos];
            if (obj.name == ObjectT::SOUP && is_cooking(ws, obj)) {
                obj.cooking_tick++;
            }
        }
        
        // calculate reward here
        ws.calculated_reward = 0;
        for (int i = 0; i < ws.num_players; i++) {
            ws.calculated_reward += ctx.getUnsafe<Reward>(ctx.data().agents[i]).rew;
        }
    }    

    static void resetWorld(Engine &ctx)
{
    WorldState &ws = ctx.getSingleton<WorldState>();
    
    ws.timestep = 0;
    for (int i = 0; i < ws.size; i++) {
        ws.objects[i] = { .name = ObjectT::NONE };
    }

    for (int i = 0; i < ws.num_players; i++) {
        PlayerState &p = ctx.getUnsafe<PlayerState>(ctx.data().agents[i]);
        p.position = ws.start_player_y[i] * ws.width + ws.start_player_x[i];
        p.orientation = ActionT::NORTH;
        p.proposed_position = p.position;
        p.proposed_orientation = p.orientation;
        
        p.held_object = { .name = ObjectT::NONE };
    }
}

    inline void check_reset_system(Engine &ctx, WorldState &ws)
    {
        ctx.getSingleton<WorldReset>().resetNow = (ws.timestep >= ws.horizon);
        for (int i = 0; i < ws.num_players; i++) {
            ctx.getUnsafe<Reward>(ctx.data().agents[i]).rew = ws.calculated_reward;
        }

        if (ctx.getSingleton<WorldReset>().resetNow) {
            resetWorld(ctx);
        }
    }

    

void Sim::setupTasks(TaskGraph::Builder &builder, const Config &cfg)
{

    auto interact_sys = builder.addToGraph<ParallelForNode<Engine, resolve_interacts, WorldState>>({});

    auto move_sys = builder.addToGraph<ParallelForNode<Engine, _move_if_direction, PlayerState, Action>>({});

    auto collision_sys = builder.addToGraph<ParallelForNode<Engine, _handle_collisions, WorldState>>({move_sys});

    auto env_step_sys = builder.addToGraph<ParallelForNode<Engine, step_environment_effects, WorldState>>({interact_sys, collision_sys});
    
    auto terminate_sys = builder.addToGraph<ParallelForNode<Engine, check_reset_system, WorldState>>({env_step_sys});

    auto obs_sys = builder.addToGraph<ParallelForNode<Engine, observationSystem, LocationID, LocationObservation>>({terminate_sys});

    (void)obs_sys;
}


Sim::Sim(Engine &ctx, const Config& cfg, const WorldInit &init)
    : WorldBase(ctx),
      episodeMgr(init.episodeMgr)
{
    // Make a buffer that will last the duration of simulation for storing
    // agent entity IDs
    agents = (Entity *)rawAlloc(cfg.num_players * sizeof(Entity));

    locations = (Entity *)rawAlloc(cfg.width * cfg.height * sizeof(Entity));

    WorldState &ws = ctx.getSingleton<WorldState>();

    for (int r = 0; r < NUM_RECIPES; r++) {
        ws.recipe_values[r] = cfg.recipe_values[r];
        ws.recipe_times[r] = cfg.recipe_times[r];
    }

    ws.height = cfg.height;
    ws.width = cfg.width;
    ws.size = cfg.height * cfg.width;

    ws.num_pots = 0;
    ws.num_counters = 0;
    
    for (int x = 0; x < ws.size; x++) {
        ws.terrain[x] = (TerrainT) cfg.terrain[x];

        if (ws.terrain[x] == TerrainT::POT) {
            ws.pot_locs[ws.num_pots++] = x;
        } else if (ws.terrain[x] == TerrainT::COUNTER) {
            ws.counter_locs[ws.num_counters++] = x;
        }
    }
    ws.num_players = cfg.num_players;
    
    for (int p = 0; p < ws.num_players; p++) {
        ws.start_player_x[p] = cfg.start_player_x[p];
        ws.start_player_y[p] = cfg.start_player_y[p];
    }

    ws.horizon = cfg.horizon;
    ws.placement_in_pot_rew = cfg.placement_in_pot_rew;
    ws.dish_pickup_rew = cfg.dish_pickup_rew;
    ws.soup_pickup_rew = cfg.soup_pickup_rew;
    
    // Set Everything Else
    
    for (int i = 0; i < cfg.num_players; i++) {
        agents[i] = ctx.makeEntityNow<Agent>();
        ctx.getUnsafe<Action>(agents[i]).choice = ActionT::NORTH;
        
        ctx.getUnsafe<AgentID>(agents[i]).id = i;
        for (int t = 0; t < NUM_MOVES; t++) {
            ctx.getUnsafe<ActionMask>(agents[i]).isValid[t] = true;
        }
        ctx.getUnsafe<ActiveAgent>(agents[i]).isActive = true;
        ctx.getUnsafe<Reward>(agents[i]).rew = 0.f;
    }
    
    // Initial reset
    resetWorld(ctx);
    ctx.getSingleton<WorldReset>().resetNow = false;
    
    // SETUP Base Observation
    for (int p = 0; p < cfg.height * cfg.width; p++) {
        locations[p] = ctx.makeEntityNow<LocationType>();
        ctx.getUnsafe<LocationID>(locations[p]).id = p;
        LocationObservation& obs = ctx.getUnsafe<LocationObservation>(locations[p]);

        TerrainT t = (TerrainT) cfg.terrain[p];
        if (t) {
            obs.x[t - 1 + 5 * cfg.num_players] = 1;
        }

        observationSystem(ctx, ctx.getUnsafe<LocationID>(locations[p]), obs);

    }
}

    MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}

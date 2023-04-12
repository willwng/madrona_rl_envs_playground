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
    registry.registerComponent<LocationData>();

    registry.registerComponent<PotInfo>();

    registry.registerFixedSizeArchetype<Agent>(cfg.num_players);
    registry.registerFixedSizeArchetype<LocationType>(cfg.width * cfg.height);


    int num_pots = 0;
    for (int x = 0; x < cfg.height * cfg.width; x++) {
        if (cfg.terrain[x] == TerrainT::POT) {
            num_pots++;
        }
    }
    registry.registerFixedSizeArchetype<PotType>(num_pots);


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

    inline void observationSystem(Engine &ctx, LocationID &id, LocationObservation &obs, LocationData &dat)
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

    if (dat.current_player != -1) {
        int i = dat.current_player;
        PlayerState &ps = ctx.getUnsafe<PlayerState>(ctx.data().agents[i]);

        obs.x[i] = 1;
        obs.x[ws.num_players + 4 * i + ps.orientation] = 1;

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
    
    inline void pre_resolve_interacts(Engine &, WorldState &ws)
    {
        ws.calculated_reward.store_relaxed(0);
    }
    
    inline void resolve_interacts(Engine &ctx, WorldState &ws)
    {
        for (int i = 0; i < ws.num_players; i++) {
            PlayerState &player = ctx.getUnsafe<PlayerState>(ctx.data().agents[i]);
            Action &action = ctx.getUnsafe<Action>(ctx.data().agents[i]);

            if (action.choice != ActionT::INTERACT) {
                continue;
            }

            int32_t pos = player.position;
            int32_t o = player.orientation;

            int32_t i_pos = move_in_direction(pos, o, ws.width);
            TerrainT terrain_type = ws.terrain[i_pos];

            if (terrain_type == TerrainT::COUNTER) {
                // TODO: parallel pick and place without conflicts
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
                    player.held_object = { .name = ObjectT::DISH };
                }
            } else if (terrain_type == TerrainT::POT) {
                // TODO: parallel POT interactions without conflicts
                if (!player.has_object()) {
                    // order doesn't matter if soup_to_be_cooked_at_location
                    if (soup_to_be_cooked_at_location(ws, i_pos)) {
                        ws.objects[i_pos].cooking_tick = 0;
                    }
                } else {
                    if (player.get_object().name == ObjectT::DISH && soup_ready_at_location(ws, i_pos)) {
                        // order matters, only agent with lowest index can take soup
                        player.set_object(ws.objects[i_pos]);
                        ws.objects[i_pos] = { .name = ObjectT::NONE };
                        ws.calculated_reward.fetch_add_relaxed(ws.soup_pickup_rew);
                    } else if (player.get_object().name == ObjectT::ONION || player.get_object().name == ObjectT::TOMATO) {
                        // order matters, only some agents can actually add to pot
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
                            ws.calculated_reward.fetch_add_relaxed(ws.placement_in_pot_rew);
                        }
                    }
                }
            } else if (terrain_type == TerrainT::SERVING) {
                if (player.has_object()) {
                    Object obj = player.get_object();
                    if (obj.name == ObjectT::SOUP) {
                        ws.calculated_reward.fetch_add_relaxed(deliver_soup(ws, player, obj));
                    }
                }
            }

        }
    }




    inline void _move_if_direction(Engine &ctx, PlayerState &ps, Action &action, AgentID &id)
    {
        if (action.choice == ActionT::INTERACT) {
            ps.propose_pos_and_or(ps.position, ps.orientation);
        } else {
            WorldState &ws = ctx.getSingleton<WorldState>();
            
            int32_t new_pos = move_in_direction(ps.position, action.choice, ws.width);

            int32_t new_orientation = (action.choice == ActionT::STAY ? ps.orientation : (int32_t) action.choice);
            ps.propose_pos_and_or((ws.terrain[new_pos] != TerrainT::AIR ? ps.position : new_pos), new_orientation);
        }

        ctx.getUnsafe<LocationData>(ctx.data().locations[ps.proposed_position]).future_player.store_relaxed(id.id);
    }

inline void _check_collisions(Engine &ctx, PlayerState &ps, AgentID &id)
    {
        WorldState &ws = ctx.getSingleton<WorldState>();

        LocationData &origloc = ctx.getUnsafe<LocationData>(ctx.data().locations[ps.position]);
        LocationData &proploc = ctx.getUnsafe<LocationData>(ctx.data().locations[ps.proposed_position]);

        int comp_id = proploc.current_player;
        
        if (proploc.future_player.load_acquire() != id.id) {
            ws.should_update_pos.store_relaxed(false);
            return;
        }
        
        if (comp_id != -1 && comp_id != id.id && origloc.future_player.load_acquire() == comp_id) {
            ws.should_update_pos.store_relaxed(false);
        }
    }

    inline void _unset_loc_info(Engine &ctx, PlayerState &ps)
    {
        ctx.getUnsafe<LocationData>(ctx.data().locations[ps.position]).current_player = -1;
        ctx.getUnsafe<LocationData>(ctx.data().locations[ps.proposed_position]).future_player.store_relaxed(-1);
    }


    inline void _handle_collisions(Engine &ctx, PlayerState &ps, AgentID &id)
    {
        if (ctx.getSingleton<WorldState>().should_update_pos.load_acquire()) {
            ps.update_pos_and_or();
            int new_pos = ps.position;
            ctx.getUnsafe<LocationData>(ctx.data().locations[new_pos]).current_player = id.id;
        } else {
            ps.update_or();
            int new_pos = ps.position;
            ctx.getUnsafe<LocationData>(ctx.data().locations[new_pos]).current_player = id.id;
        }
    }


    inline void step_environment_effects(Engine &, WorldState &ws)
    {
        ws.timestep += 1;
    }

    inline void step_pot_effects(Engine &ctx, PotInfo &pi)
    {
        WorldState &ws = ctx.getSingleton<WorldState>();
        int pos = pi.id;
        Object &obj = ws.objects[pos];
        if (obj.name == ObjectT::SOUP && is_cooking(ws, obj)) {
            obj.cooking_tick++;
        }
    }

    inline void _reset_world_system(Engine &ctx, WorldState &ws)
{
    ws.should_update_pos.store_release(true);
    if (ctx.getSingleton<WorldReset>().resetNow) {
        ws.timestep = 0;
    }
}

    inline void _reset_objects_system(Engine &ctx, LocationID &id)
{
    WorldState &ws = ctx.getSingleton<WorldState>();
    if (ctx.getSingleton<WorldReset>().resetNow) {
        ws.objects[id.id] = { .name = ObjectT::NONE };
    }
}

    inline void _pre_reset_actors_system(Engine &ctx, PlayerState &p)
{
    if (ctx.getSingleton<WorldReset>().resetNow) {
        ctx.getUnsafe<LocationData>(ctx.data().locations[p.position]).current_player = -1;
    }
}

    inline void _reset_actors_system(Engine &ctx, PlayerState &p, AgentID &id)
{
    WorldState &ws = ctx.getSingleton<WorldState>();
    int i = id.id;
    ctx.getUnsafe<Reward>(ctx.data().agents[i]).rew = ws.calculated_reward.load_acquire();
    if (ctx.getSingleton<WorldReset>().resetNow) {
        p.position = ws.start_player_y[i] * ws.width + ws.start_player_x[i];
        ctx.getUnsafe<LocationData>(ctx.data().locations[p.position]).current_player = i;
        p.orientation = ActionT::NORTH;
        p.proposed_position = p.position;
        p.proposed_orientation = p.orientation;
        
        p.held_object = { .name = ObjectT::NONE };
    }
}

    inline void check_reset_system(Engine &ctx, WorldState &ws)
    {
        ctx.getSingleton<WorldReset>().resetNow = (ws.timestep >= ws.horizon);
    }

    

void Sim::setupTasks(TaskGraph::Builder &builder, const Config &)
{
    auto pre_interact_sys = builder.addToGraph<ParallelForNode<Engine, pre_resolve_interacts, WorldState>>({});
    auto interact_sys = builder.addToGraph<ParallelForNode<Engine, resolve_interacts, WorldState>>({pre_interact_sys});

    auto move_sys = builder.addToGraph<ParallelForNode<Engine, _move_if_direction, PlayerState, Action, AgentID>>({});

    // auto collision_sys = builder.addToGraph<ParallelForNode<Engine, _handle_collisions, WorldState>>({move_sys});
    auto check_collision_sys = builder.addToGraph<ParallelForNode<Engine, _check_collisions, PlayerState, AgentID>>({move_sys});
    auto unset_loc_info = builder.addToGraph<ParallelForNode<Engine, _unset_loc_info, PlayerState>>({check_collision_sys});
    auto collision_sys = builder.addToGraph<ParallelForNode<Engine, _handle_collisions, PlayerState, AgentID>>({unset_loc_info});

    
    auto time_step_sys = builder.addToGraph<ParallelForNode<Engine, step_environment_effects, WorldState>>({});
    auto env_step_sys = builder.addToGraph<ParallelForNode<Engine, step_pot_effects, PotInfo>>({interact_sys, collision_sys});
    
    
    auto terminate_sys = builder.addToGraph<ParallelForNode<Engine, check_reset_system, WorldState>>({time_step_sys, env_step_sys});

    auto reset_world_sys = builder.addToGraph<ParallelForNode<Engine, _reset_world_system, WorldState>>({terminate_sys});
    auto reset_obj_sys = builder.addToGraph<ParallelForNode<Engine, _reset_objects_system, LocationID>>({terminate_sys});
    auto pre_reset_actors_sys = builder.addToGraph<ParallelForNode<Engine, _pre_reset_actors_system, PlayerState>>({terminate_sys});
    auto reset_actors_sys = builder.addToGraph<ParallelForNode<Engine, _reset_actors_system, PlayerState, AgentID>>({pre_reset_actors_sys});

    auto obs_sys = builder.addToGraph<ParallelForNode<Engine, observationSystem, LocationID, LocationObservation, LocationData>>({reset_world_sys, reset_obj_sys, reset_actors_sys});

    (void)obs_sys;
}

    static void resetWorld(Engine &ctx)
{
    WorldState &ws = ctx.getSingleton<WorldState>();
    
    _reset_world_system(ctx, ws);
    for (int i = 0; i < ws.size; i++) {
        _reset_objects_system(ctx, ctx.getUnsafe<LocationID>(ctx.data().locations[i]));
    }
    
    for (int i = 0; i < ws.num_players; i++) {
        PlayerState &p = ctx.getUnsafe<PlayerState>(ctx.data().agents[i]);
        _pre_reset_actors_system(ctx, p);
    }
    for (int i = 0; i < ws.num_players; i++) {
        PlayerState &p = ctx.getUnsafe<PlayerState>(ctx.data().agents[i]);
        AgentID &id = ctx.getUnsafe<AgentID>(ctx.data().agents[i]);
        _reset_actors_system(ctx, p, id);
    }
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

    pots = (Entity *)rawAlloc(ws.num_pots * sizeof(Entity));
    for (int i = 0; i < ws.num_pots; i++) {
        pots[i] = ctx.makeEntityNow<PotType>();
        ctx.getUnsafe<PotInfo>(pots[i]).id = ws.pot_locs[i];
    }
    
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
    
    //  Base Observation
    for (int p = 0; p < cfg.height * cfg.width; p++) {
        locations[p] = ctx.makeEntityNow<LocationType>();
        ctx.getUnsafe<LocationID>(locations[p]).id = p;
        ctx.getUnsafe<LocationData>(locations[p]).current_player = -1;
        ctx.getUnsafe<LocationData>(locations[p]).future_player.store_release(-1);
        LocationObservation& obs = ctx.getUnsafe<LocationObservation>(locations[p]);

        TerrainT t = (TerrainT) cfg.terrain[p];
        if (t) {
            obs.x[t - 1 + 5 * cfg.num_players] = 1;
        }
    }

    // Initial reset
    ctx.getSingleton<WorldReset>().resetNow = true;    
    resetWorld(ctx);
    ctx.getSingleton<WorldReset>().resetNow = false;

    for (int p = 0; p < cfg.height * cfg.width; p++) {
        LocationObservation& obs = ctx.getUnsafe<LocationObservation>(locations[p]);
        LocationData& dat = ctx.getUnsafe<LocationData>(locations[p]);
        observationSystem(ctx, ctx.getUnsafe<LocationID>(locations[p]), obs, dat);
    }
}

    MADRONA_BUILD_MWGPU_ENTRY(Engine, Sim, Config, WorldInit);

}

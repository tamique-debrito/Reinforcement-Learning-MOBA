from DisplayElements import DisplayElementProjectileLerpSourceTarget
from Stats import BaseStats, LevelStats
from Units import *

class CooldownAbility:
    def __init__(self, cooldown, cast_hook = None) -> None:
        self.cooldown = cooldown
        self.timer = 0
        self.cast_hook = cast_hook
    
    def cast(self):
        if self.timer <= 0:
            self.timer = self.cooldown
            if self.cast_hook is not None: self.cast_hook()
            return True
        return False

    def step(self):
        self.timer -= TIMESTEP

def targeted_damage_callback(target, damage):
    return lambda: apply_damage(target, damage)

class SimpleMage(Player):
    Q_Cooldown = 4.0
    Q_Time_To_Hit = 1.0
    Q_Sim_Step_To_Hit = Q_Time_To_Hit / TIMESTEP
    Q_Range = 15
    Q_Range_Squared = Q_Range ** 2
    Q_Display_Size = 2.0
    Q_Color = (150, 10, 10)

    def __init__(self, x, y, stats: BaseStats, level_stats: LevelStats, uid, index, team, return_display_element, add_event_hook):
        super().__init__(x, y, stats, level_stats, uid, index, team, return_display_element)
        self.add_event_hook = add_event_hook
        self.Q_cooldown_tracker = CooldownAbility(self.Q_Cooldown)
    
    def cast_Q(self, unitList: List[AnyUnit], x, y):
        target = get_closest_enemy_from_point(unitList, x, y, self.team)
        if target is None:
            return
        if self.get_dist_squared_to_specified_target(target) < self.Q_Range_Squared:
            if self.Q_cooldown_tracker.cast():
                damage = self.stats.attack_damage * 10 # TODO: Implement mana, magic damage, ability power and switch over
                self.add_event_hook(targeted_damage_callback(target, damage), self.Q_Sim_Step_To_Hit) # add a callback event. This uses Simulator.add_callback_event_by_sim_step_delta
                if self.return_display_element:
                    return DisplayElementProjectileLerpSourceTarget(self, target, self.Q_Display_Size, self.Q_Color, self.Q_Sim_Step_To_Hit)
        
    def step(self, unit_list, add_aa_event_hook):
        super().step(unit_list, add_aa_event_hook)
        self.Q_cooldown_tracker.step()
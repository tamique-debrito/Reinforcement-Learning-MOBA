from enum import Enum

from CONSTANTS import *
from dataclasses import dataclass
from typing import Callable, List, Optional, Union
from DisplayElements import BaseDisplayElement
from Stats import BaseStats, LevelStats

class UnitType(Enum):
    MINION = 1
    TURRET = 2
    PLAYER = 3

class UnitState(Enum):
    MOVING = 1
    CHASING = 2
    ATTACKING = 3
    STILL = 4
    CHANNELING = 5

class ActCCState(Enum):
    NONE = 0
    NO_ACT = 1

class MoveCCState(Enum):
    NONE = 0
    NO_MOVE = 1
    CONTROLLED_MOVE = 2

class CCTracker:
    def __init__(self) -> None:
        self.act_cc_state = ActCCState.NONE
        self.move_cc_state = MoveCCState.NONE
        self.act_cc_timer = 0.0
        self.move_cc_timer = 0.0
    
    def add_stun(self, timer):
        self.act_cc_state = ActCCState.NO_ACT
        self.move_cc_state = MoveCCState.NO_MOVE
        self.act_cc_timer = max(timer, self.act_cc_timer)
        self.move_cc_timer = max(timer, self.move_cc_timer)
    
    def step(self):
        if self.act_cc_state != ActCCState.NONE:
            self.act_cc_timer -= TIMESTEP
            if self.act_cc_timer <=0:
                self.act_cc_state = ActCCState.NONE
        
        if self.move_cc_state != MoveCCState.NONE:
            self.move_cc_timer -= TIMESTEP
            if self.move_cc_timer <=0:
                self.act_cc_state = ActCCState.NONE
    
    def has_any_cc(self):
        return self.act_cc_state != ActCCState.NONE or self.move_cc_state != MoveCCState.NONE

class ChannelReason(Enum):
    RECALL = 1

class AAState(Enum):
    READY = 0
    WINDUP = 1
    COOLDOWN = 2

class EventType(Enum):
    AUTOATTACK = 1
    AGGRO = 2
    CALLBACK = 3

@dataclass
class AutoAttackEvent:
    eventType = EventType.AUTOATTACK
    damage: float
    sourceUid: int
    targetUid: int

@dataclass
class CallbackEvent:
    eventType = EventType.CALLBACK
    callback: Callable

class Minion:
    unitType = UnitType.MINION
    aggroRange = MINION_AGGRO_DISTANCE
    def __init__(self, x, y, stats: BaseStats, uid, index, team):
        self.stats = stats
        self.uid = uid
        self.index = index
        self.target: Optional[Minion] = None
        self.x = x
        self.y = y
        self.team = team
        self.state = UnitState.MOVING
        self.aa_state = AAState.READY
        self.last_damaging_uid = None

        self.cc_state = CCTracker()

        self.last_dx = 0.0
        self.last_dy = 0.0
        self.last_d = 0.0

        self.active = True

        self.aa_timer = 0.0

        self.aggro_reset_timer = 0.0

        self.minion_line = MINION_LINE
    
    def can_take_damage(self):
        return True

    def get_dist_to_target(self):
        assert self.target is not None
        dx = self.target.x - self.x
        dy = self.target.y - self.y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        dist_to_edge = dist - self.target.stats.size # This is what should be used for in-range calculations
        return dx, dy, dist, dist_to_edge
    
    def reset_target(self):
        self.target = None
        self.state = UnitState.MOVING
        if self.aa_state == AAState.WINDUP:
            self.aa_state = AAState.READY
    
    def apply_disp(self, dx, dy, d):
        self.last_dx = dx
        self.last_dy = dy
        self.last_d = d
        self.x += dx
        self.y += dy
    
    def out_of_aggro_range(self, dist_to_edge):
        return dist_to_edge > MINION_AGGRO_DISTANCE

    def step(self, unit_list, add_aa_event_hook):
        self.cc_state.step()
        if self.cc_state.act_cc_state == ActCCState.NO_ACT:
            pass
        elif self.state == UnitState.MOVING:
            self.handle_move()
        elif self.state == UnitState.STILL:
            pass # Shouldn't happen for minions
        else:
            if self.target is None or not self.target.active:
                self.reset_target()
            elif self.state == UnitState.CHASING:
                dx, dy, dist, dist_to_edge = self.get_dist_to_target()
                if dist_to_edge < self.stats.attack_range:
                    self.state = UnitState.ATTACKING
                if dist_to_edge > self.stats.size and dist > 0.01:
                    factor = self.stats.speed * TIMESTEP / dist
                    self.apply_disp(dx * factor, dy * factor, self.stats.speed * TIMESTEP)
                if self.out_of_aggro_range(dist_to_edge):
                    self.reset_target()
            elif self.state == UnitState.ATTACKING:
                if self.aa_state == AAState.READY:
                    self.aa_state = AAState.WINDUP
                    self.aa_timer = 0.5 / self.stats.attack_speed
                elif self.aa_state == AAState.WINDUP:
                    self.aa_timer -= TIMESTEP
                    if self.aa_timer < 0:
                        add_aa_event_hook(self.stats.attack_damage, AA_HIT_TIME / self.stats.attack_speed, self.uid, self.target.uid)
                        self.aa_state = AAState.COOLDOWN
                        self.aa_timer = 0.5 / self.stats.attack_speed
                else:
                    self.aa_timer -= TIMESTEP
                    if self.aa_timer < 0:
                        self.aa_state = AAState.READY
            
            self.step_aggro_reset()
            
            if self.target is not None:
                _, _, _, dist_to_edge = self.get_dist_to_target()
                if dist_to_edge > self.stats.attack_range:
                    if self.aa_state == AAState.WINDUP:
                        self.aa_state = AAState.READY
                    self.state = UnitState.CHASING

    def step_aggro_reset(self):
        self.aggro_reset_timer -= TIMESTEP
        if self.aggro_reset_timer < 0:
            self.reset_target()

    def handle_move(self):
        if self.cc_state.move_cc_state == MoveCCState.NO_MOVE:
            return
        diff = self.minion_line - self.x
        move_dist = self.stats.speed * TIMESTEP
        if abs(diff) > LINE_FOLLOW_THRESH:
            if diff > 0:
                self.apply_disp(move_dist, 0.0, move_dist)
            else:
                self.apply_disp( -move_dist, 0.0, move_dist)
        else:
            if self.team == TEAM_A:
                self.apply_disp(0.0, move_dist, move_dist)
            else:
                self.apply_disp(0.0, - move_dist, move_dist)

    def set_aggro_target(self, target):
        if target.team != self.team:
            self.state = UnitState.CHASING
            self.target = target
            self.aggro_reset_timer = AGGRO_RESET_TIME


class Turret(Minion):
    aggroRange = TURRET_AGGRO_DISTANCE
    unitType = UnitType.TURRET

    def __init__(self, x, y, stats: BaseStats, uid, index, team):
        super().__init__(x, y, stats, uid, index, team)
        self.state = UnitState.STILL
        self.stats.attack_range = TURRET_AGGRO_DISTANCE
        self.invulnerable = True

    def can_take_damage(self):
        return not self.invulnerable
    
    def step(self, unit_list, add_aa_event_hook):
        if self.target is not None:
            _, _, dist, dist_to_edge = self.get_dist_to_target()
            if dist_to_edge > self.stats.attack_range or not self.target.active:
                self.reset_target()
        if self.state == UnitState.STILL:
            if self.aa_state == AAState.COOLDOWN:
                self.aa_timer -= TIMESTEP
                if self.aa_timer < 0:
                    self.aa_state = AAState.READY
        elif self.target is None or not self.target.active:
            self.reset_target()
        elif self.state == UnitState.ATTACKING:
            if self.aa_state == AAState.READY:
                self.aa_state = AAState.WINDUP
                self.aa_timer = 0.5 / self.stats.attack_speed
            elif self.aa_state == AAState.WINDUP:
                self.aa_timer -= TIMESTEP
                if self.aa_timer < 0:
                    add_aa_event_hook(self.stats.attack_damage, AA_HIT_TIME / self.stats.attack_speed, self.uid, self.target.uid)
                    self.aa_state = AAState.COOLDOWN
                    self.aa_timer = 0.5 / self.stats.attack_speed
        if self.state != UnitState.ATTACKING:
            if self.aa_state == AAState.WINDUP:
                self.aa_state = AAState.READY
        # Always step cooldown
        if self.aa_state == AAState.COOLDOWN:
            self.aa_timer -= TIMESTEP
            if self.aa_timer < 0:
                self.aa_state = AAState.READY

    
    def reset_target(self):
        self.target = None
        self.state = UnitState.STILL
        if self.aa_state == AAState.WINDUP:
            self.aa_state = AAState.READY

    def set_aggro_target(self, target: Minion):
        if target.team != self.team:
            self.state = UnitState.ATTACKING
            self.target = target
            self.aggro_reset_timer = AGGRO_RESET_TIME



class Player(Minion):
    unitType = UnitType.PLAYER

    def __init__(self, x, y, stats: BaseStats, level_stats: LevelStats, uid, index, team, return_display_element):
        super().__init__(x, y, stats, uid, index, team)
        self.state = UnitState.STILL
        self.move_target_x = 0.0
        self.move_target_y = 0.0
        self.channel_timer = 0.0
        self.channel_reason = None
        self.recall_x = x
        self.recall_y = y

        self.gold = 150
        self.xp = 0
        self.level = 1
        self.level_stats = level_stats

        self.return_display_element = return_display_element

    def step(self, unit_list, add_aa_event_hook):
        if self.state == UnitState.CHANNELING:
            if self.cc_state.has_any_cc():
                self.interrupt_channel()
            self.channel_timer -= TIMESTEP
            if self.channel_timer <= 0:
                self.handle_channel_finish()
        else:
            super().step(unit_list, add_aa_event_hook)

    def start_channel(self, channel_reason: ChannelReason, timer):
        self.channel_reason = channel_reason
        self.channel_timer = timer
        self.state = UnitState.CHANNELING
    
    def interrupt_channel(self):
        if self.state == UnitState.CHANNELING:
            self.channel_reason = None
            self.state = UnitState.STILL

    def handle_channel_finish(self):
        if self.channel_reason == ChannelReason.RECALL:
            self.x = self.recall_x
            self.y = self.recall_y
            self.state = UnitState.STILL

    def handle_move(self):
        dx = self.move_target_x - self.x
        dy = self.move_target_y - self.y
        dist = (dx ** 2 + dy ** 2) ** 0.5
        if dist < 0.01:
            factor = 2
        else:
            factor = self.stats.speed * TIMESTEP / dist
        if factor > 1:
            self.x = self.move_target_x
            self.y = self.move_target_y
            self.state = UnitState.STILL
        else:
            self.apply_disp(dx * factor, dy * factor, self.stats.speed * TIMESTEP)
    
    def set_move_target(self, x, y):
        self.move_target_x = x
        self.move_target_y = y
        self.state = UnitState.MOVING
    
    def set_attack_target(self, target):
        self.target = target
        self.state = UnitState.CHASING
    
    def cast_Q(self, *args, **kwargs) -> Optional[BaseDisplayElement]:
        return None

    def cast_W(self, *args, **kwargs) -> Optional[BaseDisplayElement]:
        return None

    def cast_E(self, *args, **kwargs) -> Optional[BaseDisplayElement]:
        return None

    def cast_R(self, *args, **kwargs) -> Optional[BaseDisplayElement]:
        return None

    def get_dist_squared_to_specified_target(self, target):
        dx = target.x - self.x
        dy = target.y - self.y
        dist_squared = dx ** 2 + dy ** 2
        return dist_squared

    def add_gold_and_xp(self, gold, xp):
        self.gold += gold
        self.xp += xp

    def try_level_up(self):
        next_level = self.level + 1
        advance_levels = 0
        while next_level in XP_THRESHOLDS and self.xp >= XP_THRESHOLDS[next_level]:
            next_level += 1
            advance_levels += 1
        if advance_levels > 0:
            self.level = next_level
            for i in range(advance_levels):
                self.level_stats.apply(self.stats)
        

    def out_of_aggro_range(self, dist_to_edge):
        return False # player doesn't have an aggro range
    
    def step_aggro_reset(self):
        pass # doesn't apply to player

AnyUnit = Union[Minion, Turret, Player]


def calc_damage(stats, damage):
    return damage * 100 / (100 + stats.armor)

def apply_damage(target: Minion, damage):
    dmg = calc_damage(target.stats, damage)
    target.stats.health -= dmg
    return dmg

def get_closest_enemy_from_point(unitList: List[AnyUnit], x, y, team):
    target = None
    shortest_dist_squared = None
    for unit in unitList:
        if unit.team != team:
            dist_squared = (unit.x - x) ** 2 + (unit.y - y) ** 2
            if shortest_dist_squared is None or dist_squared < shortest_dist_squared:
                shortest_dist_squared = dist_squared
                target = unit
    return target


def get_vec_k_minions(unitList: List[AnyUnit], x, y, k):
    minions_by_distance = []
    for unit in unitList:
        if unit.unitType == UnitType.MINION:
            dist_squared = (unit.x - x) ** 2 + (unit.y - y) ** 2
            minions_by_distance.append((dist_squared, unit))
            
    minions = [x[1] for x in sorted(minions_by_distance, key=lambda x: x[0])[:k]]

    return minions

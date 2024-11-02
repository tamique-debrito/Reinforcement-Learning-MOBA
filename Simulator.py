from typing import List
import numpy as np
from enum import Enum

from CONSTANTS import *
from Units import *
from Units import calc_damage




@dataclass
class AggroEvent():
    eventType = EventType.AGGRO
    source: Minion

class Simulation:
    def __init__(self, track_aa_per_step=False):
        self.unitList: List[AnyUnit] = []
        self.sim_step = 0
        self.next_uid = 0
        self.events = {} # Map of sim_step to list of events
        self.uidToIndex = {}

        self.track_aa_per_step = track_aa_per_step
        self.aa_track = set()

    def add_player(self, stats, level_stats, x, y, team):
        index = len(self.unitList)
        player = Player(x, y, stats, level_stats, self.next_uid, index, team, True)
        self.next_uid += 1 
        self.unitList.append(player)
        self.uidToIndex[player.uid] = index
        return player

    def add_player_by_hook(self, hook):
        # Since there are many types of players, and the only init params we need from the Simulation are a uid and index
        index = len(self.unitList)
        player = hook(self.next_uid, index)
        self.next_uid += 1 
        self.unitList.append(player)
        self.uidToIndex[player.uid] = index
        return player
    
    def add_minion(self, stats, x, y, team):
        index = len(self.unitList)
        minion = Minion(x, y, stats, self.next_uid, index, team)
        self.next_uid += 1 
        self.unitList.append(minion)
        self.uidToIndex[minion.uid] = index
        
    
    def add_structure(self, stats, x, y, team):
        index = len(self.unitList)
        turret = Turret(x, y, stats, self.next_uid, index, team)
        self.next_uid += 1 
        self.unitList.append(turret)
        self.uidToIndex[turret.uid] = index

    def get_event_list_at_step(self, sim_step):
        if sim_step not in self.events:
            self.events[sim_step] = []
        return self.events[sim_step]

    def add_aa_event(self, damage, timeToHit, sourceUid, targetUid):
        sim_steps_to_hit = int(timeToHit/TIMESTEP)
        simStepWhenHit = int(timeToHit/TIMESTEP) + self.sim_step
        event_list = self.get_event_list_at_step(simStepWhenHit)
        event = AutoAttackEvent(damage, sourceUid, targetUid)
        event_list.append(event)
        if self.track_aa_per_step:
            self.aa_track.add((sourceUid, targetUid, sim_steps_to_hit))
        #print(f"added aa event: {sourceUid} -> {targetUid}")

    def add_callback_event_by_sim_step_delta(self, callback, sim_step_delta):
        event_list = self.get_event_list_at_step(self.sim_step + sim_step_delta)
        event = CallbackEvent(callback)
        event_list.append(event)

    
    def step(self):
        if self.track_aa_per_step:
            self.aa_track = set()
        for unit in self.unitList:
            if not unit.active: continue
            unit.step(self.unitList, self.add_aa_event)
        coords = np.array([[unit.x for unit in self.unitList], [unit.y for unit in self.unitList]])
        pairwise_displacements = np.expand_dims(coords, 1) - np.expand_dims(coords, 2) # [..., i, j] is the displacement vector from i to j
        pairwise_distances = np.linalg.norm(pairwise_displacements, axis=0)
        self.process_events(pairwise_distances)
        self.handle_collisions(pairwise_distances, pairwise_displacements)
        self.set_auto_targets(pairwise_distances)
        if self.process_deaths():
            self.rebuild_list()
        self.sim_step += 1

    def handle_collisions(self, pairwise_distances, pairwise_displacements):
        sizes = np.array([unit.stats.size for unit in self.unitList])
        collision_thresholds = np.expand_dims(sizes, 0) + np.expand_dims(sizes, 1)
        overlap = collision_thresholds - pairwise_distances
        collided = (overlap > 0) * (np.eye(len(pairwise_distances)) != 1)
        collided_indices = np.where(collided)
        for i, j in zip(collided_indices[0], collided_indices[1]):
            unit = self.unitList[i]
            if unit.state == UnitState.ATTACKING or unit.state == UnitState.STILL or not unit.active: continue
            other_unit = self.unitList[j]
            # if other_unit.state == MinionState.ATTACKING or other_unit.state == MinionState.STILL:
            #     factor = 1.02
            # else:
            #     factor = 0.501
            if pairwise_distances[i, j] > 0:
                dx, dy = pairwise_displacements[..., i, j] / pairwise_distances[i, j]
                if unit.last_dx * dx + unit.last_dy * dy > 0:
                    # trying to move into the collided unit - go around
                    if unit.last_dx * dy - unit.last_dy * dx > 0: #TODO: Make this use the tangent to the colliding unit rather than the perpendicular direction to the last displacement
                        # One clock direction
                        unit.x = unit.x - unit.last_dx + unit.last_dy
                        unit.y = unit.y - unit.last_dy - unit.last_dx
                    else:
                        # The other clock direction
                        unit.x = unit.x - unit.last_dx - unit.last_dy
                        unit.y = unit.y - unit.last_dy + unit.last_dx
                # otherwise, we are moving away and don't need to handle the collision
            else:
                unit.x += (sizes[i] + sizes[j]) * np.random.randint(-1, 1) # If units somehow end up exactly on top of each other, just randomly split them up

    def rebuild_list(self):
        # Rebuilds the unit list and uid->index mapping, removing any inactive units
        newList = []
        newUidToIndex = {}
        i = 0
        for unit in self.unitList:
            if unit.active:
                unit.index = i
                newUidToIndex[unit.uid] = i
                i += 1
                newList.append(unit)
        self.unitList = newList
        self.uidToIndex = newUidToIndex
    
    def process_deaths(self):
        anyDeaths = False
        for unit in self.unitList:
            if unit.stats.health <= 0:
                anyDeaths = True
                unit.active = False
                lastDamagingUnit = self.get_unit_by_uid_safe(unit.last_damaging_uid)
                if lastDamagingUnit is None: continue # TODO: update to account for player last damage on other players
                if lastDamagingUnit.unitType == UnitType.PLAYER:
                    lastDamagingUnit.add_gold_and_xp(unit.stats.worth_gold, unit.stats.worth_xp)
                elif unit.unitType == UnitType.TURRET:
                    player = self.get_player_by_team(unit.team)
                    if player is not None:
                        player.add_gold_and_xp(unit.stats.worth_gold, unit.stats.worth_xp)
        return anyDeaths
    
    def get_player_by_team(self, enemy_team) -> Optional[Player]:
        # Get first player on opposite team of the one specified
        for unit in self.unitList:
            if unit.unitType == UnitType.PLAYER and unit.team != enemy_team:
                return unit #type: ignore

    def get_unit_by_uid_safe(self, uid):
        if uid not in self.uidToIndex: return None
        return self.unitList[self.uidToIndex[uid]]
    
    def get_unit_by_uid(self, uid):
        return self.unitList[self.uidToIndex[uid]]

    def process_events(self, pairwise_distances):
        if self.sim_step not in self.events:
            return []
        events = self.events[self.sim_step] # Get current step events
        aggroEvents = []
        for event in events:
            if event.eventType == EventType.AUTOATTACK:
                if event.targetUid not in self.uidToIndex: continue # Target can die before the event is processed
                target_index = self.uidToIndex[event.targetUid]
                target = self.unitList[target_index]
                if target.can_take_damage():
                    dmg = calc_damage(target.stats, event.damage)
                    target.stats.health -= dmg
                    if dmg > 0 and target.unitType == UnitType.PLAYER:
                        target.interrupt_channel()
                    target.last_damaging_uid = event.sourceUid
                    if event.sourceUid not in self.uidToIndex: continue # Source can die before the event is processed
                    source_index = self.uidToIndex[event.sourceUid]
                    source = self.unitList[source_index]
                    if source.unitType == UnitType.PLAYER:
                        aggroEvents.append(AggroEvent(source))
            elif event.eventType == EventType.CALLBACK:
                event.callback()
        del self.events[self.sim_step]
        
        for aggroEvent in aggroEvents:
            pairwise_distances_row = pairwise_distances[aggroEvent.source.index]
            for i in range(len(pairwise_distances_row)):
                if i != aggroEvent.source.index:
                    if pairwise_distances_row[i] < self.unitList[i].aggroRange and self.unitList[i].unitType != UnitType.PLAYER:
                        self.unitList[i].set_aggro_target(aggroEvent.source)
    
    def set_auto_targets(self, pairwise_distances):
        for unit in self.unitList:
            if (unit.unitType == UnitType.MINION or unit.unitType == UnitType.TURRET) and (unit.state == UnitState.MOVING or unit.state == UnitState.STILL):
                other_distances = pairwise_distances[unit.index]
                for i in range(len(other_distances)):
                    if i != unit.index and other_distances[i] - self.unitList[i].stats.size < unit.aggroRange:
                        other = self.unitList[i]
                        unit.set_aggro_target(other)

    def get_closest_enemy_from_point(self, x, y, team):
        return get_closest_enemy_from_point(self.unitList, x, y, team)
    
    
    def get_vec_k_minions(self, x, y, k):
        return get_vec_k_minions(self.unitList, x, y, k)




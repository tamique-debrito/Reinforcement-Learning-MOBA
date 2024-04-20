from dataclasses import dataclass


@dataclass
class BaseStats:
    max_health: float
    health: float
    armor: float
    speed: float
    size: float
    attack_damage: float
    attack_speed: float
    attack_range: float
    worth_gold: int
    worth_xp: int

@dataclass
class LevelStats:
    max_health: float
    armor: float
    speed: float
    attack_damage: float
    attack_speed: float
    def apply(self, stats):
        stats.max_health += self.max_health
        stats.health += self.max_health
        stats.armor += self.armor
        stats.speed += self.speed
        stats.attack_damage += self.attack_damage
        stats.attack_speed += self.attack_speed
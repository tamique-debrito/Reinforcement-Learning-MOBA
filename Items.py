from dataclasses import dataclass
from enum import Enum
from Stats import BaseStats

@dataclass
class ItemStats:
    cost: int
    max_health: float = 0.0
    armor: float = 0.0
    speed: float = 0.0
    size: float = 0.0
    attack_damage: float = 0.0
    attack_speed: float = 0.0
    attack_range: float = 0.0

    def apply_to_stats(self, stats: BaseStats):
        stats.max_health += self.max_health
        stats.armor += self.armor
        stats.speed += self.speed
        stats.attack_damage += self.attack_damage
        stats.attack_speed += self.attack_speed

class Item(Enum):
    SWORD = 1
    SHIELD = 2
    BOOTS = 3

ITEMS = {
    Item.SWORD: ItemStats(100, attack_damage=50, attack_speed=0.1),
    Item.SHIELD: ItemStats(100, max_health=150, armor=10),
    Item.BOOTS: ItemStats(100, speed=1.0)
}

from dataclasses import dataclass


@dataclass
class Entity:
    uid: int
    x: float
    y: float
    size: float
    speed: float

class VectorLibWrapper:
    def __init__(self):
        self.entities = []

    def register_entity(self, uid, x, y, size, speed):
        self.entities.append(Entity(uid, x, y, size, speed))
        
    def get_pairwise_distances(self):
        pass
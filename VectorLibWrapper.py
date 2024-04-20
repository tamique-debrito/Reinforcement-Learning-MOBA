from dataclasses import dataclass


@dataclass
class Entity:
    x: float
    y: float
    size: float
    speed: float

class VectorLibWrapper:
    def __init__(self):
        self.x = []
        self.y = []

    def register_entity(self, uid, x, y, size, speed):
        self.x
        
    def get_pairwise_distances(self):
        pass
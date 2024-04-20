import pygame

def transform_coords(x, y, s=0.0, shiftX=0, shiftY=0, scale=1.0):
   return x * scale + shiftX, y * scale + shiftY, s * scale


class BaseDisplayElement:
   def __init__(self,sim_steps_of_lifetime) -> None:
      self.total_steps = sim_steps_of_lifetime
      self.current_step = 0
   
   def step_only(self):
      self.current_step += 1
      return self.current_step >= self.total_steps
   
   def step_and_render(self, screen, shiftX, shiftY, scale):
      return True # Expire immediately
   
   def get_coord_and_delta(self):
      return 0.0, 0.0, 0.0, 0.0


class DisplayElementProjectileSourceOnly(BaseDisplayElement):
   def __init__(self, source, size, color, sim_steps_of_lifetime) -> None:
      self.source = source
      self.size = size
      self.color = color
      self.total_steps = sim_steps_of_lifetime
      self.current_step = 0

   def step_and_render(self, screen, shiftX, shiftY, scale):
      # If this returns true, the display element is expired
      projX, projY, projSize = transform_coords(self.source.x, self.source.y, self.size, shiftX, shiftY, scale)

      self.current_step += 1
      if self.current_step >= self.total_steps:
         return True

      pygame.draw.circle(screen, self.color, (projX, projY), projSize)
   
      return False

class DisplayElementProjectileLerpSourceTarget(BaseDisplayElement):
   def __init__(self, source, target, size, color, sim_steps_of_lifetime) -> None:
      self.source = source
      self.target = target
      self.size = size
      self.color = color
      self.total_steps = sim_steps_of_lifetime
      self.current_step = 0
   
   def get_coord_and_delta(self):
      dx = self.target.x - self.source.x
      dy = self.target.y - self.source.y
      lerp_factor = self.current_step / self.total_steps
      x = self.source.x + lerp_factor * dx
      y = self.source.y + lerp_factor * dy
      return x, y, dx, dy

   def step_and_render(self, screen, shiftX, shiftY, scale):
      # If this returns true, the display element is expired
      x, y, _, _ = self.get_coord_and_delta()
      projX, projY, projSize = transform_coords(x, y, self.size, shiftX, shiftY, scale)

      pygame.draw.circle(screen, self.color, (projX, projY), projSize)

      self.current_step += 1
      if self.current_step >= self.total_steps:
         return True
      return False
from typing import Dict, List
import pygame
from DisplayElements import *
from Items import Item
from Simulator import *
from Units import UnitType


MINION_AA_SIZE = 1.0
MINION_AA_COLOR = (150, 100, 100)

player_color = (0,255,00)
color_team_A = (255,0,0)
color_team_B = (0,0,255)
background_colour = (255,255,255)
moving_unit_color = (50,100,50)
chasing_unit_color = (100,100,0)
attacking_unit_windup_color = (155,25,25)
attacking_unit_cooldown_color = (100,15,15)
still_unit_color = (50,50,50)
chanelling_unit_color = (100,100,255)
error_unit_color = (255,0,255)

aggro_range_color = chasing_unit_color
attack_range_color = attacking_unit_windup_color

class Display:
   def __init__(self, width, height, show_aggro_range = False, show_attack_range = False, display_elem_tracking_only = False, shiftX = 0, shiftY = 0, scale = 1.0) -> None:
      self.display_elem_tracking_only = display_elem_tracking_only
      if not display_elem_tracking_only: self.screen = self.setUpScreen(width, height)
      self.display_elements: Dict[int, BaseDisplayElement] = {}
      self.next_display_elem_uid = 0

      self.show_aggro_range = show_aggro_range
      self.show_attack_range = show_attack_range

      # Game -> screen coordinate mapping
      self.shiftX = shiftX
      self.shiftY = shiftY
      self.scale = scale

   def setUpScreen(self, width, height):
      pygame.init()
      screen = pygame.display.set_mode((width, height))
      pygame.display.set_caption('Game')
      screen.fill(background_colour)
      pygame.display.flip()
      return screen
   
   def put_circle(self, x, y, r, color, width=0):
      x, y, r = transform_coords(x, y, r, self.shiftX, self.shiftY, self.scale)
      pygame.draw.circle(self.screen, color, (x, y), float(r), width=width)
   
   def put_X(self, x, y, r, color, width=2):
      x, y, r = transform_coords(x, y, r, self.shiftX, self.shiftY, self.scale)
      pygame.draw.line(self.screen, color, (x - r, y - r), (x + r, y + r), width=width)
      pygame.draw.line(self.screen, color, (x - r, y + r), (x + r, y - r), width=width)

   def put_line(self, x1, y1, x2, y2, color, width=1):
      x1, y1, _ = transform_coords(x1, y1, 0, self.shiftX, self.shiftY, self.scale)
      x2, y2, _ = transform_coords(x2, y2, 0, self.shiftX, self.shiftY, self.scale)
      pygame.draw.line(self.screen, color, (x1, y1), (x2, y2), width=width)

   def renderState(self, units: List[Minion], delay=100, skip_render=False):
      if self.display_elem_tracking_only or skip_render:
         self.process_display_elements(self.shiftX, self.shiftY, self.scale, skip_render=skip_render)
         return
      self.screen.fill(background_colour)
      for unit in units:
         entX, entY, entSize = transform_coords(unit.x, unit.y, unit.stats.size, self.shiftX, self.shiftY, self.scale)
         if unit.unitType == UnitType.PLAYER:
            if unit.state == UnitState.CHANNELING: color_to_use=chanelling_unit_color
            else: color_to_use = player_color
         elif unit.state == UnitState.MOVING:
            color_to_use = moving_unit_color
         elif unit.state == UnitState.CHASING:
            color_to_use = chasing_unit_color
         elif unit.state == UnitState.ATTACKING:
            if unit.aa_state == AAState.WINDUP: color_to_use = attacking_unit_windup_color
            else: color_to_use = attacking_unit_cooldown_color
         elif unit.state == UnitState.STILL:
            color_to_use = still_unit_color
         else:
            color_to_use = error_unit_color
         pygame.draw.circle(self.screen, color_to_use, (entX, entY), float(entSize))
         team_color_to_use = color_team_A if unit.team == TEAM_A else color_team_B
         pygame.draw.circle(self.screen, team_color_to_use, (entX, entY), float(entSize * 0.5))
         if self.show_aggro_range: pygame.draw.circle(self.screen, aggro_range_color, (entX, entY), float(unit.aggroRange * self.scale), width=1)
         if self.show_attack_range: pygame.draw.circle(self.screen, attack_range_color, (entX, entY), float(unit.stats.attack_range * self.scale), width=1)
         pygame.draw.rect(self.screen,(0, 255, 0), pygame.rect.Rect(entX - entSize, entY + entSize + 3, unit.stats.health / unit.stats.max_health * 2 * entSize, 5.0))

      self.process_display_elements(self.shiftX, self.shiftY, self.scale)

      pygame.display.flip()
      pygame.time.delay(delay)

   def addDisplayElement(self, elem):
      assert elem is not None
      self.display_elements[self.next_display_elem_uid] = elem
      self.next_display_elem_uid += 1
   
   def process_display_elements(self, shiftX, shiftY, scale, skip_render=False):
      indices_to_remove = set()
      for i in self.display_elements:
         elem = self.display_elements[i]
         if self.display_elem_tracking_only or skip_render:
            remove = elem.step_only()
         else:
            remove = elem.step_and_render(self.screen, shiftX, shiftY, scale)
         if remove:
            indices_to_remove.add(i)
      for i in indices_to_remove:
         del self.display_elements[i]



class InputCommandType(Enum):
   MOVE = 1
   ATTACK = 2
   BUY = 3
   RECALL = 4
   FLASH_CAST = 5
   Q_CAST = 6

@dataclass
class InputCommand:
   commandType: InputCommandType
   x: Optional[float] = None
   y: Optional[float] = None
   item: Optional[Item] = None

   def __str__(self) -> str:
      extra_info = ""
      if self.commandType in [InputCommandType.MOVE, InputCommandType.ATTACK, InputCommandType.FLASH_CAST, InputCommandType.Q_CAST]:
         extra_info = f"x={self.x}, y={self.y}"
      elif self.commandType == InputCommandType.BUY:
         extra_info = f"item={self.item}"
      return str(self.commandType).split(".")[1].split(":")[0] + " " + extra_info

class KeyEvent(Enum):
   ATTACK = 1
   RECALL = 2
   FLASH_CAST = 3
   Q_CAST = 4

class InputTracker:
   def __init__(self, shiftX, shiftY, scale) -> None:
      self.next_click_is_attack = False
      self.next_command_for_alt = False

      self.shiftX = shiftX
      self.shiftY = shiftY
      self.scale = scale

   def get_game_mouse_coords(self):
      # Gets unscaled/unshifted mouse coords (i.e. in game coordinates)
      x, y = pygame.mouse.get_pos()
      x = (x - self.shiftX) / self.scale
      y = (y - self.shiftY) / self.scale
      return x, y
   
   def clear_events(self):
      pygame.event.get()
      return
   
   def step(self) -> Optional[InputCommand]:
      key_event = None
      click_event = None
      for event in pygame.event.get():
         if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_a:
               key_event = KeyEvent.ATTACK
            elif event.key == pygame.K_x:
               key_event = KeyEvent.RECALL
            elif event.key == pygame.K_QUOTE:
               key_event = KeyEvent.Q_CAST
            elif event.key == pygame.K_LALT:
               self.next_command_for_alt = True # Needs to be reset by consumer when action is processed
         if event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1: # Left click, apparently no constant available
               click_event = "L"
            elif event.button == 3: # Right click, apparently no constant available
               click_event = "R"
      if key_event is not None or click_event is not None:
         x, y = self.get_game_mouse_coords()
      if key_event == KeyEvent.ATTACK:
         self.next_click_is_attack = True
      elif key_event == KeyEvent.RECALL:
         return InputCommand(commandType=InputCommandType.RECALL)
      elif key_event == KeyEvent.Q_CAST:
         return InputCommand(commandType=InputCommandType.Q_CAST, x=x, y=y)
      
      if click_event is not None:
         if click_event == "R":
            return InputCommand(InputCommandType.MOVE, x=x, y=y)
         elif click_event == "L" and self.next_click_is_attack:
            return InputCommand(InputCommandType.ATTACK, x=x, y=y)
         self.next_click_is_attack = False


      
      return None


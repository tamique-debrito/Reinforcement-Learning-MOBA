from DisplayElements import DisplayElementProjectileLerpSourceTarget
from SimpleMage import SimpleMage
from Simulator import *
from Stats import BaseStats
from UIForTest import *

VISUAL_PADDING = 50
SCALE = 4.0

STRUCTURE_STATS = lambda: BaseStats(200, 200, 50, 0, TURRET_SIZE, 50, 2.0, TURRET_AGGRO_DISTANCE, 300, 300)
MELEE_MINION_STATS = lambda: BaseStats(100, 100, 0, 8.0, MINION_SIZE, 5, 1.0, MINION_SIZE * 2, 20, 20)
RANGED_MINION_STATS = lambda: BaseStats(70, 70, 0, 8.0, MINION_SIZE * 0.8, 15, 1.0, MINION_AGGRO_DISTANCE, 20, 20)

STRONG_MINION_STATS = lambda: BaseStats(1000, 1000, 0, 8.0, MINION_SIZE * 1.3, 100, 1.0, MINION_AGGRO_DISTANCE * 0.7, 20, 20)

PLAYER_STATS = lambda: BaseStats(500, 500, 100, 10.0, PLAYER_SIZE, 50, 1.0, MINION_AGGRO_DISTANCE * 1.0, 200, 100)
PLAYER_LEVEL_STATS = lambda: LevelStats(100, 10, 0.1, 10, 0.1)

class Game:
    def __init__(self, team_to_play=TEAM_B, display_elem_tracking_only=False) -> None:
        sim: Simulation = Simulation(track_aa_per_step=True) # TODO: probably pass use_display as the parameter for whether to track aa
        self.sim = sim
        self.inputTracker = InputTracker(VISUAL_PADDING, VISUAL_PADDING, SCALE)
        quarter_mark = MAP_Y_DIM / 4.0
        sim.add_structure(STRUCTURE_STATS(), TURRET_LINE, quarter_mark * 0, TEAM_A)
        sim.add_structure(STRUCTURE_STATS(), TURRET_LINE, quarter_mark * 1, TEAM_A)
        sim.add_structure(STRUCTURE_STATS(), TURRET_LINE, quarter_mark * 3, TEAM_B)
        sim.add_structure(STRUCTURE_STATS(), TURRET_LINE, quarter_mark * 4, TEAM_B)
        
        self.nexus_A = sim.unitList[0]
        self.nexus_B = sim.unitList[3]

        self.use_input_tracking = False

        #self.player: Player = sim.add_player(PLAYER_STATS(), PLAYER_LEVEL_STATS(), PLAYER_LINE, quarter_mark * 4, TEAM_B)
        team_a_player: Player = sim.add_player_by_hook(lambda uid, index: SimpleMage(PLAYER_LINE, 0, PLAYER_STATS(), PLAYER_LEVEL_STATS(), uid, index, TEAM_A, True, sim.add_callback_event_by_sim_step_delta))
        team_b_player: Player = sim.add_player_by_hook(lambda uid, index: SimpleMage(PLAYER_LINE, quarter_mark * 4, PLAYER_STATS(), PLAYER_LEVEL_STATS(), uid, index, TEAM_B, True, sim.add_callback_event_by_sim_step_delta))

        if team_to_play == TEAM_A:
            self.player = team_a_player
            self.alt_player = team_b_player
        else:
            self.player = team_b_player
            self.alt_player = team_a_player
        
        self.team_to_play = team_to_play

        self.display = Display(SCALE * MAP_X_DIM + 2 * VISUAL_PADDING, SCALE * MAP_Y_DIM + 2 * VISUAL_PADDING, display_elem_tracking_only=display_elem_tracking_only) # TODO: add a "tracking only" property so that the screen doesn't get created
    
    def step(self, command=None, alt_command=None, skip_render=False, delay=100):
        # Player inputs
        if self.use_input_tracking:
            command = self.inputTracker.step()
            if command is not None and self.inputTracker.next_command_for_alt:
                alt_command = command
                command = None
                self.inputTracker.next_command_for_alt = False
        elif not skip_render and not self.display.display_elem_tracking_only:
            self.inputTracker.clear_events()
        self.perform_command(command, True)
        self.perform_command(alt_command, False)

        # Simulation and rendering
        self.generate_minion_stream()
        self.sim.step()
        for sourceUid, targetUid, sim_steps in self.sim.aa_track:
            source = self.sim.get_unit_by_uid_safe(sourceUid)
            target = self.sim.get_unit_by_uid_safe(targetUid)
            if source is not None and target is not None:
                self.display.addDisplayElement(DisplayElementProjectileLerpSourceTarget(source, target, MINION_AA_SIZE, MINION_AA_COLOR, sim_steps))
        self.renderState(delay=delay, skip_render=skip_render)

    def perform_command(self, command, main_player):
        if main_player:
            player = self.player
        else:
            player = self.alt_player
        if command is not None:
            if command.commandType == InputCommandType.MOVE:
                player.set_move_target(command.x, command.y)
            elif command.commandType == InputCommandType.ATTACK:
                target = self.sim.get_closest_enemy_from_point(command.x, command.y, player.team)
                if target is not None:
                    player.set_attack_target(target)
            elif command.commandType == InputCommandType.RECALL:
                player.start_channel(ChannelReason.RECALL, RECALL_TIMER)
            elif command.commandType == InputCommandType.Q_CAST:
                display_elem = player.cast_Q(self.sim.unitList, command.x, command.y)
                if display_elem is not None: self.display.addDisplayElement(display_elem)

    def renderState(self, delay=100, skip_render=False):
        self.display.renderState(self.sim.unitList, VISUAL_PADDING, VISUAL_PADDING, scale=SCALE, delay=delay, skip_render=skip_render)

    def generate_minion_stream(self):
        sim_step = self.sim.sim_step
        #if sim_step > 100: return
        if sim_step % 200 <= 50 and sim_step % 10 == 0:
            if sim_step % 100 // 10 < 3:
                self.sim.add_minion(MELEE_MINION_STATS(), MINION_LINE, 0, TEAM_A)
                self.sim.add_minion(MELEE_MINION_STATS(), MINION_LINE, MAP_Y_DIM, TEAM_B)
            else:
                self.sim.add_minion(RANGED_MINION_STATS(), MINION_LINE, 0, TEAM_A)
                self.sim.add_minion(RANGED_MINION_STATS(), MINION_LINE, MAP_Y_DIM, TEAM_B)

def test():
    game = Game()
    for i in range(1000):
        game.step()

if __name__ == "__main__":
    test()
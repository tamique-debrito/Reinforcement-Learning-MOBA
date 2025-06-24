This project explores MARL algorithms in a simplified [MOBA](https://en.wikipedia.org/wiki/Multiplayer_online_battle_arena) environment. The goal is to train AI Agents to play the MOBA game successfully. So far, the agents have learned some basic useful behaviors.

I used Stable Baselines to allow focusing on the multi-agent training/tracking/management framework, experimentation with representations, and reward shaping. Also, I implemented some efficiency enhancements (multiprocessing to train multiple agents at the same time considering the low GPU utilization from small models, cuda and cython for game engine), but those didn't end up being practically useful at the scale of this project. They're not included in this repository.

When I last worked on this, the agents had not yet learned optimal behaviors, but did occasionally demonstrate basic strategies. For example, what's called "kiting" in MOBAs (movement in between attacks to minimize vulnerability. Mostly forward/backward motion) and retreating when low on health.

This is what the environment looks like:

![image](https://github.com/user-attachments/assets/5d017e43-681d-40bc-a05c-e71e428c1d44)

Each circle is a unit. The color of the center of the circle signifies which team it's on. The size of the circle and the color of the outer layer signifies the unit type.

Biggest circles with grey outer layer = Turrets. They launch powerful projectiles and give a large reward if destroyed

Smallest circles with green/red outer layer = Minions. Weak units that give a small reward.

Medium circles with green outer layer = Players. These are the agent-controllable units. Nearly as powerful as a turret and give a large reward to opponent if destroyed.

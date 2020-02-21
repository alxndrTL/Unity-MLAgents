The gym_wrapper UnityEnvModified() allows one to interact with an Unity environment (built with ML-Agents) in a gym-like manner.  
Only 2 key functions are necessary to interact with the env:
* reset(): reset every agents in the environment, and sends back their first observation.
* step(action): give the agents their action to execute, and step forward in the environment until the agents request another action. Returns observations, rewards, dones for all agents.

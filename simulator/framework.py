
from simulator import simulator
from pricing_agent import price_agent       # uppeer level pricing agent is a directory
from matching_agent import matching_agent   # upper level matching agent is a directory
from reposition_agent import reposition_agent # upper level reposition agent is a directory

class Simulator:
    def __init__(self):
        pass

    def pricing(self):
        pass

    def matching(self):
        pass

    def reposition(self):
        pass



class SimulatorTrainer:
    def __init__(self, simulator):
        self.simulator = simulator

    def train(self):
        price_agent = price_agent()
        matching_agent = matching_agent()
        reposition_agent = reposition_agent()
        # obervation for pricing agent
        pricing_state = self.simulator.get_pricing_state()
        # generation action for pricing agent
        pricing_action = price_agent.get_action(pricing_state)
        # execute action
        self.simulator.update_pricing_agent(pricing_action)
        
        
        
        # observation for matching agent
        matching_state = self.simulator.get_matching_state()
        # generation action for matching agent
        matching_action = matching_agent.get_action(matching_state)
        # execute action
        self.simulator.update_matching_agent(matching_action)
        # get reward for pricing agent
        pricing_reward = self.simulator.get_pricing_reward()
        # get reward for matching agent
        matching_reward = self.simulator.get_matching_reward()
        # update matching agent with reward
        # update pricing agent with reward
        price_agent.update([pricing_state,pricing_action,pricing_reward])
        matching_agent.update([matching_state,matching_action,matching_reward])   


        # observation for reposition agent
        reposition_state = self.simulator.get_reposition_state()
        # generation action for reposition agent
        reposition_action = reposition_agent.get_action(reposition_state)
        # execute action
        self.simulator.update_reposition_agent(reposition_action)
        # get reward for reposition agent
        reposition_reward = self.simulator.get_reposition_reward()
        # update reposition agent with reward
        reposition_agent.update([reposition_state,reposition_action,reposition_reward])



    def test(self):
        self.simulator.test()

    def render(self):
        self.simulator.render()

    def pricing(self):
        self.simulator.pricing()

    def matching(self):
        self.simulator.matching()

    def reposition(self):
        self.simulator.reposition()




simulator = simulator.Simulator()

simulator_trainer = SimulatorTrainer(simulator)





for episode in range(100):
    simulator_trainer.render()
    for epoch in range(100):
        simulator_trainer.train()
        if epoch % 10 == 0:
            print("epoch: ", epoch)
            # validation
            simulator_trainer.test()
# pricing

# matching

# reposition
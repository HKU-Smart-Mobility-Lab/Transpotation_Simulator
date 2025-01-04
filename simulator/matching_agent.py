from sarsa import SarsaAgent
class MatchingAgent:
    def __init__(self, strategy_type, strategy_params, load_path=None, flag_load=False):
        """
        Initialize the MatchingAgent with a specific strategy type.
        :param strategy_type: The strategy type, e.g., 'sarsa', 'sarsa_no_subway', etc.
        :param strategy_params: Parameters for the chosen strategy.
        :param load_path: Path to load pre-trained strategy parameters (optional).
        :param flag_load: Boolean indicating whether to load parameters from the specified path.
        """
        self.strategy = None
        self._initialize_strategy(strategy_type, strategy_params, load_path, flag_load)

    def _initialize_strategy(self, strategy_type, strategy_params, load_path, flag_load):
        """
        Dynamically initialize the strategy based on the type.
        """
        if strategy_type in ['sarsa', 'sarsa_no_subway', 'sarsa_travel_time',
                             'sarsa_travel_time_no_subway', 'sarsa_total_travel_time',
                             'sarsa_total_travel_time_no_subway']:
            self.strategy = SarsaAgent(**strategy_params)
            if flag_load and load_path:
                print(f"Loading parameters for strategy: {strategy_type} from {load_path}")
                self.strategy.load_parameters(load_path)
        else:
            raise ValueError(f"Unsupported strategy type: {strategy_type}")

    def get_action(self, state):
        """
        Get the action based on the current state.
        :param state: The current state of the environment.
        :return: The selected action.
        """
        if self.strategy:
            return self.strategy.get_action(state)
        raise RuntimeError("No strategy initialized in MatchingAgent")

    def update(self, transitions):
        """
        Update the agent's strategy based on the feedback from the environment.
        :param transitions: Feedback data for updating the strategy.
        """
        if self.strategy:
            self.strategy.perceive(transitions)
        else:
            raise RuntimeError("No strategy initialized in MatchingAgent")

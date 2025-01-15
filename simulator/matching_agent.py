from sarsa import SarsaAgent
from utilities import *

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

    def get_action(self, matching_state, epsilon=0):
        """
        Generate matching actions based on the current state.
        :param matching_state: Dictionary containing the state information (e.g., requests, drivers, distances).
        :param epsilon: Exploration rate for RL-based decision-making.
        :return: Matched order-driver pairs.
        """
        wait_requests = matching_state['wait_requests']
        driver_table = matching_state['driver_table']
        maximal_pickup_distance = matching_state['maximal_pickup_distance']
        dispatch_method = matching_state['dispatch_method']
        method = matching_state['method']

        con_ready_to_dispatch = (driver_table['status'] == 0) | (driver_table['status'] == 4)
        idle_driver_table = driver_table[con_ready_to_dispatch]
        num_wait_request = wait_requests.shape[0]
        num_idle_driver = idle_driver_table.shape[0]
        matched_pair_actual_indexs = []

        # If no requests or no idle drivers, return empty actions
        if num_wait_request == 0 or num_idle_driver == 0:
            return matched_pair_actual_indexs  # Return an empty list

        # Step 2: Perform matching logic
        if dispatch_method == 'LD':
            # Generate order-driver pairs
            request_array_temp = wait_requests.loc[:, ['origin_lng', 'origin_lat', 'order_id', 'weight']]
            request_array = np.repeat(request_array_temp.values, num_idle_driver, axis=0)
            driver_loc_array_temp = idle_driver_table.loc[:, ['lng', 'lat', 'driver_id']]
            driver_loc_array = np.tile(driver_loc_array_temp.values, (num_wait_request, 1))
            dis_array = distance_array(request_array[:, :2], driver_loc_array[:, :2])

            if method == "pickup_distance":
                # Update weights for distance
                request_array[:, -1] = maximal_pickup_distance - dis_array + 1

            flag = np.where(dis_array <= maximal_pickup_distance)[0]
            if len(flag) > 0:
                order_driver_pair = np.vstack(
                    [request_array[flag, 2], driver_loc_array[flag, 2], request_array[flag, 3], dis_array[flag]]).T
                matched_pair_actual_indexs = LD(order_driver_pair.tolist())

        return matched_pair_actual_indexs


    def update(self, transitions):
        """
        Update the agent's strategy based on the feedback from the environment.
        :param transitions: Feedback data for updating the strategy.
        """
        if self.strategy:
            self.strategy.perceive(transitions)
        else:
            raise RuntimeError("No strategy initialized in MatchingAgent")

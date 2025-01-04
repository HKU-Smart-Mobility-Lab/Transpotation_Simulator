# 导入核心类
from simulator_env import Simulator
from pricing_agent import PricingAgent
from matching_agent import MatchingAgent
from reposition_agent import RepositionAgent

# 导入工具库
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import time
import pickle
import os


# SimulatorTrainer: Andrew
class SimulatorTrainer:
    def __init__(self, simulator: Simulator, pricing_agent: PricingAgent, matching_agent: MatchingAgent):
        self.simulator = simulator
        self.pricing_agent = pricing_agent
        self.matching_agent = matching_agent


    def log_epoch_metrics(self, epoch, duration, simulator: Simulator):
        """
        Log the metrics for a given epoch.
        :param epoch: Current epoch number.
        :param duration: Duration of the epoch.
        :param simulator: Simulator instance with metrics to log.
        """
        print(f"Epoch: {epoch}")
        print(f"Epoch running time: {duration:.2f}s")
        print(f"Total reward: {simulator.total_reward}")
        print(f"Total orders: {simulator.total_request_num}")
        print(f"Matched orders: {simulator.matched_requests_num}")
        print(f"Step details:")
        print(f"  Step 1 - Order dispatching: {simulator.time_step1}")
        print(f"  Step 2 - Reaction: {simulator.time_step2}")
        print(f"  Step 3 - Bootstrap new orders: {simulator.step3}")
        print(f"  Step 4 - Cruise: {simulator.step4}")
        print(f"  Step 4.1 - Track recording: {simulator.step4_1}")
        print(f"  Step 5 - Update state: {simulator.step5}")
        print(f"  Step 6 - Offline update: {simulator.step6}")
        print(f"  Step 7 - Update time: {simulator.step7}")

    def run_training_epoch(self, simulator: Simulator, agent: MatchingAgent, epoch, epsilon, train_config):
        """
        Run a single training epoch.
        :param simulator: Simulator instance.
        :param agent: MatchingAgent or other RL agent.
        :param epoch: Current epoch number.
        :param epsilon: Exploration rate for this epoch.
        :param train_config: Training configuration dictionary.
        :return: Dictionary containing metrics for this epoch.
        """
        # Initialize metrics
        metrics = {
            'total_reward': 0,
            'epoch_duration': 0
        }

        # Set up simulator for this epoch
        simulator.experiment_date = train_config['train_dates'][epoch % len(train_config['train_dates'])]
        simulator.reset()

        # Run the simulation
        start_time = time.time()
        for step in range(simulator.finish_run_step):
            dispatch_transitions = simulator.rl_step(agent, epsilon)
            if agent is not None:
                agent.update(dispatch_transitions)
        end_time = time.time()

        # Collect metrics
        metrics['total_reward'] = simulator.total_reward
        metrics['epoch_duration'] = end_time - start_time

        # Log metrics
        self.log_epoch_metrics(epoch, metrics['epoch_duration'], simulator)
        return metrics

    def save_training_results(self, simulator_record, total_reward_record, output_path, epoch, save_interval=200):
        """
        Save training results to specified output paths.
        :param simulator_record: Simulator record to be saved.
        :param total_reward_record: Array of total rewards for each epoch.
        :param output_path: Directory where results will be saved.
        :param epoch: Current epoch number.
        :param save_interval: Frequency of saving results (e.g., every `save_interval` epochs).
        """
        # Save simulator record
        record_file = f"output/order_record_refactor.pickle"
        with open(record_file, "wb") as f:
            pickle.dump(simulator_record, f)
        print(f"Simulator record saved to {record_file}")

        # Save total reward record periodically
        if epoch % save_interval == 0:
            reward_file = f"{output_path}/training_results_record.pickle"
            with open(reward_file, "wb") as f:
                pickle.dump(total_reward_record, f)
            print(f"Training reward record saved to {reward_file}")


    
    def train(self, simulator: Simulator, train_config):
        """
        Full training logic for the simulator.
        :param simulator: Simulator instance.
        :param train_config: Training configuration (e.g., number of epochs, save intervals).
        """
        total_reward_record = np.zeros(train_config['num_epochs'])
        epsilons = train_config['epsilons']
        agent = self.matching_agent
    
        for epoch in range(train_config['num_epochs']):
            # Run a single training epoch
            metrics = self.run_training_epoch(simulator, agent, epoch, epsilons[epoch], train_config)

            # Record total reward
            total_reward_record[epoch] = metrics['total_reward']

            # Save results periodically
            self.save_training_results(simulator.record, total_reward_record, train_config['output_path'], epoch,
                                       save_interval=train_config['save_interval'])




    def accumulate_metrics(self, simulator: Simulator, metrics):
        """
        Accumulate metrics for one test run.
        :param simulator: Simulator instance after one test run.
        :param metrics: Dictionary to store cumulative metrics.
        """
        metrics['total_reward'] += simulator.total_reward
        metrics['total_request_num'] += simulator.total_request_num
        metrics['transfer_request_num'] += simulator.transfer_request_num
        metrics['occupancy_rate'] += simulator.occupancy_rate
        metrics['matched_request_num'] += simulator.matched_requests_num
        metrics['long_request_num'] += simulator.long_requests_num
        metrics['medium_request_num'] += simulator.medium_requests_num
        metrics['short_request_num'] += simulator.short_requests_num
        metrics['matched_long_request_num'] += simulator.matched_long_requests_num
        metrics['matched_medium_request_num'] += simulator.matched_medium_requests_num
        metrics['matched_short_request_num'] += simulator.matched_short_requests_num
        metrics['occupancy_rate_no_pickup'] += simulator.occupancy_rate_no_pickup
        metrics['pickup_time'] += simulator.pickup_time / simulator.matched_requests_num
        metrics['waiting_time'] += simulator.waiting_time / simulator.matched_requests_num

    def run_test_episode(self, simulator: Simulator, agent, dates):
        """
        Run a test episode over multiple dates.
        :param simulator: Simulator instance.
        :param agent: MatchingAgent or other RL agent.
        :param dates: List of test dates.
        :return: Accumulated metrics.
        """
        metrics = {
            'total_reward': 0, 'matched_transfer_request_num': 0, 'total_request_num': 0, 'transfer_request_num': 0,
            'occupancy_rate': 0, 'matched_request_num': 0,
            'long_request_num': 0, 'medium_request_num': 0, 'short_request_num': 0,
            'matched_long_request_num': 0, 'matched_medium_request_num': 0,
            'matched_short_request_num': 0, 'occupancy_rate_no_pickup': 0,
            'pickup_time': 0, 'waiting_time': 0
        }
        for date in dates:
            simulator.experiment_date = date
            simulator.reset()
            for step in range(simulator.finish_run_step):
                simulator.rl_step(agent)
            self.accumulate_metrics(simulator, metrics)
        return metrics

    def initialize_test_dataframe(self, test_num, column_list):
        """
        Initialize or load the test DataFrame.
        """
        df = pd.DataFrame(np.zeros([test_num, len(column_list)]), columns=column_list)
        remaining_index_array = np.where(df['total_reward'].values == 0)[0]
        last_stopping_index = remaining_index_array[0] if len(remaining_index_array) > 0 else 0
        return df, last_stopping_index

    def save_results(self, df, output_path, num, method=None):
        """
        Save test results to the specified path.
        :param df: DataFrame containing test results.
        :param output_path: Directory where results should be saved.
        :param num: Current test iteration number.
        :param method: The method string used for naming the file.
        """
        import os
        if not os.path.exists(output_path):
            os.makedirs(output_path)  # Create directory if it doesn't exist

        # Construct the file name dynamically
        if method:
            file_name = f"performance_record_test_{method}_{num}.pickle"
        else:
            file_name = f"performance_record_test_{num}.pickle"

        save_path = os.path.join(output_path, file_name)

        # Save the DataFrame
        with open(save_path, 'wb') as f:
            pickle.dump(df, f)
        print(f"Results saved to {save_path}")

    def check_convergence(self, df, current_num, interval, threshold):
        """
        Check if the testing process has converged.
        :param df: DataFrame containing test results.
        :param current_num: Current test iteration number.
        :param interval: Number of previous iterations to consider for convergence.
        :param threshold: Convergence threshold.
        :return: Tuple (converged, index) where converged is a boolean indicating if convergence occurred.
        """
        if current_num >= (interval - 1):
            profit_array = df.loc[(current_num - interval):current_num, 'total_reward'].values
            error = np.abs(np.max(profit_array) - np.min(profit_array))
            print('Error for convergence check: ', error)
            if error < threshold:
                print(f'Converged at index {current_num}')
                return True, current_num
        return False, None

    def save_and_calculate_ratios(self, df, output_path, num, method):
        """
        Calculate ratios and save results at the end of testing.
        :param df: DataFrame containing results.
        :param output_path: Path to save results.
        :param num: Current test iteration.
        :param method: Simulation method for naming files.
        """
        # Calculate ratios
        df.loc[:num, 'matched_transfer_request_ratio'] = (
            df.loc[:num, 'matched_transfer_request_num'].values /
            df.loc[:num, 'matched_request_num'].values
        )
        df.loc[:num, 'transfer_long_request_ratio'] = (
            df.loc[:num, 'transfer_request_num'].values /
            df.loc[:num, 'long_request_num'].values
        )
        df.loc[:num, 'matched_long_request_ratio'] = (
            df.loc[:num, 'matched_long_request_num'].values /
            df.loc[:num, 'long_request_num'].values
        )
        df.loc[:num, 'matched_medium_request_ratio'] = (
            df.loc[:num, 'matched_medium_request_num'].values /
            df.loc[:num, 'medium_request_num'].values
        )
        df.loc[:num, 'matched_short_request_ratio'] = (
            df.loc[:num, 'matched_short_request_num'].values /
            df.loc[:num, 'short_request_num'].values
        )
        df.loc[:num, 'matched_request_ratio'] = (
            df.loc[:num, 'matched_request_num'].values /
            df.loc[:num, 'total_request_num'].values
        )

        # Save results with calculated ratios
        file_path = f"{output_path}performance_record_test_{method}.pickle"
        pickle.dump(df, open(file_path, 'wb'))
        print(f"Final results saved to: {file_path}")
        print(df.columns)
        print(df.iloc[num, :])

    def test(self, simulator:Simulator, test_config):
        """
        Full test logic for the simulator.
        :param simulator: Simulator instance.
        :param agent: MatchingAgent or other RL agent.
        :param test_config: Test configuration (e.g., test_num, intervals).
        """
        df, last_stopping_index = self.initialize_test_dataframe(test_config['test_num'], test_config['column_list'])
        ax, ay = [], []

        for num in range(last_stopping_index, test_config['test_num']):
            print('Test number: ', num)
            metrics = self.run_test_episode(simulator, self.matching_agent, test_config['test_dates'])
            metrics = {k: v / len(test_config['test_dates']) for k, v in metrics.items()}  # Normalize by test dates

            # Record metrics
            ax.append(num + 1)
            ay.append(metrics['total_reward'])
            print("total_reward: ", metrics['total_reward'])
            print("pickup_time: ", metrics['pickup_time'])
            print("matching_rate: ", metrics['matched_request_num'] / metrics['total_request_num'])
            print("occupancy_rate: ", metrics['occupancy_rate'])
            record_array = [
                metrics['total_reward'], metrics['matched_transfer_request_num'], metrics['matched_request_num'],
                metrics['transfer_request_num'], metrics['long_request_num'], metrics['matched_long_request_num'],
                metrics['matched_medium_request_num'], metrics['medium_request_num'], metrics['matched_short_request_num'],
                metrics['short_request_num'], metrics['total_request_num'], metrics['waiting_time'], metrics['pickup_time'],
                metrics['occupancy_rate'], metrics['occupancy_rate_no_pickup']
            ]
            if num == 0:
                df.iloc[0, :15] = record_array
            else:
                df.iloc[num, :15] = (df.iloc[(num - 1), : 1].values * num + record_array) / (num + 1)

            if num % test_config['save_interval'] == 0:
                self.save_results(df, test_config['output_path'], num, simulator.method)
                
            # Convergence check
            converged, index = self.check_convergence(df, num, test_config['interval'], test_config['threshold'])
            if converged:
                break
        # 最终保存并计算比率
        self.save_and_calculate_ratios(df, test_config['output_path'], test_config['test_num'] - 1, test_config['method'])        

    def render(self):
        """
        可视化
        """
        self.simulator.render()

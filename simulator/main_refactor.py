from simulator_env import Simulator
from simulator_trainer import SimulatorTrainer 
from pricing_agent import PricingAgent
from matching_agent import MatchingAgent  
import pickle
import numpy as np
from config import *
from path import *
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")
import os
from utilities import *
from sarsa import SarsaAgent
from matplotlib import pyplot as plt
from A2C import * # you may comment this import if you are running matching 
# python D:\Feng\drl_subway_comp\main.py    


if __name__ == "__main__":
    driver_num = [100]
    max_distance_num = [1]

    cruise_flag = [True if env_params['rl_mode'] == 'matching' else False]
    pickup_flag = ['rg']
    delivery_flag = ['rg']  

    # track的格式为[{'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]],
    # 'driver_2' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]},
    # {'driver_1' : [[lng, lat, status, time_a], [lng, lat, status, time_b]]}]
    for pc_flag in pickup_flag:
        for dl_flag in delivery_flag:
            for cr_flag in cruise_flag:  
                for single_driver_num in driver_num:
                    for single_max_distance_num in max_distance_num:
                        env_params['pickup_mode'] = pc_flag
                        env_params['delivery_mode'] = dl_flag
                        env_params['cruise_flag'] = cr_flag
                        env_params['driver_num'] = single_driver_num
                        env_params['maximal_pickup_distance'] = single_max_distance_num
                        env_params['experiment_mode'] = 'train'
                        simulator = Simulator(**env_params)
                        # Comment simulator.reset() below if you are not running matching with instant_reward_no_subway
                        simulator.reset() # 每次循环都重置simulator的状态
                        track_record = []
                        t = time.time()

                        if env_params['rl_mode'] == "matching":
                            if simulator.experiment_mode == 'test':
                                # TODO:继续修改、重构
                                pricing_agent = PricingAgent(strategy="static")
                                matching_agent_params = {
                                    'strategy_type': simulator.method,
                                    'strategy_params': sarsa_params,
                                    'load_path': load_path + 'episode_4000\\sarsa_q_value_table_epoch_4000.pickle',
                                    'flag_load': FLAG_LOAD  # 新增 FLAG_LOAD 参数
                                }

                                matching_agent = MatchingAgent(**matching_agent_params)
                                # Initialize SimulatorTrainer
                                trainer = SimulatorTrainer(
                                    simulator=simulator,
                                    matching_agent=matching_agent,
                                    pricing_agent=pricing_agent
                                )

                                column_list = ['total_reward', 'matched_transfer_request_num', 'matched_request_num',
                                               'transfer_request_num',
                                               'long_request_num',
                                               'matched_long_request_num', 'matched_medium_request_num',
                                               'medium_request_num',
                                               'matched_short_request_num',
                                               'short_request_num', 'total_request_num',
                                               'matched_transfer_request_ratio', 'transfer_long_request_ratio',
                                               'matched_long_request_ratio', 'matched_medium_request_ratio',
                                               'matched_short_request_ratio',
                                               'matched_request_ratio','waiting_time','pickup_time','occupancy_rate','occupancy_rate_no_pickup']
                                
                                trainer.test(
                                        simulator=simulator,
                                        test_config={
                                            'test_num': 10,
                                            'test_dates': TEST_DATE_LIST,
                                            'column_list': column_list,
                                            'method': simulator.method,
                                            'save_interval': 10,
                                            'interval': 5,         # Interval for checking convergence
                                            'threshold': 5,        # Convergence threshold
                                            'output_path': load_path
                                        }
                                        )

                            elif simulator.experiment_mode == 'train':
                                print("training process")
                                epsilons = get_exponential_epsilons(INIT_EPSILON, FINAL_EPSILON, 2500, decay=DECAY,
                                                                    pre_steps=PRE_STEP)
                                epsilons = np.concatenate([epsilons, np.zeros(NUM_EPOCH - 2500)])
                                # epsilons = np.zeros(NUM_EPOCH)
                                total_reward_record = np.zeros(NUM_EPOCH)
                                matching_agent_params = {
                                    'strategy_type': simulator.method,
                                    'strategy_params': sarsa_params,
                                    'load_path': load_path + 'episode_1800\\sarsa_q_value_table_epoch_1800.pickle',
                                    'flag_load': FLAG_LOAD  # 新增 FLAG_LOAD 参数
                                }
                                pricing_agent = PricingAgent(strategy="static")
                                matching_agent = MatchingAgent(**matching_agent_params)
                                # Initialize SimulatorTrainer
                                trainer = SimulatorTrainer(
                                    simulator=simulator,
                                    matching_agent=matching_agent,
                                    pricing_agent=pricing_agent
                                )

                                trainer.train(
                                    simulator=simulator,
                                    train_config={
                                        'num_epochs': NUM_EPOCH,
                                        'train_dates': TRAIN_DATE_LIST,
                                        'epsilons': epsilons,
                                        'save_interval': 200,
                                        'output_path': load_path,
                                        'flag_load': FLAG_LOAD,
                                    }
                                )


                            for step in tqdm(range(simulator.finish_run_step)):
                                new_tracks = simulator.rl_step()
                                track_record.append(new_tracks)

                            # output3:    
                            match_and_cancel_track_list = simulator.match_and_cancel_track
                            file_path = './output3/' + pc_flag + "_" + dl_flag + "_" + "cruise="+str(cr_flag)
                            if not os.path.exists(file_path):
                                os.makedirs(file_path)
                            pickle.dump(track_record, open(file_path + '/records_driver_num_'+str(single_driver_num)+'.pickle', 'wb'))
                            pickle.dump(simulator.requests, open(file_path + '/passenger_records_driver_num_'+str(single_driver_num)+'.pickle', 'wb'))

                            pickle.dump(match_and_cancel_track_list,open(file_path+'/match_and_cacel_'+str(single_driver_num)+'.pickle','wb'))
                            file = open(file_path + '/time_statistic.txt', 'a')
                            file.write(str(time.time()-t)+'\n')

                        elif env_params['rl_mode'] == "reposition":
                            len_time_binary = 18
                            len_grid_binary = 7
                            if simulator.experiment_mode == 'train':
                                epsilons = get_exponential_epsilons(INIT_EPSILON, FINAL_EPSILON, NUM_EPOCH, decay=DECAY,
                                                                    pre_steps=PRE_STEP)
                                total_reward_record = np.zeros(NUM_EPOCH)

                                # parameters
                                # here 8 is the length of binary total time steps
                                # 6 is the the length of binary total grids
                                # 37 is the length of global state vetors, equal to the numeber of grids
                                if simulator.reposition_method == 'A2C':
                                    agent_params = dict(action_dim=5, state_dim=(len_time_binary + len_grid_binary),
                                                        available_directions=df_available_directions,
                                                        load_model=False, discount_factor=DISCOUNT_FACTOR,
                                                        actor_lr=ACTOR_LR, critic_lr=CRITIC_LR,
                                                        actor_structure=ACTOR_STRUCTURE,
                                                        critic_structure=CRITIC_STRUCTURE,
                                                        model_name=simulator.reposition_method)
                                elif simulator.reposition_method == 'A2C_global_aware':
                                    agent_params = dict(action_dim=5, state_dim=(
                                                len_time_binary + len_grid_binary + 2 * side**2),
                                                        available_directions=df_available_directions,
                                                        load_model=False, discount_factor=DISCOUNT_FACTOR,
                                                        actor_lr=ACTOR_LR,
                                                        critic_lr=CRITIC_LR,
                                                        actor_structure=ACTOR_STRUCTURE,
                                                        critic_structure=CRITIC_STRUCTURE,
                                                        model_name=simulator.reposition_method)

                                repo_agent = A2C(**agent_params)
                                ax,ay= [],[]
                                for epoch in range(NUM_EPOCH):
                                    start_time = time.time()
                                    date = TRAIN_DATE_LIST[epoch % len(TRAIN_DATE_LIST)]
                                    simulator.experiment_date = date
                                    simulator.reset()
                                    for step in range(simulator.finish_run_step):
                                        grid_array, time_array, idle_drivers_by_grid, waiting_orders_by_grid = simulator.step1()

                                        action_array = np.array([])
                                        if len(grid_array) > 0:
                                            # state transformation
                                            index_grid = \
                                            np.where(grid_array.reshape(grid_array.size, 1) == simulator.zone_id_array)[
                                                1]
                                            index_time = (time_array - simulator.t_initial) // simulator.delta_t
                                            binary_index_grid = s2e(index_grid, total_len=len_grid_binary)
                                            binary_index_time = s2e(index_time, total_len=len_time_binary)
                                            state_array = np.hstack([binary_index_grid, binary_index_time])
                                            if simulator.reposition_method == 'A2C_global_aware':
                                                global_idle_driver_array = np.tile(idle_drivers_by_grid,
                                                                                   [state_array.shape[0], 1])
                                                global_wait_orders_array = np.tile(waiting_orders_by_grid,
                                                                                   [state_array.shape[0], 1])
                                                state_array = np.hstack(
                                                    [state_array, global_idle_driver_array, global_wait_orders_array])

                                            # make actions
                                            action_array = np.zeros(state_array.shape[0])
                                            for i in range(len(action_array)):
                                                action = repo_agent.egreedy_actions(state_array[i], epsilons[epoch],
                                                                                    index_grid[i])
                                                action_array[i] = action

                                        simulator.step2(action_array)

                                    # collect transitions and train the model under the end of an epoch

                                    grid_array = simulator.state_grid_array_done
                                    time_array = simulator.state_time_array_done
                                    action_array = simulator.action_array_done.astype(int)
                                    next_grid_array = simulator.next_state_grid_array_done
                                    next_time_array = simulator.next_state_time_array_done
                                    reward_array = simulator.reward_array_done
                                    done_array = np.zeros(grid_array.shape)
                                    # transform the states and next states
                                    index_grid = \
                                    np.where(grid_array.reshape(grid_array.size, 1) == simulator.zone_id_array)[1]
                                    index_time = (time_array - simulator.t_initial) // simulator.delta_t
                                    binary_index_grid = s2e(index_grid, total_len=len_grid_binary)
                                    binary_index_time = s2e(index_time, total_len=len_time_binary)
                                    state_array = np.hstack([binary_index_grid, binary_index_time])

                                    index_next_grid = np.where(
                                        next_grid_array.reshape(next_grid_array.size, 1) == simulator.zone_id_array)[1]
                                    index_next_time = (next_time_array - simulator.t_initial) // simulator.delta_t
                                    binary_index_next_grid = s2e(index_next_grid, total_len=len_grid_binary)
                                    binary_index_next_time = s2e(index_next_time, total_len=len_time_binary)
                                    next_state_array = np.hstack([binary_index_next_grid, binary_index_next_time])

                                    if simulator.reposition_method == 'A2C_global_aware':
                                        index_time = np.where(
                                            time_array.reshape(time_array.size, 1) == np.array(simulator.global_time))[
                                            1]
                                        global_idle_driver_array = np.array(simulator.global_drivers_num)[index_time, :]
                                        global_wait_orders_array = np.array(simulator.global_orders_num)[index_time, :]
                                        state_array = np.hstack(
                                            [state_array, global_idle_driver_array, global_wait_orders_array])

                                        next_time_array = ((
                                                                       next_time_array - simulator.t_initial) // simulator.delta_t) * simulator.delta_t + simulator.t_initial
                                        index_next_time = np.where(
                                            next_time_array.reshape(next_time_array.size, 1) == np.array(
                                                simulator.global_time))[1]
                                        global_next_idle_driver_array = np.array(simulator.global_drivers_num)[
                                                                        index_next_time, :]
                                        global_next_wait_orders_array = np.array(simulator.global_orders_num)[
                                                                        index_next_time, :]
                                        next_state_array = np.hstack([next_state_array, global_next_idle_driver_array,
                                                                      global_next_wait_orders_array])
                                    transitions = [state_array, action_array, reward_array, next_state_array,
                                                   done_array]
                                    # if len(transitions[1]) == 0:
                                    #     continue
                                    perceive_flag = True
                                    for i in range(5):
                                        if transitions[i].shape[0] == 0:
                                            perceive_flag = False 
                                            break 
                                    if perceive_flag:
                                        repo_agent.perceive(transitions)

                                    end_time = time.time()
                                    total_reward_record[epoch] = simulator.total_reward

                                    print('epoch:', epoch)
                                    print('epoch running time: ', end_time - start_time)
                                    print('epoch epsilon: ', epsilons[epoch])
                                    print('epoch total reward: ', simulator.total_reward)

                                    ax.append(epoch)
                                    ay.append(simulator.total_reward)

                                    if epoch % 100 == 0:  # save the result every 100 epochs
                                        repo_agent.save_model(epoch)
                                        pickle.dump(total_reward_record, open(load_path + 'training_results_record_' +
                                                                              env_params[
                                                                                  'reposition_method'] + '.pickle',
                                                                              'wb'))

                                    if epoch == STOP_EPOCH:
                                        repo_agent.save_model(epoch)
                                        pickle.dump(total_reward_record, open(load_path + 'training_results_record_' +
                                                                              env_params[
                                                                                  'reposition_method'] + '.pickle',
                                                                              'wb'))
                                        break

                                plt.plot(ax,ay)
                                plt.plot(ax,ay,'r+')
                                plt.show()

                            elif simulator.experiment_mode == 'test':
                                simulator = Simulator(**env_params)
                                column_list = ['total_reward', 'matched_request_num',
                                               'long_request_num',
                                               'matched_long_request_num', 'matched_medium_request_num',
                                               'medium_request_num',
                                               'matched_short_request_num',
                                               'short_request_num', 'total_request_num',
                                               'waiting_time','pickup_time','occupancy_rate','occupancy_rate_no_pickup',
                                               'matched_long_request_ratio', 'matched_medium_request_ratio',
                                               'matched_short_request_ratio',
                                               'matched_request_ratio']
                                test_num = 6
                                test_interval = 3
                                threshold = 10

                                df = pd.DataFrame(np.zeros([test_num, len(column_list)]), columns=column_list)
                                remaining_index_array = np.where(df['total_reward'].values == 0)[0]
                                if len(remaining_index_array > 0):
                                    last_stopping_index = remaining_index_array[0]

                                if simulator.reposition_method == 'A2C':
                                    agent_params = dict(action_dim=5, state_dim=(len_time_binary + len_grid_binary),
                                                        available_directions=df_available_directions,
                                                        load_model=True, discount_factor=DISCOUNT_FACTOR,
                                                        actor_lr=ACTOR_LR,
                                                        critic_lr=CRITIC_LR,
                                                        actor_structure=ACTOR_STRUCTURE,
                                                        critic_structure=CRITIC_STRUCTURE,
                                                        model_name=simulator.reposition_method)
                                elif simulator.reposition_method == 'A2C_global_aware':
                                    agent_params = dict(action_dim=5,
                                                        state_dim=(
                                                                    len_time_binary + len_grid_binary + 2 * side**2),
                                                        available_directions=df_available_directions,
                                                        load_model=True, discount_factor=DISCOUNT_FACTOR,
                                                        actor_lr=ACTOR_LR,
                                                        critic_lr=CRITIC_LR,
                                                        actor_structure=ACTOR_STRUCTURE,
                                                        critic_structure=CRITIC_STRUCTURE,
                                                        model_name=simulator.reposition_method)
                                elif simulator.reposition_method == 'random_cruise' or simulator.reposition_method == 'stay' \
                                or simulator.reposition_method == "nearby":
                                    agent_params = dict(action_dim=5,
                                                        state_dim=(
                                                                    len_time_binary + len_grid_binary + 2 * side**2),
                                                        available_directions=df_available_directions,
                                                        load_model=False, discount_factor=DISCOUNT_FACTOR,
                                                        actor_lr=ACTOR_LR,
                                                        critic_lr=CRITIC_LR,
                                                        actor_structure=ACTOR_STRUCTURE,
                                                        critic_structure=CRITIC_STRUCTURE,
                                                        model_name='')
                                repo_agent = A2C(**agent_params)

                                for num in range(last_stopping_index, test_num):
                                    total_reward = 0
                                    total_request_num = 0
                                    long_request_num = 0
                                    medium_request_num = 0
                                    short_request_num = 0
                                    matched_request_num = 0
                                    matched_long_request_num = 0
                                    matched_medium_request_num = 0
                                    matched_short_request_num = 0
                                    occupancy_rate = 0
                                    occupancy_rate_no_pickup = 0
                                    wait_time = 0
                                    pickup_time = 0
                                    for date in TEST_DATE_LIST:
                                        simulator.experiment_date = date
                                        simulator.reset()
                                        for step in range(simulator.finish_run_step):
                                            # print(step)
                                            grid_array, time_array, idle_drivers_by_grid, waiting_orders_by_grid = simulator.step1()

                                            action_array = np.array([])
                                            if len(grid_array) > 0:
                                                # state transformation
                                                index_grid = np.where(
                                                    grid_array.reshape(grid_array.size, 1) == simulator.zone_id_array)[
                                                    1]
                                                index_time = (time_array - simulator.t_initial) // simulator.delta_t
                                                binary_index_grid = s2e(index_grid, total_len=len_grid_binary)
                                                binary_index_time = s2e(index_time, total_len=len_time_binary)
                                                state_array = np.hstack([binary_index_grid, binary_index_time])
                                                if simulator.reposition_method == 'A2C_global_aware':
                                                    global_idle_driver_array = np.tile(idle_drivers_by_grid,
                                                                                       [state_array.shape[0], 1])
                                                    global_wait_orders_array = np.tile(waiting_orders_by_grid,
                                                                                       [state_array.shape[0], 1])
                                                    state_array = np.hstack([state_array, global_idle_driver_array,
                                                                             global_wait_orders_array])

                                                # make actions
                                                action_array = np.zeros(state_array.shape[0])
                                                for i in range(len(action_array)):
                                                    if simulator.reposition_method == 'A2C' or simulator.reposition_method == 'A2C_global_aware':
                                                        action = repo_agent.egreedy_actions(state_array[i], -1,
                                                                                            index_grid[i])
                                                    elif simulator.reposition_method == 'random_cruise':
                                                        action = repo_agent.egreedy_actions(state_array[i], 2,
                                                                                            index_grid[i])
                                                    elif simulator.reposition_method == 'stay':
                                                        action = 0
                                                    elif simulator.reposition_method == 'nearby':
                                                        grid_id = index_grid[i]
                                                        target = [grid_id]
                                                        
                                                        if grid_id - side > 0:
                                                            target.append(grid_id - side)                                            
                                                        elif int((grid_id + 1) / side) == int(grid_id / side) and grid_id + 1 < side * side:
                                                            target.append(grid_id + 1)
                                                        elif grid_id + side < side * side:
                                                            target.append(grid_id + side)
                                                        elif int((grid_id - 1) / side) == int(grid_id / side) and grid_id - 1 > 0:
                                                            target.append(grid_id - 1)

                                                        action = 0
                                                        max_value = -1
                                                        for idx,item in enumerate(target):
                                                            if item in simulator.grid_value.keys() and simulator.grid_value[item] > max_value:
                                                                max_value = simulator.grid_value[item]
                                                                action = idx
                                            
                                                    action_array[i] = action

                                            simulator.step2(action_array)

                                        total_reward += simulator.total_reward
                                        total_request_num += simulator.total_request_num
                                        matched_request_num += simulator.matched_requests_num
                                        long_request_num += simulator.long_requests_num
                                        medium_request_num += simulator.medium_requests_num
                                        short_request_num += simulator.short_requests_num
                                        matched_long_request_num += simulator.matched_long_requests_num
                                        matched_medium_request_num += simulator.matched_medium_requests_num
                                        matched_short_request_num += simulator.matched_short_requests_num
                                        occupancy_rate += simulator.occupancy_rate
                                        occupancy_rate_no_pickup = simulator.occupancy_rate_no_pickup
                                        wait_time += simulator.waiting_time / simulator.matched_requests_num
                                        pickup_time += simulator.pickup_time / simulator.matched_requests_num

                                    total_reward = total_reward / len(TEST_DATE_LIST)
                                    total_request_num = total_request_num / len(TEST_DATE_LIST)
                                    matched_request_num = matched_request_num / len(TEST_DATE_LIST)
                                    long_request_num /= len(TEST_DATE_LIST)
                                    medium_request_num /= len(TEST_DATE_LIST)
                                    short_request_num /= len(TEST_DATE_LIST)
                                    matched_long_request_num /= len(TEST_DATE_LIST)
                                    matched_medium_request_num /= len(TEST_DATE_LIST)
                                    matched_short_request_num /= len(TEST_DATE_LIST)
                                    occupancy_rate /= len(TEST_DATE_LIST)
                                    occupancy_rate_no_pickup /= len(TEST_DATE_LIST)
                                    wait_time /= len(TEST_DATE_LIST)
                                    pickup_time /= len(TEST_DATE_LIST)

                                    record_array = np.array(
                                        [total_reward, matched_request_num,
                                         long_request_num,matched_long_request_num, 
                                         matched_medium_request_num,medium_request_num,
                                         matched_short_request_num,short_request_num, total_request_num,
                                         wait_time,pickup_time,occupancy_rate,occupancy_rate_no_pickup,
                                               ])

                                    if num == 0:
                                        df.iloc[0, :13] = record_array
                                    else:
                                        df.iloc[num, :13] = (df.iloc[(num - 1), :13].values * num + record_array) / (
                                                    num + 1)

                                    if num % 1 == 0:  # save the result every 10
                                        pickle.dump(df, open(load_path + 'performance_record_test_' + env_params[
                                            'reposition_method'] + '.pickle', 'wb'))

                                    if num >= (test_interval - 1):
                                        profit_array = df.loc[(num - test_interval):num, 'total_reward'].values
                                        # print(profit_array)
                                        error = np.abs(np.max(profit_array) - np.min(profit_array))
                                        print('error: ', error)
                                        if error < threshold:
                                            index = num
                                            print('converged at index ', index)
                                            break
                                
                                df.loc[:(num), 'matched_long_request_ratio'] = df.loc[:(num),
                                                                               'matched_long_request_num'].values / df.loc[
                                                                                                                    :(num),
                                                                                                                    'long_request_num'].values
                                df.loc[:(num), 'matched_medium_request_ratio'] = df.loc[:(num),
                                                                                 'matched_medium_request_num'].values / df.loc[
                                                                                                                        :(
                                                                                                                            num),
                                                                                                                        'medium_request_num'].values
                                df.loc[:(num), 'matched_short_request_ratio'] = df.loc[:(num),
                                                                                'matched_short_request_num'].values / df.loc[
                                                                                                                      :(
                                                                                                                          num),
                                                                                                                      'short_request_num'].values
                                df.loc[:(num), 'matched_request_ratio'] = df.loc[:(num),
                                                                          'matched_request_num'].values / df.loc[:(num),
                                                                                                          'total_request_num'].values



                                pickle.dump(df, open(load_path + 'performance_record_test_' + env_params[
                                    'reposition_method'] + '.pickle', 'wb'))
                                print(df.iloc[num, :])
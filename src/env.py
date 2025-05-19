import gzip
import json
import logging
import math
import os
import random
import requests
import traceback
import habitat_sim

import pandas as pd
import numpy as np

from PIL import Image
from simWrapper import PolarAction, SimWrapper
from agent import *
from utils import *

class Env:
    """
    Base class for creating an environment for embodied navigation tasks.
    This class defines the setup, logging, running, and evaluation of episodes.
    """

    task = 'Not defined'

    def __init__(self, cfg: dict):
        """
        Initializes the environment with the provided configuration.

        Args:
            cfg (dict): Configuration dictionary containing environment, simulation, and agent settings.
        """
        self.cfg = cfg['env_cfg']
        self.sim_cfg = cfg['sim_cfg']
        if self.cfg['name'] == 'default':
            self.cfg['name'] = f'default_{random.randint(0, 1000)}'
        self._initialize_logging(cfg)
        self._initialize_agent(cfg)
        self.outer_run_name = self.task + '_' + self.cfg['name']
        self.inner_run_name = f'{self.cfg["instance"]}_of_{self.cfg["instances"]}'
        self.curr_run_name = "Not initialized"
        self.path_calculator = habitat_sim.MultiGoalShortestPath()
        self.simWrapper: SimWrapper = None
        self.num_episodes = 0
        self._initialize_experiment()

    def _initialize_agent(self, cfg: dict):
        """Initializes the agent for the environment."""
        PolarAction.default = PolarAction(cfg['agent_cfg']['default_action'], 0, 'default')
        cfg['agent_cfg']['sensor_cfg'] = cfg['sim_cfg']['sensor_cfg']
        agent_cls = globals()[cfg['agent_cls']]
        self.agent: Agent = agent_cls(cfg['agent_cfg']) 

    def _initialize_logging(self, cfg: dict):
        """
        Initializes logging for the environment.

        Args:
            cfg (dict): Configuration dictionary containing logging settings.
        """
        self.log_file = f'logs/{cfg["task"]}_{self.cfg["name"]}/{self.cfg["instance"]}_of_{self.cfg["instances"]}.txt'
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        if self.cfg['parallel']:
            logging.basicConfig(
                filename=self.log_file,
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )
        else:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s %(levelname)s: %(message)s'
            )

    def _initialize_experiment(self):
        """
        Abstract method for setting up the environment and initializing all required variables.
        Should be implemented in derived classes.
        """
        raise NotImplementedError

    def run_experiment(self):
        """
        Runs the experiment by iterating over episodes.
        """
        base_start = self.cfg.get('start_from', 0)

        instance_id = self.cfg['instance']
        start_ndx = base_start + (instance_id * self.cfg['num_episodes'])
        end_ndx = min(start_ndx + self.cfg['num_episodes'], self.num_episodes)
        
        logging.info(f"Instance {instance_id} processing episodes from {start_ndx} to {end_ndx-1}")
        
        for episode_ndx in range(start_ndx, end_ndx):
            self.wandb_log_data = {
                'episode_ndx': episode_ndx,
                'instance': self.inner_run_name,
                'total_episodes': self.cfg['instances'] * self.cfg['num_episodes'],
                'task': self.task,
                'task_data': {},
                'spl': 0,
                'goal_reached': False,
                'trustworthy_metrics':0,
            }

            try:
                self._run_episode(episode_ndx)
            except Exception as e:
                log_exception(e)
                self.simWrapper.reset()


    def _run_episode(self, episode_ndx: int):
        """
        Runs a single episode.

        Args:
            episode_ndx (int): The index of the episode to run.
        """
        obs = self._initialize_episode(episode_ndx)

        logging.info(f'\n===================STARTING RUN: {self.curr_run_name} ===================\n')
        for _ in range(self.cfg['max_steps']):
            try:
                if hasattr(self, 'habitat_steps') and self.habitat_steps >= 500:
                    logging.info(f"Episode terminated: Reached Habitat max steps limit (500)")
                    break
                    
                agent_action = self._step_env(obs)
                if agent_action is None:
                    break
                obs = self.simWrapper.step(agent_action)

            except Exception as e:
                log_exception(e)

            finally:
                self.step += 1
        self._post_episode()

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode. This method should be implemented in derived classes.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        self.step = 0
        self.habitat_steps = 0  
        self.init_pos = None
        self.df = pd.DataFrame({})
        self.agent_distance_traveled = 0
        self.prev_agent_position = None
        if hasattr(self, 'area_tracker'):
            self.area_tracker.reset()

    def _calculate_habitat_steps(self, action: PolarAction):
        if action.type == 'stop':
            return 1  

        step_count = 0

        if action.r > 0:
            forward_steps = math.ceil(action.r / 0.25)
            step_count += forward_steps

        if abs(action.theta) > 0:
            degrees = abs(math.degrees(action.theta))
            turn_steps = math.ceil(degrees / 30)
            step_count += turn_steps
        
        return max(1, step_count)

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment. This method should be implemented in derived classes.
        """
        logging.info(f'Step {self.step}')
        agent_state = obs['agent_state']
        if self.prev_agent_position is not None:
            self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)

            if hasattr(self, 'last_agent_action') and self.last_agent_action:
                habitat_step_delta = self._calculate_habitat_steps(self.last_agent_action)
                self.habitat_steps += habitat_step_delta
                logging.info(f"Habitat step count: {self.habitat_steps}/500")
                
        self.prev_agent_position = agent_state.position
        obs['habitat_steps'] = self.habitat_steps
        return None

    def _post_episode(self):
        """
        Called after the episode is complete, saves the dataframe log, and resets the environment.
        Sends a request to the aggregator server if parallel is set to True.
        """
        self.df.to_pickle(f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/df_results.pkl')
        self.simWrapper.reset()
        self.agent.reset()
        
        if self.cfg['parallel']:
            try:
                session = requests.Session()
                session.trust_env = False 
                
                self.wandb_log_data['spend'] = self.agent.get_spend()
                self.wandb_log_data['default_rate'] = len(self.df[self.df['success'] == 0]) / len(self.df)

                response = session.post(
                    f'http://localhost:{self.cfg["port"]}/log',
                    json=self.wandb_log_data,
                    proxies=None,
                    timeout=30
                )
                
                if response.status_code != 200:
                    logging.error(f"Failed to send metrics: {response.text}")
                    
            except Exception as e:
                tb = traceback.extract_tb(e.__traceback__)
                for frame in tb:
                    logging.error(f"Frame {frame.filename} line {frame.lineno}")
                logging.error(e)

        logging.info('\n===================RUN COMPLETE===================\n')

        if hasattr(self, 'simWrapper') and self.simWrapper is not None:
            try:
                logging.info("close habitat...")
                self.simWrapper.reset()
                self.simWrapper = None 
            except Exception as e:
                logging.warning(f"close wrong: {e}")

        if hasattr(self, 'path_calculator'):
            if hasattr(self.path_calculator, 'requested_ends'):
                self.path_calculator.requested_ends = np.array([], dtype=np.float32) 

        if hasattr(self, 'current_episode'):
            self.current_episode = {}

        import gc
        gc.collect()

    def _log(self, images: dict, step_metadata: dict, logging_data: dict):
        """
        Appends the step metadata to the dataframe, and saves the images and general metadata to disk.

        Args:
            images (dict): Images generated during the step.
            step_metadata (dict): Metadata for the current step.
            logging_data (dict): General logging data.
        """
        self.df = pd.concat([self.df, pd.DataFrame([step_metadata])], ignore_index=True)

        if self.step % self.cfg['log_freq'] == 0 or step_metadata['success'] == 0:
            path = f'logs/{self.outer_run_name}/{self.inner_run_name}/{self.curr_run_name}/step{self.step}'
            if not step_metadata['success']:
                path += '_ERROR'
            os.makedirs(path, exist_ok=True)
            for name, im in images.items():
                im = Image.fromarray(im[:, :, 0:3], mode='RGB')
                im.save(f'{path}/{name}.png')
            with open(f'{path}/details.txt', 'w') as file:
                if step_metadata['success']:
                    for k, v in logging_data.items():
                        file.write(f'{k}\n{v}\n\n')

    def _calculate_metrics(self, agent_state: habitat_sim.AgentState, agent_action: PolarAction, geodesic_path: int, max_steps: int):
        """
        Calculates the navigation metrics at a given step.
        """
        metrics = {}

        self.path_calculator.requested_start = agent_state.position
        nav_distance = self.simWrapper.get_path(self.path_calculator)
        metrics['distance_to_goal'] = nav_distance

        view_euclidean_distance = float('inf')
        if hasattr(self, 'current_episode'):
            if isinstance(self.current_episode, dict) and 'view_positions' in self.current_episode:
                distances = []
                for pos in self.current_episode['view_positions']:
                    distances.append(np.linalg.norm(np.array(agent_state.position) - np.array(pos)))
                if distances:
                    view_euclidean_distance = min(distances)
                    metrics['view_euclidean_distance'] = view_euclidean_distance

        euclidean_distance = float('inf')
        if hasattr(self, 'current_episode') and 'object_positions' in self.current_episode:
            if self.current_episode['object_positions']:
                euclidean_distances = []
                for pos in self.current_episode['object_positions']:
                    if isinstance(pos, (list, np.ndarray)):
                        euclidean_distances.append(np.linalg.norm(np.array(agent_state.position) - np.array(pos)))
                
                if euclidean_distances:
                    euclidean_distance = min(euclidean_distances)
                    metrics['euclidean_distance'] = euclidean_distance

        logging.info(f"nav_distance: {nav_distance}m, view_distance: {view_euclidean_distance}m")
        
        metrics['distance_to_goal'] = self.simWrapper.get_path(self.path_calculator)
        metrics['spl'] = 0
        metrics['goal_reached'] = False
        metrics['done'] = False
        metrics['finish_status'] = 'running'
        metrics['habitat_steps'] = self.habitat_steps
        if hasattr(self.agent, 'area_tracker'):
            if hasattr(self.agent.area_tracker, 'calculate_aori'):
                current_aori = self.agent.area_tracker.calculate_aori()
                metrics['trustworthy_metrics'] = float(current_aori)
            elif hasattr(self.agent, 'calculate_trustworthy_metrics'):
                trustworthy_data = self.agent.calculate_trustworthy_metrics()
                metrics['trustworthy_metrics'] = float(trustworthy_data.get("aori", 0.0)
            else:
                metrics['trustworthy_metrics'] = 0.0
        else:
            metrics['trustworthy_metrics'] = 0.0

        if hasattr(self.agent, 'calculate_trustworthy_metrics'):
            trustworthy_data = self.agent.calculate_trustworthy_metrics()
            metrics['repeat_percentage'] = trustworthy_data.get("repeat_percentage", 0)
            metrics['coverage_efficiency'] = trustworthy_data.get("coverage_efficiency", 0)
        
        if agent_action is PolarAction.stop or self.step + 1 == max_steps or self.habitat_steps >= 500:
            metrics['done'] = True
            if self.habitat_steps >= 500:
                metrics['finish_status'] = 'habitat_max_steps'
                logging.info(f"Reached Habitat max steps limit (500)")

            logging.info(f"Stop action triggered:")
            logging.info(f"导航距离: {metrics['distance_to_goal']}m")
            logging.info(f"直线距离: {euclidean_distance}m")
            logging.info(f"Success threshold: {self.cfg['success_threshold']}")
            logging.info(f"Agent position: {agent_state.position}")

            effective_distance = min(metrics['distance_to_goal'], view_euclidean_distance)
            if effective_distance < self.cfg['success_threshold']:
                metrics['finish_status'] = 'success'
                metrics['goal_reached'] = True
                metrics['spl'] = geodesic_path / max(geodesic_path, self.agent_distance_traveled)
                self.wandb_log_data.update({
                    'spl': metrics['spl'],
                    'goal_reached': metrics['goal_reached'],
                    'trustworthy_metrics': metrics['trustworthy_metrics']
                })
            else:
                if agent_action is PolarAction.stop:
                    metrics['finish_status'] = 'fp'
                else:
                    metrics['finish_status'] = 'max_steps'
        
        logging.info(f"spl: {metrics['spl']}; goal_reached: {metrics['goal_reached']}; trustworthy_metrics: {metrics['trustworthy_metrics']}")

        return metrics

class GOATEnv(Env):
    """
    Environment for the GOAT task, extending the base Env class.
    This class defines the setup, initialization, and running of GOAT episodes.
    """

    task = 'GOAT'

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.split = 'val' if 'val' in self.cfg['split'] else 'train'
        self.sim_cfg['scene_config'] = "data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

        self.all_episodes = []
        self.goals = {}
        for f in sorted(os.listdir(f'data/datasets/goat_bench/hm3d/v1/{self.cfg["split"]}/content')):
            with gzip.open(f'data/datasets/goat_bench/hm3d/v1/{self.cfg["split"]}/content/{f}', 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals']
                self.all_episodes += js['episodes']
        self.num_episodes = len(self.all_episodes)

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the GOAT task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)

        episode = self.all_episodes[episode_ndx]
        f, glb = episode['scene_id'].split('/')[-2:]
        hsh = f[6:]
        goals = self.goals[hsh]
        self.sim_cfg['scene_id'] = f[2:5]
        self.sim_cfg['scene_path'] = f'data/scene_datasets/hm3d/{self.split}/{f}/{glb}'
        self.simWrapper = SimWrapper(self.sim_cfg)
        self.current_episode = []

        for goal in episode['tasks']:
            name = goal[0]
            mode = goal[1]
            subgoal = {'name': name, 'mode': mode, 'id': goal[2], 'view_points': []}
            for obj in goals[f'{f[6:]}.basis.glb_{name}']:
                if mode == 'object':
                    subgoal['view_points'] += [a['agent_state']['position'] for a in obj['view_points']]
                else:
                    if obj['object_id'] == goal[2]:
                        subgoal['view_points'] = [a['agent_state']['position'] for a in obj['view_points']]
                        if mode == 'description':
                            subgoal['lang_desc'] = obj['lang_desc']
                        if mode == 'image':
                            image_ndx = goal[3]
                            subgoal['image_position'] = obj['image_goals'][image_ndx]['position']
                            subgoal['image_rotation'] = obj['image_goals'][image_ndx]['rotation']

            self.current_episode.append(subgoal)

        logging.info(f'\nRUNNING EPISODE {episode_ndx}, SCENE: {self.simWrapper.scene_id}')
        for i, subgoal in enumerate(self.current_episode):
            logging.info(f'Goal {i}: {subgoal["name"]}, {subgoal["mode"]}')

        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_goal_ndx = 0
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"
        self.last_goal_reset = -1
        self.path_calculator.requested_ends = np.array(self.current_episode[self.curr_goal_ndx]['view_points'], dtype=np.float32)
        self.path_calculator.requested_start = self.init_pos
        self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)

        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment for the GOAT task.

        Args:
            obs (dict): The current observation.

        Returns:
            list: The next action to be taken by the agent.
        """
        super()._step_env(obs)

        goal = self.current_episode[self.curr_goal_ndx]
        obs['goal'] = goal
        if goal['mode'] == 'image':
            position = goal['image_position']
            rotation = goal['image_rotation']
            goal_im = self.simWrapper.get_goal_image(position, rotation)
            put_text_on_image(goal_im, f"GOAL IMAGE: {goal['name']}", bg_color=(255, 255, 255), location='top_center')
            obs['goal']['goal_image'] = goal_im

        agent_state = obs['agent_state']
        self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position
        agent_action, metadata = self.agent.step(obs)
        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']

        metrics = self._calculate_metrics(agent_state, agent_action, self.curr_shortest_path, self.last_goal_reset + 1 + self.cfg['max_steps_per_subgoal'])
        step_metadata.update(metrics)

        if metrics['done']:
            self.wandb_log_data['task_data'].setdefault('goal_data', []).append({
                'goal_mode': goal['mode'],
                'goal_reached': metrics['goal_reached'],
                'spl': metrics['spl'],
                'trustworthy_metrics': metrics['trustworthy_metrics']
            })

            if 'spl' in self.wandb_log_data:
                del self.wandb_log_data['spl']
            if 'goal_reached' in self.wandb_log_data:
                del self.wandb_log_data['goal_reached']
            if 'trustworthy_metrics' in self.wandb_log_data:
                del self.wandb_log_data['trustworthy_metrics']
            if self.curr_goal_ndx + 1 == len(self.current_episode):
                agent_action = None
            else:
                self.agent.reset_goal()
                self.agent_distance_traveled = 0
                agent_action = PolarAction.null
                self.curr_goal_ndx += 1
                self.last_goal_reset = self.step
                goal = self.current_episode[self.curr_goal_ndx]
                self.path_calculator.requested_ends = np.array(goal['view_points'], dtype=np.float32)
                self.path_calculator.requested_start = obs['agent_state'].position
                self.curr_shortest_path = self.simWrapper.get_path(self.path_calculator)

                logging.info(f'New goal {goal["mode"]}: {goal["name"]}, GEODESIC: {self.curr_shortest_path}')

        self._log(images, step_metadata, logging_data)

        return agent_action

class ObjectNavEnv(Env):
    """
    Environment for the ObjectNav task, extending the base Env class.
    This class defines the setup, initialization, and running of ObjectNav episodes.
    """

    task = 'ObjectNav'

    def _initialize_experiment(self):
        """
        Initializes the experiment by setting up the dataset split, scene configuration, and goals.
        """
        self.all_episodes = []
        self.sim_cfg['scene_config'] = "data/scene_datasets/hm3dv1/hm3d_annotated_basis.scene_dataset_config.json"
        self.goals = {}

        for f in sorted(os.listdir(f'data/datasets/objectnav_hm3d_v1/{self.cfg["split"]}/content')):
            with gzip.open(f'data/datasets/objectnav_hm3d_v1/{self.cfg["split"]}/content/{f}', 'rt') as gz:
                js = json.load(gz)
                hsh = f.split('.')[0]
                self.goals[hsh] = js['goals_by_category']
                self.all_episodes += js['episodes']

        self.num_episodes = len(self.all_episodes)

    def _initialize_episode(self, episode_ndx: int):
        """
        Initializes the episode for the ObjectNav task.

        Args:
            episode_ndx (int): The index of the episode to initialize.
        """
        super()._initialize_episode(episode_ndx)
        episode = self.all_episodes[episode_ndx]
        f = episode['scene_id'].split('/')[1:]
        self.sim_cfg['scene_id'] = f[1][2:5]
        self.sim_cfg['scene_path'] = f'data/scene_datasets/hm3dv1/{self.cfg["split"]}/{f[1]}/{f[2]}'
        self.simWrapper = SimWrapper(self.sim_cfg)

        goals = self.goals[f[1][6:]]
        all_objects = goals[f'{f[-1]}_{episode["object_category"]}']
        view_positions = []
        for obj in all_objects:
            for vp in obj['view_points']:
                view_positions.append(vp['agent_state']['position'])
        self.path_calculator.requested_ends = np.array(view_positions, dtype=np.float32)
        logging.info(f'RUNNING EPISODE {episode_ndx} with {episode["object_category"]} and {len(all_objects)} instances. GEODESIC DISTANCE: {episode["info"]["geodesic_distance"]}')
        if episode['object_category'] == 'tv_monitor':
            episode['object_category'] = 'television, tv screen or tv monitor'
        elif episode['object_category'] == "sofa":
            episode['object_category'] = 'sofa, couch or settee'
        elif episode['object_category'] == "chair":
            episode['object_category'] = 'chair, seat stool'
        elif episode['object_category'] == "plant":
            episode['object_category'] = 'plant or dried/artificial floral arrangement' 
        self.current_episode = {
            'object': episode['object_category'],
            'shortest_path': episode['info']['geodesic_distance'],
            'object_positions': [a['position'] for a in all_objects],
            'view_positions': view_positions
        }
        self.init_pos = np.array(episode['start_position'])
        self.simWrapper.set_state(pos=self.init_pos, quat=episode['start_rotation'])
        self.curr_run_name = f"{episode_ndx}_{self.simWrapper.scene_id}"

        obs = self.simWrapper.step(PolarAction.null)
        return obs

    def _step_env(self, obs: dict):
        """
        Takes a step in the environment for the ObjectNav task.
        """
        super()._step_env(obs)
        obs['goal'] = self.current_episode['object']
        agent_state = obs['agent_state']
        self.agent_distance_traveled += np.linalg.norm(agent_state.position - self.prev_agent_position)
        self.prev_agent_position = agent_state.position
        agent_action, metadata = self.agent.step(obs)
        self.last_agent_action = agent_action
        step_metadata = metadata['step_metadata']
        logging_data = metadata['logging_data']
        images = metadata['images']

        metrics = self._calculate_metrics(agent_state, agent_action, self.current_episode['shortest_path'], self.cfg['max_steps'])
        step_metadata.update(metrics)

        self.wandb_log_data['trustworthy_metrics'] = metrics['trustworthy_metrics']

        self._log(images, step_metadata, logging_data)

        if metrics['done']:
            agent_action = None

        return agent_action

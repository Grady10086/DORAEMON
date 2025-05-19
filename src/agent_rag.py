import logging
import math
import random
import ast
import concurrent.futures

import numpy as np
import cv2

import habitat_sim

from simWrapper import PolarAction
from utils import *
from vlm import *
from pivot import PIVOT

from sentence_transformers import SentenceTransformer, util
from PIL import Image

import faiss  # 添加 faiss 导入
from sklearn.metrics.pairwise import pairwise_distances, cosine_similarity  # 添加必要的 sklearn 导入
from scipy.cluster.hierarchy import linkage, fcluster  # 添加必要的 scipy 导入
from sentence_transformers import SentenceTransformer


class Agent:
    def __init__(self, cfg: dict):
        pass

    def step(self, obs: dict):
        """Primary agent loop to map observations to the agent's action and returns metadata."""
        raise NotImplementedError

    def get_spend(self):
        """Returns the dollar amount spent by the agent on API calls."""
        return 0

    def reset(self):
        """To be called after each episode."""
        pass


class RandomAgent(Agent):
    """Example implementation of a random agent."""
    
    def step(self, obs):
        rotate = random.uniform(-0.2, 0.2)
        forward = random.uniform(0, 1)

        agent_action = PolarAction(forward, rotate)
        metadata = {
            'step_metadata': {'success': 1}, # indicating the VLM succesfully selected an action
            'logging_data': {}, # to be logged in the txt file
            'images': {'color_sensor': obs['color_sensor']} # to be visualized in the GIF
        }
        return agent_action, metadata


class VLMNavAgent(Agent):
    """
    Primary class for the VLMNav agent. Four primary components: navigability, action proposer, projection, and prompting. Runs seperate threads for stopping and preprocessing. This class steps by taking in an observation and returning a PolarAction, along with metadata for logging and visulization.
    """
    explored_color = GREY
    unexplored_color = GREEN
    map_size = 5000
    explore_threshold = 3
    voxel_ray_size = 60
    e_i_scaling = 0.8

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.fov = cfg['sensor_cfg']['fov']
        self.resolution = (
            1080 // cfg['sensor_cfg']['res_factor'],
            1920 // cfg['sensor_cfg']['res_factor']
        )

        self.focal_length = calculate_focal_length(self.fov, self.resolution[1])
        self.scale = cfg['map_scale']
        self._initialize_vlms(cfg['vlm_cfg'])       
        self.pivot = PIVOT(self.actionVLM, self.fov, self.resolution, max_action_length=cfg['max_action_dist']) if cfg['pivot'] else None

        assert cfg['navigability_mode'] in ['none', 'depth_estimate', 'segmentation', 'depth_sensor']
        self.depth_estimator = DepthEstimator() if cfg['navigability_mode'] == 'depth_estimate' else None
        self.segmentor = Segmentor() if cfg['navigability_mode'] == 'segmentation' else None
        self.reset()

    def step(self, obs: dict):
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position

        agent_action, metadata = self._choose_action(obs)
        metadata['step_metadata'].update(self.cfg)

        if metadata['step_metadata']['action_number'] == 0:
            self.turned = self.step_ndx

        # Visualize the chosen action
        chosen_action_image = obs['color_sensor'].copy()
        self._project_onto_image(
            metadata['a_final'], chosen_action_image, agent_state,
            agent_state.sensor_states['color_sensor'], 
            chosen_action=metadata['step_metadata']['action_number']
        )
        metadata['images']['color_sensor_chosen'] = chosen_action_image

        self.step_ndx += 1
        return agent_action, metadata
    
    def get_spend(self):
        return self.actionVLM.get_spend() + self.stoppingVLM.get_spend()

    def reset(self):
        self.voxel_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.explored_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        self.stopping_calls = [-2]
        self.step_ndx = 0
        self.init_pos = None
        self.turned = -self.cfg['turn_around_cooldown']
        self.actionVLM.reset()

    def _construct_prompt(self, goal: str, prompt_type:str, adjacent_info="", num_actions: int=0, memory_context: str = ""):
        if goal['mode'] == 'object':
            task = f'Navigate to the nearest {goal["name"]}'
            first_instruction = f'Find the nearest {goal["name"]} and navigate as close as you can to it. '
        if goal['mode'] == 'description':
            first_instruction = f"Find and navigate to the {goal['lang_desc']}. Navigate as close as you can to it. "
            task = first_instruction
        if goal['mode'] == 'image':
            task = f'Navigate to the specific {goal["name"]} shown in the image labeled GOAL IMAGE. Pay close attention to the details, and note you may see the object from a different angle than in the goal image. Navigate as close as you can to it '
            first_instruction = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate as close as you can to it. "

        if prompt_type == 'stopping':        
            stopping_prompt = (f"The agent has the following navigation task: \n{task}\n. The agent has sent you an image taken from its current location{' as well as the goal image. ' if goal['mode'] == 'image' else '. '} "
                                f'Your job is to determine whether the agent is close to the specified {goal["name"].upper()}'
                                f"First, tell me what you see in the image, and tell me if there is a {goal['name']} that matches the description. Then, return 1 if the agent is close to the {goal['name']}, and 0 if it isn't. Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt

        if prompt_type == 'pivot':
            return f'{first_instruction} Use your prior knowledge about where items are typically located within a home. '
        
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: {first_instruction} use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see, and if you have any leads on finding the {goal['name']}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return baseline_prompt
        
        if prompt_type == 'action':
            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible.  Use your prior knowledge about where items are typically located within a home.\n\n"
                f"CURRENT SITUATION:\n"
                f"- There are {num_actions - 1} red arrows showing possible movements\n"
                f"- Each arrow is labeled with a number in a white circle\n"
                f"{' - NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS.' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}\n\n"
                "Let's solve this step by step:\n"
                "1. Current State Analysis:\n"
                "   - What objects and pathways do you see?\n"
                "   - Are there any landmarks from previous observations?\n\n"
                "2. Goal Analysis:\n"
                f"   - Based on the target and home layout knowledge, where is the {goal} likely to be?\n"
                f"   - Consider the objects adjacent to the target: {adjacent_info}. Prioritize actions that lead to these adjacent objects.\n\n"
                "3. Memory Integration:\n"
                f"   - {memory_context}\n"
                "   - How does the current view relate to previous observations?\n"
                "   - Are there any promising paths based on past exploration?\n\n"
                "4. Path Planning:\n"
                "   - Which direction is most likely to lead to the goal?\n"
                "   - Are there any areas we haven't explored yet?\n\n"
                "5. Action Selection:\n"
                "   Choose the best action and explain why. Which numbered arrow best serves your plan? Return your choice as {'action': <action_key>}. Note:\n"
                "   - You CANNOT GO THROUGH CLOSED DOORS\n"
                "   - You DO NOT NEED TO GO UP OR DOWN STAIRS\n"
                "   - If you see the target object, even partially, take time to confirm its identity\n"
                "   - Choose paths that lead to unexplored areas when possible\n"
                "   - Avoid repeatedly visiting the same areas"
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

    def _choose_action(self, obs):
        try:
            agent_state = obs['agent_state']
            goal = obs['goal']
            
            # 记忆构建
            current_node = self._build_memory_node(obs, agent_state)
            # 添加到语义森林
            self.semantic_forest.add_node(current_node)
            
            # 语义森林检索
            relevant_nodes = self.semantic_forest.retrieve(
                query=goal,
                current_pos=agent_state.position,
                top_k=5
            )
            
            # 获取与目标物体相邻的物体
            adjacent_prompt = (
                f"List the five objects that are most likely to be adjacent to "
                f"the target \"{goal['name']}\" and the most likely location of \"{goal['name']}\" in the room."
            )
            adjacent_response = self.actionVLM.call(obs['color_sensor'], adjacent_prompt)
            adjacent_objects = self._eval_response(adjacent_response)
            if not isinstance(adjacent_objects, list):
                adjacent_objects = []
            
            # 运行主要的动作选择逻辑
            if goal['mode'] == 'image':
                stopping_images = [obs['color_sensor'], goal['goal_image']]
            else:
                stopping_images = [obs['color_sensor']]

            a_final, images, step_metadata, stopping_response = self._run_threads(obs, stopping_images, goal)
            
            if goal['mode'] == 'image':
                images['goal_image'] = goal['goal_image']

            # 如果连续两次调用停止，则终止episode
            if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
                step_metadata['action_number'] = -1
                agent_action = PolarAction.stop
                logging_data = {}
            else:
                # 使用提示获取动作
                step_metadata, logging_data, _ = self._prompting(
                    goal, a_final, images, step_metadata, 
                    adjacent_objects, relevant_nodes
                )
                agent_action = self._action_number_to_polar(
                    step_metadata['action_number'], 
                    list(a_final)
                )

            metadata = {
                'step_metadata': step_metadata,
                'logging_data': logging_data,
                'a_final': a_final,
                'images': images,
                'relevant_nodes': relevant_nodes  # 添加相关节点到元数据
            }
            
            return agent_action, metadata
            
        except Exception as e:
            logging.error(f"Error in _choose_action: {e}")
            # 返回默认动作和元数据
            return PolarAction.default, {
                'step_metadata': {'success': 0, 'action_number': -1},
                'logging_data': {},
                'images': {'color_sensor': obs['color_sensor']},
                'a_final': []
            }

    def _initialize_vlms(self, cfg: dict):
        vlm_cls = globals()[cfg['model_cls']]
        system_instruction = (
            "You are an embodied robotic assistant, with an RGB image sensor. You observe the image and instructions "
            "given to you and output a textual response, which is converted into actions that physically move you "
            "within the environment. You cannot move through closed doors. "
        )
        self.actionVLM: VLM = vlm_cls(**cfg['model_kwargs'], system_instruction=system_instruction)
        self.stoppingVLM: VLM = vlm_cls(**cfg['model_kwargs'])

    def _run_threads(self, obs: dict, stopping_images: list[np.array], goal):
        """Concurrently runs the stopping thread to determine if the agent should stop, and the preprocessing thread to calculate potential actions."""
        with concurrent.futures.ThreadPoolExecutor() as executor:
            preprocessing_thread = executor.submit(self._preprocessing_module, obs)
            stopping_thread = executor.submit(self._stopping_module, stopping_images, goal)

            a_final, images = preprocessing_thread.result()
            called_stop, stopping_response = stopping_thread.result()
        
        if called_stop:
            logging.info('Model called stop')
            self.stopping_calls.append(self.step_ndx)
            # If the model calls stop, turn off navigability and explore bias tricks
            if self.cfg['navigability_mode'] != 'none' and self.cfg['project']:
                new_image = obs['color_sensor'].copy()
                a_final = self._project_onto_image(
                    self._get_default_arrows(), new_image, obs['agent_state'],
                    obs['agent_state'].sensor_states['color_sensor']
                )
                images['color_sensor'] = new_image

        step_metadata = {
            'action_number': -10,
            'success': 1,
            'pivot': 1 if self.pivot is not None else 0,
            'model': self.actionVLM.name,
            'agent_location': obs['agent_state'].position,
            'called_stopping': called_stop
        }
        return a_final, images, step_metadata, stopping_response

    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        if not self.cfg['project']:
            # Actions for the w/o proj baseline
            a_final = {
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            }
            return a_final, images

        if self.cfg['navigability_mode'] == 'none':
            a_final = [
                # Actions for the w/o nav baseline
                (self.cfg['max_action_dist'], -0.36 * np.pi),
                (self.cfg['max_action_dist'], -0.28 * np.pi),
                (self.cfg['max_action_dist'], 0),
                (self.cfg['max_action_dist'], 0.28 * np.pi),
                (self.cfg['max_action_dist'], 0.36 * np.pi)
            ]
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)

        a_final_projected = self._projection(a_final, images, agent_state)
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state)
        return a_final_projected, images

    def _stopping_module(self, stopping_images: list[np.array], goal):
        """Determines if the agent should stop."""
        stopping_prompt = self._construct_prompt(goal, 'stopping')
        stopping_response = self.stoppingVLM.call(stopping_images, stopping_prompt)
        dct = self._eval_response(stopping_response)
        if 'done' in dct and int(dct['done']) == 1:
            return True, stopping_response
        
        return False, stopping_response

    def _navigability(self, obs: dict):
        """Generates the set of navigability actions and updates the voxel map accordingly."""
        agent_state: habitat_sim.AgentState = obs['agent_state']
        sensor_state = agent_state.sensor_states['color_sensor']
        rgb_image = obs['color_sensor']
        depth_image = obs[f'depth_sensor']
        if self.cfg['navigability_mode'] == 'depth_estimate':
            depth_image = self.depth_estimator.call(rgb_image)
        if self.cfg['navigability_mode'] == 'segmentation':
            depth_image = None

        navigability_mask = self._get_navigability_mask(
            rgb_image, depth_image, agent_state, sensor_state
        )

        sensor_range =  np.deg2rad(self.fov / 2) * 1.5

        all_thetas = np.linspace(-sensor_range, sensor_range, self.cfg['num_theta'])
        start = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state,
            resolution=self.resolution, focal_length=self.focal_length
        )

        a_initial = []
        for theta_i in all_thetas:
            r_i, theta_i = self._get_radial_distance(start, theta_i, navigability_mask, agent_state, sensor_state, depth_image)
            if r_i is not None:
                self._update_voxel(
                    r_i, theta_i, agent_state,
                    clip_dist=self.cfg['max_action_dist'], clip_frac=self.e_i_scaling
                )
                a_initial.append((r_i, theta_i))

        return a_initial

    def _action_proposer(self, a_initial: list, agent_state: habitat_sim.AgentState):
        """Refines the initial set of actions, ensuring spacing and adding a bias towards exploration."""
        min_angle = self.fov/self.cfg['spacing_ratio']
        explore_bias = self.cfg['explore_bias']
        clip_frac = self.cfg['clip_frac']
        clip_mag = self.cfg['max_action_dist']

        explore = explore_bias > 0
        unique = {}
        for mag, theta in a_initial:
            if theta in unique:
                unique[theta].append(mag)
            else:
                unique[theta] = [mag]
        arrowData = []

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color
        for theta, mags in unique.items():
            # Reference the map to classify actions as explored or unexplored
            mag = min(mags)
            cart = [self.e_i_scaling*mag*np.sin(theta), 0, -self.e_i_scaling*mag*np.cos(theta)]
            global_coords = local_to_global(agent_state.position, agent_state.rotation, cart)
            grid_coords = self._global_to_grid(global_coords)
            score = (sum(np.all((topdown_map[grid_coords[1]-2:grid_coords[1]+2, grid_coords[0]] == self.explored_color), axis=-1)) + 
                    sum(np.all(topdown_map[grid_coords[1], grid_coords[0]-2:grid_coords[0]+2] == self.explored_color, axis=-1)))
            arrowData.append([clip_frac*mag, theta, score<3])

        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))

        filtered.sort(key=lambda x: x[1])
        if filtered == []:
            return []
        if explore:
            # Add unexplored actions with spacing, starting with the longest one
            f = list(filter(lambda x: x[2], filtered))
            if len(f) > 0:
                longest = max(f, key=lambda x: x[0])
                longest_theta = longest[1]
                smallest_theta = longest[1]
                longest_ndx = f.index(longest)
            
                out.append([min(longest[0], clip_mag), longest[1], longest[2]])
                thetas.add(longest[1])
                for i in range(longest_ndx+1, len(f)):
                    if f[i][1] - longest_theta > (min_angle*0.9):
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        longest_theta = f[i][1]
                for i in range(longest_ndx-1, -1, -1):
                    if smallest_theta - f[i][1] > (min_angle*0.9):
                        
                        out.append([min(f[i][0], clip_mag), f[i][1], f[i][2]])
                        thetas.add(f[i][1])
                        smallest_theta = f[i][1]

                for r_i, theta_i, e_i in filtered:
                    if theta_i not in thetas and min([abs(theta_i - t) for t in thetas]) > min_angle*explore_bias:
                        out.append((min(r_i, clip_mag), theta_i, e_i))
                        thetas.add(theta_i)

        if len(out) == 0:
            # if no explored actions or no explore bias
            longest = max(filtered, key=lambda x: x[0])
            longest_theta = longest[1]
            smallest_theta = longest[1]
            longest_ndx = filtered.index(longest)
            out.append([min(longest[0], clip_mag), longest[1], longest[2]])
            
            for i in range(longest_ndx+1, len(filtered)):
                if filtered[i][1] - longest_theta > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    longest_theta = filtered[i][1]
            for i in range(longest_ndx-1, -1, -1):
                if smallest_theta - filtered[i][1] > min_angle:
                    out.append([min(filtered[i][0], clip_mag), filtered[i][1], filtered[i][2]])
                    smallest_theta = filtered[i][1]


        if (out == [] or max(out, key=lambda x: x[0])[0] < self.cfg['min_action_dist']) and (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
            return self._get_default_arrows()
        
        out.sort(key=lambda x: x[1])
        return [(mag, theta) for mag, theta, _ in out]

    def _projection(self, a_final: list, images: dict, agent_state: habitat_sim.AgentState):
        """
        Projection component of VLMnav. Projects the arrows onto the image, annotating them with action numbers.
        Note actions that are too close together or too close to the boundaries of the image will not get projected.
        """
        a_final_projected = self._project_onto_image(
            a_final, images['color_sensor'], agent_state,
            agent_state.sensor_states['color_sensor']
        )

        if not a_final_projected and (self.step_ndx - self.turned < self.cfg['turn_around_cooldown']):
            logging.info('No actions projected and cannot turn around')
            a_final = self._get_default_arrows()
            a_final_projected = self._project_onto_image(
                a_final, images['color_sensor'], agent_state,
                agent_state.sensor_states['color_sensor']
            )

        return a_final_projected

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict, adjacent_objects: list, relevant_nodes):
        """
        Prompting component of VLMNav. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        # 将相邻物体信息添加到 action_prompt
        adjacent_info = ', '.join(adjacent_objects) if adjacent_objects else "no adjacent object information available"
        # action_prompt += f" The objects adjacent to the target are: {adjacent_info}."

        prompt_type = 'action' if self.cfg['project'] else 'no_project'

        # 生成多层级记忆上下文
        memory_context = self._generate_memory_context(relevant_nodes)  # relevant_nodes现在来自语义森林
        
        action_prompt = self._construct_prompt(goal, prompt_type, adjacent_info, num_actions=len(a_final), memory_context=memory_context)

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])
        
        # 添加检查
        for i, img in enumerate(prompt_images):
            if img is None:
                logging.error(f"Image {i} is None")
                continue
            logging.info(f"Image {i} shape: {img.shape}, dtype: {img.dtype}")
            if img.ndim != 3 or img.shape[2] != 3:
                logging.error(f"Unexpected image format: {img.shape}")
        
        response = self.actionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

        logging_data = {}
        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except (IndexError, KeyError, TypeError, ValueError) as e:
            logging.error(f'Error parsing response {e}')
            step_metadata['success'] = 0
        finally:
            logging_data['ACTION_NUMBER'] = step_metadata.get('action_number')
            logging_data['PROMPT'] = action_prompt
            logging_data['RESPONSE'] = response

        return step_metadata, logging_data, response

    def _get_navigability_mask(self, rgb_image: np.array, depth_image: np.array, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Get the navigability mask for the current state, according to the configured navigability mode.
        """
        if self.cfg['navigability_mode'] == 'segmentation':
            navigability_mask = self.segmentor.get_navigability_mask(rgb_image)
        else:
            thresh = 1 if self.cfg['navigability_mode'] == 'depth_estimate' else self.cfg['navigability_height_threshold']
            height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
            navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh

        return navigability_mask

    def _get_default_arrows(self):
        """
        Get the action options for when the agent calls stop the first time, or when no navigable actions are found.
        """
        angle = np.deg2rad(self.fov / 2) * 0.7
        
        default_actions = [
            (self.cfg['stopping_action_dist'], -angle),
            (self.cfg['stopping_action_dist'], -angle / 4),
            (self.cfg['stopping_action_dist'], angle / 4),
            (self.cfg['stopping_action_dist'], angle)
        ]
        
        default_actions.sort(key=lambda x: x[1])
        return default_actions

    def _get_radial_distance(self, start_pxl: tuple, theta_i: float, navigability_mask: np.ndarray, 
                             agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, 
                             depth_image: np.ndarray):
        """
        Calculates the distance r_i that the agent can move in the direction theta_i, according to the navigability mask.
        """
        agent_point = [2 * np.sin(theta_i), 0, -2 * np.cos(theta_i)]
        end_pxl = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_pxl is None or end_pxl[1] >= self.resolution[0]:
            return None, None

        H, W = navigability_mask.shape

        # Find intersections of the theoretical line with the image boundaries
        intersections = find_intersections(start_pxl[0], start_pxl[1], end_pxl[0], end_pxl[1], W, H)
        if intersections is None:
            return None, None

        (x1, y1), (x2, y2) = intersections
        num_points = max(abs(x2 - x1), abs(y2 - y1)) + 1
        x_coords = np.linspace(x1, x2, num_points)
        y_coords = np.linspace(y1, y2, num_points)

        out = (int(x_coords[-1]), int(y_coords[-1]))
        if not navigability_mask[int(y_coords[0]), int(x_coords[0])]:
            return 0, theta_i

        for i in range(num_points - 4):
            # Trace pixels until they are not navigable
            y = int(y_coords[i])
            x = int(x_coords[i])
            if sum([navigability_mask[int(y_coords[j]), int(x_coords[j])] for j in range(i, i + 4)]) <= 2:
                out = (x, y)
                break

        if i < 5:
            return 0, theta_i

        if self.cfg['navigability_mode'] == 'segmentation':
            #Simple estimation of distance based on number of pixels
            r_i = 0.0794 * np.exp(0.006590 * i) + 0.616

        else:
            #use depth to get distance
            out = (np.clip(out[0], 0, W - 1), np.clip(out[1], 0, H - 1))
            camera_coords = unproject_2d(
                *out, depth_image[out[1], out[0]], resolution=self.resolution, focal_length=self.focal_length
            )
            local_coords = global_to_local(
                agent_state.position, agent_state.rotation,
                local_to_global(sensor_state.position, sensor_state.rotation, camera_coords))
            r_i = np.linalg.norm([local_coords[0], local_coords[2]])  # 修复缩进

        return r_i, theta_i

    def _can_project(self, r_i: float, theta_i: float, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose):
        """
        Checks whether the specified polar action can be projected onto the image, i.e., not too close to the boundaries of the image.
        """
        agent_point = [r_i * np.sin(theta_i), 0, -r_i * np.cos(theta_i)]
        end_px = agent_frame_to_image_coords(
            agent_point, agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        if end_px is None:
            return None

        if (
            self.cfg['image_edge_threshold'] * self.resolution[1] <= end_px[0] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[1] and
            self.cfg['image_edge_threshold'] * self.resolution[0] <= end_px[1] <= (1 - self.cfg['image_edge_threshold']) * self.resolution[0]
        ):
            return end_px
        return None

    def _project_onto_image(self, a_final: list, rgb_image: np.ndarray, agent_state: habitat_sim.AgentState, sensor_state: habitat_sim.SixDOFPose, chosen_action: int=None):
        """
        Projects a set of actions onto a single image. Keeps track of action-to-number mapping.
        """
        scale_factor = rgb_image.shape[0] / 1080
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = BLACK
        circle_color = WHITE
        projected = {}
        if chosen_action == -1:
            put_text_on_image(
                rgb_image, 'TERMINATING EPISODE', text_color=GREEN, text_size=4 * scale_factor,
                location='center', text_thickness=math.ceil(3 * scale_factor), highlight=False
            )
            return projected

        start_px = agent_frame_to_image_coords(
            [0, 0, 0], agent_state, sensor_state, 
            resolution=self.resolution, focal_length=self.focal_length
        )
        for _, (r_i, theta_i) in enumerate(a_final):
            text_size = 2.4 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)

            end_px = self._can_project(r_i, theta_i, agent_state, sensor_state)
            if end_px is not None:
                action_name = len(projected) + 1
                projected[(r_i, theta_i)] = action_name

                cv2.arrowedLine(rgb_image, tuple(start_px), tuple(end_px), RED, math.ceil(5 * scale_factor), tipLength=0.0)
                text = str(action_name)
                (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
                circle_center = (end_px[0], end_px[1])
                circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

                if chosen_action is not None and action_name == chosen_action:
                    cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
                else:
                    cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
                cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
                text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
                cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)

        if (self.step_ndx - self.turned) >= self.cfg['turn_around_cooldown'] or self.step_ndx == self.turned or (chosen_action == 0):
            text = '0'
            text_size = 3.1 * scale_factor
            text_thickness = math.ceil(3 * scale_factor)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (math.ceil(0.05 * rgb_image.shape[1]), math.ceil(rgb_image.shape[0] / 2))
            circle_radius = max(text_width, text_height) // 2 + math.ceil(15 * scale_factor)

            if chosen_action is not None and chosen_action == 0:
                cv2.circle(rgb_image, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(rgb_image, circle_center, circle_radius, circle_color, -1)
            cv2.circle(rgb_image, circle_center, circle_radius, RED, math.ceil(2 * scale_factor))
            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.putText(rgb_image, text, text_position, font, text_size, text_color, text_thickness)
            cv2.putText(rgb_image, 'TURN AROUND', (text_position[0] // 2, text_position[1] + math.ceil(80 * scale_factor)), font, text_size * 0.75, RED, text_thickness)

        return projected


    def _update_voxel(self, r: float, theta: float, agent_state: habitat_sim.AgentState, clip_dist: float, clip_frac: float):
        """Update the voxel map to mark actions as explored or unexplored"""
        agent_coords = self._global_to_grid(agent_state.position)

        # Mark unexplored regions
        unclipped = max(r - 0.5, 0)
        local_coords = np.array([unclipped * np.sin(theta), 0, -unclipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.voxel_map, agent_coords, point, self.unexplored_color, self.voxel_ray_size)

        # Mark explored regions
        clipped = min(clip_frac * r, clip_dist)
        local_coords = np.array([clipped * np.sin(theta), 0, -clipped * np.cos(theta)])
        global_coords = local_to_global(agent_state.position, agent_state.rotation, local_coords)
        point = self._global_to_grid(global_coords)
        cv2.line(self.explored_map, agent_coords, point, self.explored_color, self.voxel_ray_size)

    def _global_to_grid(self, position: np.ndarray, rotation=None):
        """Convert global coordinates to grid coordinates in the agent's voxel map"""
        dx = position[0] - self.init_pos[0]
        dz = position[2] - self.init_pos[2]
        resolution = self.voxel_map.shape
        x = int(resolution[1] // 2 + dx * self.scale)
        y = int(resolution[0] // 2 + dz * self.scale)

        if rotation is not None:
            original_coords = np.array([x, y, 1])
            new_coords = np.dot(rotation, original_coords)
            new_x, new_y = new_coords[0], new_coords[1]
            return (int(new_x), int(new_y))

        return (x, y)

    def _generate_voxel(self, a_final: dict, zoom: int=9, agent_state: habitat_sim.AgentState=None, chosen_action: int=None):
        """For visualization purposes, add the agent's position and actions onto the voxel map"""
        agent_coords = self._global_to_grid(agent_state.position)
        right = (agent_state.position[0] + zoom, 0, agent_state.position[2])
        right_coords = self._global_to_grid(right)
        delta = abs(agent_coords[0] - right_coords[0])

        topdown_map = self.voxel_map.copy()
        mask = np.all(self.explored_map == self.explored_color, axis=-1)
        topdown_map[mask] = self.explored_color

        text_size = 1.25
        text_thickness = 1
        rotation_matrix = None
        agent_coords = self._global_to_grid(agent_state.position, rotation=rotation_matrix)
        x, y = agent_coords
        font = cv2.FONT_HERSHEY_SIMPLEX

        if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown']:
            a_final[(0.75, np.pi)] = 0

        for (r, theta), action in a_final.items():
            local_pt = np.array([r * np.sin(theta), 0, -r * np.cos(theta)])
            global_pt = local_to_global(agent_state.position, agent_state.rotation, local_pt)
            act_coords = self._global_to_grid(global_pt, rotation=rotation_matrix)

            # Draw action arrows and labels
            cv2.arrowedLine(topdown_map, tuple(agent_coords), tuple(act_coords), RED, 5, tipLength=0.05)
            text = str(action)
            (text_width, text_height), _ = cv2.getTextSize(text, font, text_size, text_thickness)
            circle_center = (act_coords[0], act_coords[1])
            circle_radius = max(text_width, text_height) // 2 + 15

            if chosen_action is not None and action == chosen_action:
                cv2.circle(topdown_map, circle_center, circle_radius, GREEN, -1)
            else:
                cv2.circle(topdown_map, circle_center, circle_radius, WHITE, -1)

            text_position = (circle_center[0] - text_width // 2, circle_center[1] + text_height // 2)
            cv2.circle(topdown_map, circle_center, circle_radius, RED, 1)
            cv2.putText(topdown_map, text, text_position, font, text_size, RED, text_thickness + 1)

        # Draw agent's current position
        cv2.circle(topdown_map, agent_coords, radius=15, color=RED, thickness=-1)

        # Zoom the map
        max_x, max_y = topdown_map.shape[1], topdown_map.shape[0]
        x1 = max(0, x - delta)
        x2 = min(max_x, x + delta)
        y1 = max(0, y - delta)
        y2 = min(max_y, y + delta)

        zoomed_map = topdown_map[y1:y2, x1:x2]
        return zoomed_map

    def _action_number_to_polar(self, action_number: int, a_final: list):
        """Converts the chosen action number to its PolarAction instance"""
        try:
            action_number = int(action_number)
            if action_number <= len(a_final) and action_number > 0:
                r, theta = a_final[action_number - 1]
                return PolarAction(r, -theta)
            if action_number == 0:
                return PolarAction(0, np.pi)
        except ValueError:
            pass

        logging.info("Bad action number: " + str(action_number))
        return PolarAction.default

    def _eval_response(self, response: str):
        """Converts the VLM response string into a dictionary, if possible"""
        try:
            eval_resp = ast.literal_eval(response[response.rindex('{'):response.rindex('}') + 1])
            if isinstance(eval_resp, dict):
                return eval_resp
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            logging.error(f'Error parsing response {response}')
            return {}

    def _generate_memory_context(self, ranked_nodes):
        """Generate hierarchical memory description"""
        if not ranked_nodes:
            return "No relevant memory context available."
        
        context = ["Memory Context (from macro to micro):"]
        try:
            for i, (node, score) in enumerate(ranked_nodes):
                level = self._get_node_level(node)
                context.append(
                    f"{i+1}. [Lv{level}] Semantic similarity: {score:.2f}\n"
                    f"   Position: {node['position']}\n"
                    f"   Description: {node.get('caption', 'Area contains multiple sub-nodes')}"
                )
            return "\n".join(context)
        except Exception as e:
            logging.error(f"Error generating memory context: {e}")
            return "Error generating memory context."


class GOATAgent(VLMNavAgent):
 
    def _choose_action(self, obs: dict):
        agent_state = obs['agent_state']
        goal = obs['goal']

        if goal['mode'] == 'image':
            stopping_images = [obs['color_sensor'], goal['goal_image']]
        else:
            stopping_images = [obs['color_sensor']]

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, stopping_images, goal)
        if goal['mode'] == 'image':
            images['goal_image'] = goal['goal_image']

        step_metadata.update({
            'goal': goal['name'],
            'goal_mode': goal['mode']
        })

        # If model calls stop two times in a row, we return the stop action and terminate the episode
        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {}
        else:
            if self.pivot is not None:
                pivot_instruction = self._construct_prompt(goal, 'pivot')
                agent_action, pivot_images = self.pivot.run(
                    obs['color_sensor'], pivot_instruction,
                    agent_state, agent_state.sensor_states['color_sensor'],
                    goal_image=goal['goal_image'] if goal['mode'] == 'image' else None
                )
                images.update(pivot_images)
                logging_data = {}
                step_metadata['action_number'] = -100
            else:
                # 获取相邻物体信息
                adjacent_prompt = (
                    f"List the five objects that are most likely to be adjacent to "
                    f"the target \"{goal['name']}\" and the most likely location of \"{goal['name']}\" in the room."
                )
                adjacent_response = self.actionVLM.call(obs['color_sensor'], adjacent_prompt)
                adjacent_objects = self._eval_response(adjacent_response)
                if not isinstance(adjacent_objects, list):
                    adjacent_objects = []

                # 获取相关节点（这里使用空列表，因为 GOAT 代理不使用记忆系统）
                relevant_nodes = []
                
                step_metadata, logging_data, _ = self._prompting(
                    goal, a_final, images, step_metadata, 
                    adjacent_objects, relevant_nodes
                )
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images
        }
        return agent_action, metadata

    def _construct_prompt(self, goal: dict, prompt_type: str, adjacent_info="", num_actions=0):
        """Constructs the prompt, depending on the goal modality. """
        if goal['mode'] == 'object':
            task = f'Navigate to the nearest {goal["name"]}'
            first_instruction = f'Find the nearest {goal["name"]} and navigate as close as you can to it. '
        if goal['mode'] == 'description':
            first_instruction = f"Find and navigate to the {goal['lang_desc']}. Navigate as close as you can to it. "
            task = first_instruction
        if goal['mode'] == 'image':
            task = f'Navigate to the specific {goal["name"]} shown in the image labeled GOAL IMAGE. Pay close attention to the details, and note you may see the object from a different angle than in the goal image. Navigate as close as you can to it '
            first_instruction = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate as close as you can to it. "

        if prompt_type == 'stopping':        
            stopping_prompt = (f"The agent has has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
            f'Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note a chair is NOT sofa which is NOT a bed. '
            f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return baseline_prompt
        if prompt_type == 'pivot':
            pivot_prompt = f"NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            return pivot_prompt
        if prompt_type == 'action':
            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible.  Use your prior knowledge about where items are typically located within a home.\n\n"
                f"CURRENT SITUATION:\n"
                f"- There are {num_actions - 1} red arrows showing possible movements\n"
                f"- Each arrow is labeled with a number in a white circle\n"
                f"{' - NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS.' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}\n\n"
                "Let's solve this step by step:\n"
                "1. Current State Analysis:\n"
                "   - What objects and pathways do you see?\n"
                "   - Are there any landmarks from previous observations?\n\n"
                "2. Goal Analysis:\n"
                f"   - Based on the target and home layout knowledge, where is the {goal} likely to be?\n"
                f"   - Consider the objects adjacent to the target: {adjacent_info}. Prioritize actions that lead to these adjacent objects.\n\n"
                "3. Memory Integration:\n"
                f"   - {memory_context}\n"
                "   - How does the current view relate to previous observations?\n"
                "   - Are there any promising paths based on past exploration?\n\n"
                "4. Path Planning:\n"
                "   - Which direction is most likely to lead to the goal?\n"
                "   - Are there any areas we haven't explored yet?\n\n"
                "5. Action Selection:\n"
                "   Choose the best action and explain why. Which numbered arrow best serves your plan? Return your choice as {'action': <action_key>}. Note:\n"
                "   - You CANNOT GO THROUGH CLOSED DOORS\n"
                "   - You DO NOT NEED TO GO UP OR DOWN STAIRS\n"
                "   - If you see the target object, even partially, take time to confirm its identity\n"
                "   - Choose paths that lead to unexplored areas when possible\n"
                "   - Avoid repeatedly visiting the same areas"
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

    def reset_goal(self):
        """Called after every subtask of GOAT. Notably does not reset the voxel map, only resets all areas to be unexplored"""
        self.stopping_calls = [self.step_ndx-2]
        self.explored_map = np.zeros_like(self.explored_map)
        self.turned = self.step_ndx - self.cfg['turn_around_cooldown']

class SemanticForest:
    def __init__(self, text_encoder=None, embedding_dim=512):
        self.text_encoder = text_encoder
        self.embedding_dim = embedding_dim
        self.leaf_index = faiss.IndexFlatIP(embedding_dim)
        self.nodes = []
        self.levels = []
        
    def add_node(self, node_data):
        """增量添加节点并更新层次结构"""
        try:
            # 检查节点是否包含必要字段
            if 'embed' not in node_data and 'image_embed' in node_data:
                node_data['embed'] = node_data['image_embed']  # 确保统一字段名
                
            # 确保嵌入向量格式正确
            embed = node_data['embed'].astype(np.float32)
            if embed.ndim == 1:
                embed = embed.reshape(1, -1)
                
            # 添加到Faiss索引
            self.leaf_index.add(embed)
            self.nodes.append(node_data)
            
            # 每积累一定数量的节点更新层次结构
            if len(self.nodes) % 50 == 0:
                self._update_hierarchy()
                
        except Exception as e:
            logging.error(f"Error adding node to semantic forest: {e}")

    def retrieve(self, query, current_pos, top_k=5):
        """
        检索相关节点，支持文本查询或嵌入查询
        
        Args:
            query: 文本字符串、字典或嵌入向量
            current_pos: 当前位置坐标
            top_k: 返回最相关节点数量
        """
        try:
            # 处理不同类型的查询
            if isinstance(query, np.ndarray):
                query_embed = query
            elif isinstance(query, dict) and 'name' in query:
                if self.text_encoder:
                    query_embed = self.text_encoder.encode(query['name'])
                else:
                    logging.warning("No text encoder available, using random embedding")
                    query_embed = np.random.rand(self.embedding_dim)
            else:
                if self.text_encoder:
                    query_embed = self.text_encoder.encode(str(query))
                else:
                    logging.warning("No text encoder available, using random embedding")
                    query_embed = np.random.rand(self.embedding_dim)
            
            # 确保查询向量格式正确
            query_embed = query_embed.astype(np.float32)
            if query_embed.ndim == 1:
                query_embed = query_embed.reshape(1, -1)
            
            # 检查索引是否为空
            if len(self.nodes) == 0:
                return []
            
            # Faiss检索
            D, I = self.leaf_index.search(query_embed, min(top_k * 2, len(self.nodes)))
            
            # 结合空间距离重排序
            candidates = []
            for idx in I[0]:
                node = self.nodes[idx]
                semantic_score = float(D[0][list(I[0]).index(idx)])
                spatial_score = 1.0 / (1.0 + np.linalg.norm(node['position'] - current_pos))
                
                # 混合得分
                final_score = 0.7 * semantic_score + 0.3 * spatial_score
                candidates.append((node, final_score))
            
            # 按最终得分排序并返回top_k结果
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[:top_k]
            
        except Exception as e:
            logging.error(f"Error during retrieval: {e}")
            return []

    def _update_hierarchy(self):
        """增量层次聚类"""
        try:
            # 获取最新的节点
            new_nodes = self.nodes[-50:]
            
            # 提取位置和嵌入向量
            positions = np.array([n['position'] for n in new_nodes])
            embeds = np.array([n.get('embed', n.get('image_embed')) for n in new_nodes])  # 兼容两种字段名
            
            # 计算混合相似度
            spatial_sim = 1.0 / (1.0 + pairwise_distances(positions))
            semantic_sim = cosine_similarity(embeds)
            hybrid_sim = 0.4 * spatial_sim + 0.6 * semantic_sim
            
            # 执行层次聚类
            Z = linkage(1.0 - hybrid_sim, method='complete')
            clusters = fcluster(Z, t=0.7, criterion='distance')
            
            # 构建新层级
            new_level = []
            for c in set(clusters):
                cluster_nodes = [new_nodes[i] for i in np.where(clusters == c)[0]]
                new_level.append({
                    'children': cluster_nodes,
                    'position': np.mean([n['position'] for n in cluster_nodes], axis=0),
                    'embed': np.mean([n.get('embed', n.get('image_embed')) for n in cluster_nodes], axis=0)
                })
            
            # 更新层级结构
            if not self.levels:
                self.levels.append(new_level)
            else:
                self.levels[-1].extend(new_level)
                
        except Exception as e:
            logging.error(f"Error updating hierarchy: {e}")

    def _get_node_level(self, node):
        """获取节点在层次结构中的级别"""
        try:
            # 先检查是否为叶节点
            if node in self.nodes:
                return 0
            
            # 检查是否为中间节点
            for level_idx, level in enumerate(self.levels):
                for parent_node in level:
                    if 'children' in parent_node and node in parent_node['children']:
                        return level_idx + 1
            return -1
        except Exception as e:
            logging.error(f"Error getting node level: {e}")
            return -1

class ObjectNavAgent(VLMNavAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.semantic_forest = SemanticForest()
        self.topological_map = {}
        # 在初始化时加载模型
        try:
            self.text_encoder = SentenceTransformer('clip-ViT-B-32')
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer: {e}")
            self.text_encoder = None
            
        self.semantic_forest = SemanticForest(text_encoder=self.text_encoder)
        self.topological_map = {}
        self.memory_graph = {}  # 添加memory_graph初始化

    def _build_memory_node(self, obs, agent_state):
        """增强的记忆节点构建"""
        try:
            image = obs['color_sensor']
            caption = self._generate_caption(image)
            
            if self.text_encoder is None:
                return {
                    'position': agent_state.position,
                    'caption': caption
                }
                
            # 使用numpy数组存储嵌入向量以节省内存
            image_embed = np.array(self.text_encoder.encode(image), dtype=np.float32)
            text_embed = np.array(self.text_encoder.encode(caption), dtype=np.float32)
            
            return {
                'position': agent_state.position,
                'image_embed': image_embed,
                'text_embed': text_embed,
                'embed': text_embed,  # 添加标准字段名
                'caption': caption,
                'timestamp': self.step_ndx
            }
            
        except Exception as e:
            logging.error(f"Error building memory node: {e}")
            # 返回基础节点信息
            return {
                'position': agent_state.position,
                'caption': caption if 'caption' in locals() else None
            }
    
    def _retrieve_relevant_nodes(self, current_node, goal):
        """统一的记忆节点检索方法"""
        try:
            # 使用语义森林进行检索
            relevant_nodes = self.semantic_forest.retrieve(
                query=goal,
                current_pos=current_node['position'],
                top_k=5
            )
            
            # 如果语义森林检索失败，回退到基于距离的简单检索
            if not relevant_nodes and hasattr(self, 'memory_graph') and self.memory_graph:
                return self._distance_based_retrieval(current_node, goal)
                
            return relevant_nodes
            
        except Exception as e:
            logging.error(f"Error retrieving relevant nodes: {e}")
            return []

    def _distance_based_retrieval(self, current_node, goal):
        """基于距离的简单检索方法"""
        relevant_nodes = []
        try:
            for node in self.memory_graph.values():
                if np.linalg.norm(current_node['position'] - node['position']) < 5.0:
                    relevant_nodes.append((node, 1.0 / (1.0 + np.linalg.norm(current_node['position'] - node['position']))))
                
            # 按距离排序
            relevant_nodes.sort(key=lambda x: x[1], reverse=True)
            return relevant_nodes[:5]
                
        except Exception as e:
            logging.error(f"Error in distance based retrieval: {e}")
            
        return []
    
    def _generate_caption(self, image):
        """使用VLM生成场景描述"""
        caption_prompt = ("You are a robot equipped with cameras. "
                        "Given the image, describe the objects you see in a single list, "
                        "and then describe their spatial relationships.")
        return self.actionVLM.call(image, caption_prompt)

    def _visualize_memory_forest(self, current_node, relevant_nodes=None):
        """可视化记忆森林和当前位置"""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Circle, FancyArrowPatch
            import matplotlib
            matplotlib.use('Agg')  # 设置为非交互式后端
            
            # 创建图形
            plt.figure(figsize=(10, 10))
            ax = plt.gca()
            
            # 绘制所有节点
            positions = []
            for node in self.semantic_forest.nodes:
                if 'position' in node:
                    pos = node['position']
                    positions.append((pos[0], pos[2]))  # 只使用x和z坐标
                    
            if positions:
                positions = np.array(positions)
                plt.scatter(positions[:, 0], positions[:, 1], c='gray', alpha=0.5, s=10)
            
            # 高亮当前节点
            if 'position' in current_node:
                curr_pos = current_node['position']
                plt.scatter(curr_pos[0], curr_pos[2], c='blue', s=100, marker='*')
                circle = Circle((curr_pos[0], curr_pos[2]), 1.5, fill=False, color='blue')
                ax.add_patch(circle)
            
            # 高亮相关节点
            if relevant_nodes:
                for node, score in relevant_nodes:
                    if 'position' in node:
                        pos = node['position']
                        plt.scatter(pos[0], pos[2], c='red', s=50*score, alpha=min(score, 1.0))
                        # 从当前节点到相关节点的箭头
                        if 'position' in current_node:
                            arrow = FancyArrowPatch((current_node['position'][0], current_node['position'][2]), 
                                                (pos[0], pos[2]),
                                                arrowstyle='-|>', color='green', alpha=min(score, 1.0))
                            ax.add_patch(arrow)
            
            # 添加标题等信息
            plt.title(f"Memory Forest (Step {self.step_ndx})")
            plt.xlabel("X Position")
            plt.ylabel("Z Position")
            plt.grid(True)
            plt.tight_layout()
            
            # 转换为numpy数组
            fig = plt.gcf()
            fig.canvas.draw()
            img = np.array(fig.canvas.renderer.buffer_rgba())
            plt.close()
            
            return img
        except Exception as e:
            logging.error(f"Error visualizing memory forest: {e}")
            return None

    def _visualize_retrievals(self, current_node, relevant_nodes, goal):
        """可视化检索到的记忆节点"""
        try:
            # 创建空白图像
            from PIL import Image, ImageDraw, ImageFont
            
            width, height = 800, 600
            image = np.ones((height, width, 3), dtype=np.uint8) * 255
            
            # 转换为PIL Image进行文字处理
            pil_img = Image.fromarray(image)
            draw = ImageDraw.Draw(pil_img)
            
            # 使用默认字体
            try:
                font = ImageFont.truetype("arial.ttf", 14)
            except:
                font = ImageFont.load_default()
            
            # 添加标题
            goal_name = goal['name'] if isinstance(goal, dict) and 'name' in goal else str(goal)
            title = f"Memory Retrieval for: {goal_name}"
            draw.text((20, 20), title, fill=(0, 0, 0), font=font)
            
            # 添加当前步骤信息
            step_info = f"Current Step: {self.step_ndx}"
            draw.text((20, 50), step_info, fill=(0, 0, 0), font=font)
            
            # 添加相关节点信息
            if not relevant_nodes or len(relevant_nodes) == 0:
                draw.text((20, 90), "No relevant nodes found", fill=(255, 0, 0), font=font)
            else:
                draw.text((20, 90), f"Top {len(relevant_nodes)} relevant nodes:", fill=(0, 0, 0), font=font)
                
                y_offset = 120
                for i, (node, score) in enumerate(relevant_nodes):
                    level = self._get_node_level(node)
                    position = node.get('position', [0, 0, 0])
                    caption = node.get('caption', 'No description available')
                    
                    # 限制描述长度
                    if len(caption) > 100:
                        caption = caption[:97] + "..."
                    
                    node_info = [
                        f"Node {i+1} [Level {level}] - Score: {score:.2f}",
                        f"Position: ({position[0]:.1f}, {position[1]:.1f}, {position[2]:.1f})",
                        f"Description: {caption}"
                    ]
                    
                    # 绘制节点信息框
                    node_color = (220, 240, 255)  # 浅蓝色背景
                    rect_height = 80
                    draw.rectangle([(15, y_offset-5), (width-15, y_offset+rect_height)], 
                                fill=node_color, outline=(0, 0, 0))
                    
                    # 添加节点信息
                    for j, line in enumerate(node_info):
                        draw.text((25, y_offset + j*25), line, fill=(0, 0, 0), font=font)
                    
                    y_offset += rect_height + 15
            
            # 将PIL图像转回numpy数组
            return np.array(pil_img)
        except Exception as e:
            logging.error(f"Error visualizing retrievals: {e}")
            return None

    def _get_node_level(self, node):
        """获取节点在层次结构中的级别"""
        return self.semantic_forest._get_node_level(node)

    def step(self, obs: dict):
        agent_state: habitat_sim.AgentState = obs['agent_state']
        if self.step_ndx == 0:
            self.init_pos = agent_state.position

        # 构建当前节点（移到这里）
        current_node = self._build_memory_node(obs, agent_state)

        agent_action, metadata = self._choose_action(obs)
        metadata['step_metadata'].update(self.cfg)

        if metadata['step_metadata']['action_number'] == 0:
            self.turned = self.step_ndx

        # Visualize the chosen action
        chosen_action_image = obs['color_sensor'].copy()
        self._project_onto_image(
            metadata['a_final'], chosen_action_image, agent_state,
            agent_state.sensor_states['color_sensor'], 
            chosen_action=metadata['step_metadata']['action_number']
        )
        metadata['images']['color_sensor_chosen'] = chosen_action_image

        # Add Embodied-RAG visualizations
        try:
            # 创建记忆森林可视化
            memory_forest_img = self._visualize_memory_forest(current_node, metadata.get('relevant_nodes', []))
            if memory_forest_img is not None:
                if len(memory_forest_img.shape) == 3 and memory_forest_img.shape[2] == 4:
                    memory_forest_img = memory_forest_img[:, :, :3]
                metadata['images']['memory_forest'] = memory_forest_img
                
            # 创建检索结果可视化
            retrieval_img = self._visualize_retrievals(current_node, metadata.get('relevant_nodes', []), obs['goal'])
            if retrieval_img is not None:
                if len(retrieval_img.shape) == 3 and retrieval_img.shape[2] == 4:
                    retrieval_img = retrieval_img[:, :, :3]
                metadata['images']['retrievals'] = retrieval_img
        except Exception as e:
            logging.error(f"Step {self.step_ndx}: Visualization error: {e}")

        self.step_ndx += 1
        return agent_action, metadata

    def _choose_action(self, obs: dict):
        try:
            # 构建当前节点
            current_node = self._build_memory_node(obs, obs['agent_state'])
            
            # 调用父类的 _choose_action 方法
            agent_action, metadata = super()._choose_action(obs)
            
            # 获取相关节点
            relevant_nodes = metadata.get('relevant_nodes', [])
            
            # 添加 Embodied-RAG 可视化
            # 创建记忆森林可视化
            memory_forest_img = self._visualize_memory_forest(current_node, relevant_nodes)
            if memory_forest_img is not None:
                if len(memory_forest_img.shape) == 3 and memory_forest_img.shape[2] == 4:
                    memory_forest_img = memory_forest_img[:, :, :3]
                metadata['images']['memory_forest'] = memory_forest_img
                
            # 创建检索结果可视化
            retrieval_img = self._visualize_retrievals(current_node, relevant_nodes, obs['goal'])
            if retrieval_img is not None:
                if len(retrieval_img.shape) == 3 and retrieval_img.shape[2] == 4:
                    retrieval_img = retrieval_img[:, :, :3]
                metadata['images']['retrievals'] = retrieval_img
            
            return agent_action, metadata
            
        except Exception as e:
            logging.error(f"Step {self.step_ndx}: Error in _choose_action: {e}")
            return super()._choose_action(obs)  # 如果出错，回退到父类的实现

    def _construct_prompt(self, goal: str, prompt_type:str, adjacent_info="", num_actions: int=0, memory_context: str = ""):
        if goal['mode'] == 'object':
            task = f'Navigate to the nearest {goal["name"]}'
            first_instruction = f'Find the nearest {goal["name"]} and navigate as close as you can to it. '
        if goal['mode'] == 'description':
            first_instruction = f"Find and navigate to the {goal['lang_desc']}. Navigate as close as you can to it. "
            task = first_instruction
        if goal['mode'] == 'image':
            task = f'Navigate to the specific {goal["name"]} shown in the image labeled GOAL IMAGE. Pay close attention to the details, and note you may see the object from a different angle than in the goal image. Navigate as close as you can to it '
            first_instruction = f"Observe the image labeled GOAL IMAGE. Find this specific {goal['name']} shown in the image and navigate as close as you can to it. "

        if prompt_type == 'stopping':
            stopping_prompt = (f"The agent has has been tasked with navigating to a {goal.upper()}. The agent has sent you an image taken from its current location. "
            f'Your job is to determine whether the agent is VERY CLOSE to a {goal}. Note a chair is NOT sofa which is NOT a bed. '
            f"First, tell me what you see in the image, and tell me if there is a {goal}. Second, return 1 if the agent is VERY CLOSE to the {goal} - make sure the object you see is ACTUALLY a {goal}, Return 0 if if there is no {goal}, or if it is far away, or if you are not sure. Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal.upper()}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return baseline_prompt
        if prompt_type == 'pivot':
            pivot_prompt = f"NAVIGATE TO THE NEAREST {goal.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            return pivot_prompt
        if prompt_type == 'action':
            
            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal.upper()}, and get as close to it as possible.  Use your prior knowledge about where items are typically located within a home.\n\n"
                f"CURRENT SITUATION:\n"
                f"- There are {num_actions - 1} red arrows showing possible movements\n"
                f"- Each arrow is labeled with a number in a white circle\n"
                f"{' - NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS.' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}\n\n"
                "Let's solve this step by step:\n"
                "1. Current State Analysis:\n"
                "   - What objects and pathways do you see?\n"
                "   - Are there any landmarks from previous observations?\n\n"
                "2. Goal Analysis:\n"
                f"   - Based on the target and home layout knowledge, where is the {goal} likely to be?\n"
                f"   - Consider the objects adjacent to the target: {adjacent_info}. Prioritize actions that lead to these adjacent objects.\n\n"
                "3. Memory Integration:\n"
                f"   - {memory_context}\n"
                "   - How does the current view relate to previous observations?\n"
                "   - Are there any promising paths based on past exploration?\n\n"
                "4. Path Planning:\n"
                "   - Which direction is most likely to lead to the goal?\n"
                "   - Are there any areas we haven't explored yet?\n\n"
                "5. Action Selection:\n"
                "   Choose the best action and explain why. Which numbered arrow best serves your plan? Return your choice as {'action': <action_key>}. Note:\n"
                "   - You CANNOT GO THROUGH CLOSED DOORS\n"
                "   - You DO NOT NEED TO GO UP OR DOWN STAIRS\n"
                "   - If you see the target object, even partially, take time to confirm its identity\n"
                "   - Choose paths that lead to unexplored areas when possible\n"
                "   - Avoid repeatedly visiting the same areas"
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

    def _generate_memory_context(self, ranked_nodes):
        """Generate hierarchical memory description"""
        if not ranked_nodes:
            return "No relevant memory context available."
        
        context = ["Memory Context (from macro to micro):"]
        try:
            for i, (node, score) in enumerate(ranked_nodes):
                level = self._get_node_level(node)
                context.append(
                    f"{i+1}. [Lv{level}] Semantic similarity: {score:.2f}\n"
                    f"   Position: {node['position']}\n"
                    f"   Description: {node.get('caption', 'Area contains multiple sub-nodes')}"
                )
            return "\n".join(context)
        except Exception as e:
            logging.error(f"Error generating memory context: {e}")
            return "Error generating memory context."

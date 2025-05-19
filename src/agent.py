import logging
import math
import random
import ast
import concurrent.futures

import numpy as np
import re
import cv2

import habitat_sim

from simWrapper import PolarAction
from utils import *
from vlm import *
from pivot import PIVOT
from sentence_transformers import SentenceTransformer, util 
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial import distance
import time

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

class TopologicalNode:
    def __init__(self, position, rotation, timestamp, image=None, caption=None):
        self.position = np.array(position)
        self.rotation = rotation
        self.timestamp = timestamp
        self.image = image
        self.caption = caption
        self.connections = [] 
        self.embedding = None

class TopologicalMap:
    def __init__(self, distance_threshold=1.0):
        self.nodes = []
        self.distance_threshold = distance_threshold
        self.last_node = None
    
    def add_node(self, position, rotation, timestamp, image=None, caption=None):
        new_node = TopologicalNode(position, rotation, timestamp, image, caption)
        
        self.nodes.append(new_node)

        if self.last_node:
            dist = np.linalg.norm(new_node.position - self.last_node.position)
            if dist < self.distance_threshold:
                self.last_node.connections.append(len(self.nodes) - 1)
                new_node.connections.append(len(self.nodes) - 2)
        
        self.last_node = new_node
        return len(self.nodes) - 1  

    def get_node(self, index):
        if 0 <= index < len(self.nodes):
            return self.nodes[index]
        return None

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

class AreaOverlapTracker:
    
    def __init__(self, grid_resolution=0.1, vision_radius=0.1, map_size=1000):
        
        self.grid_resolution = grid_resolution  
        self.vision_radius = vision_radius    
        self.map_size = map_size              
        
        self.coverage_map = np.zeros((map_size, map_size), dtype=np.int32)
        self.total_observed_cells = 0  
        self.repeat_observed_cells = 0
        self.positions = [] 
        self.origin = None 
        self.aori_history = [] 
        self.step_ndx = 0
    
    def reset(self):
        self.coverage_map = np.zeros((self.map_size, self.map_size), dtype=np.int32)
        self.total_observed_cells = 0
        self.repeat_observed_cells = 0
        self.positions = []
        self.origin = None
        self.aori_history = []
        self.step_ndx = 0
    
    def update(self, position):
        
        self.positions.append(position)
        self.step_ndx += 1
        
        if self.origin is None:
            self.origin = np.array([position[0], position[2]])
        
        grid_x, grid_y = self._world_to_grid(position[0], position[2])
        
        vision_radius_grid = int(self.vision_radius / self.grid_resolution)
        
        y, x = np.ogrid[-vision_radius_grid:vision_radius_grid+1, -vision_radius_grid:vision_radius_grid+1]
        mask = x**2 + y**2 <= vision_radius_grid**2
        
        x_min = max(0, grid_x - vision_radius_grid)
        y_min = max(0, grid_y - vision_radius_grid)
        x_max = min(self.map_size, grid_x + vision_radius_grid + 1)
        y_max = min(self.map_size, grid_y + vision_radius_grid + 1)
        mask_x_offset = max(0, vision_radius_grid - grid_x)
        mask_y_offset = max(0, vision_radius_grid - grid_y)
        mask_width = x_max - x_min
        mask_height = y_max - y_min
        
        submask = mask[
            mask_y_offset:mask_y_offset+mask_height,
            mask_x_offset:mask_x_offset+mask_width
        ]
        
        old_coverage = (self.coverage_map[y_min:y_max, x_min:x_max] > 0).astype(np.int32)
        old_repeat = (self.coverage_map[y_min:y_max, x_min:x_max] > 1).astype(np.int32)
        
        self.coverage_map[y_min:y_max, x_min:x_max][submask] += 1
        
        new_coverage = (self.coverage_map[y_min:y_max, x_min:x_max] > 0).astype(np.int32)
        new_repeat = (self.coverage_map[y_min:y_max, x_min:x_max] > 1).astype(np.int32)
        
        newly_observed = np.sum(new_coverage) - np.sum(old_coverage)
        newly_repeated = np.sum(new_repeat) - np.sum(old_repeat)

        self.total_observed_cells += newly_observed
        self.repeat_observed_cells += newly_repeated

        aori = self.calculate_aori()
        self.aori_history.append(aori)
        
        return aori
        
    def calculate_aori(self):
        if self.total_observed_cells == 0:
            return 0.0

        overlap_ratio = self.repeat_observed_cells / max(1, self.total_observed_cells)
        
        unique_coverage_ratio = 1.0 - overlap_ratio
        
        step_density = len(self.positions) / max(1, self.total_observed_cells)
        normalized_step_density = min(1.0, step_density / 10.0) 

        aori = 1.0 - (0.8 * unique_coverage_ratio**2 + 0.2 * (1.0 - normalized_step_density))
        
        return aori
    
    def get_metrics(self):
        
        total_steps = len(self.positions)
        path_length = 0.0
        for i in range(1, len(self.positions)):
            path_length += np.linalg.norm(np.array(self.positions[i][:3]) - np.array(self.positions[i-1][:3]))

        coverage_efficiency = self.total_observed_cells / max(1.0, path_length) if path_length > 0 else 0

        aori_trend = 0.0
        if len(self.aori_history) >= 10:
            mid_point = len(self.aori_history) // 2
            early_aori = self.aori_history[mid_point] - self.aori_history[0]
            late_aori = self.aori_history[-1] - self.aori_history[mid_point]
            aori_trend = late_aori / (early_aori + 1e-5)  
        
        return {
            "aori": self.calculate_aori(),
            "total_observed_cells": self.total_observed_cells,
            "repeat_observed_cells": self.repeat_observed_cells, 
            "repeat_percentage": 100.0 * self.repeat_observed_cells / max(1, self.total_observed_cells),
            "unique_coverage": self.total_observed_cells - self.repeat_observed_cells,
            "path_length": path_length,
            "coverage_efficiency": coverage_efficiency,
            "aori_trend": aori_trend,
            "step_count": total_steps
        }
    
    def _world_to_grid(self, world_x, world_z):
        if self.origin is None:
            return self.map_size // 2, self.map_size // 2
            
        grid_x = int((world_x - self.origin[0]) / self.grid_resolution + self.map_size // 2)
        grid_y = int((world_z - self.origin[1]) / self.grid_resolution + self.map_size // 2)

        grid_x = max(0, min(self.map_size - 1, grid_x))
        grid_y = max(0, min(self.map_size - 1, grid_y))
        
        return grid_x, grid_y
    
    def generate_visualization(self, show_path=True, show_current=True):

        vis_map = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)
        

        unique_mask = (self.coverage_map == 1)
        repeat_mask = (self.coverage_map > 1)
        

        vis_map[unique_mask] = [120, 200, 120] 
        
        heat_values = np.minimum(self.coverage_map[repeat_mask], 10) / 10.0  
        red_channel = (100 + 155 * heat_values).astype(np.uint8)
        vis_map[repeat_mask, 0] = red_channel
        vis_map[repeat_mask, 1] = (80 * (1 - heat_values)).astype(np.uint8) 
        
        if show_path and len(self.positions) > 1:
            for i in range(1, len(self.positions)):
                start = self._world_to_grid(self.positions[i-1][0], self.positions[i-1][2])
                end = self._world_to_grid(self.positions[i][0], self.positions[i][2])
                cv2.line(vis_map, start, end, (0, 0, 255), 1) 

        if show_current and self.positions:
            current = self._world_to_grid(self.positions[-1][0], self.positions[-1][2])
            cv2.circle(vis_map, current, 5, (255, 255, 0), -1) 

        aori = self.calculate_aori()
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        info_panel = np.ones((80, self.map_size, 3), dtype=np.uint8) * 240
        
        cv2.putText(info_panel, f"AORI: {aori:.3f}", (10, 20), font, 0.5, (0, 0, 0), 1)
        cv2.putText(info_panel, f"Unique: {self.total_observed_cells - self.repeat_observed_cells}", 
                   (10, 40), font, 0.5, (0, 0, 0), 1)
        cv2.putText(info_panel, f"Repeat: {self.repeat_observed_cells}", 
                   (10, 60), font, 0.5, (0, 0, 0), 1)
        
        cv2.putText(info_panel, f"Repeat %: {100.0 * self.repeat_observed_cells / max(1, self.total_observed_cells):.1f}%", 
                   (200, 20), font, 0.5, (0, 0, 0), 1)
        cv2.putText(info_panel, f"Steps: {len(self.positions)}", 
                   (200, 40), font, 0.5, (0, 0, 0), 1)
        
        final_vis = np.vstack([vis_map, info_panel])
        
        return final_vis

class SemanticNode:
    """表示语义记忆森林中的节点，支持层次结构"""
    def __init__(self, node_id, level, caption, position=None, embedding=None):
        self.id = node_id
        self.level = level  
        self.caption = caption
        self.position = position
        self.embedding = embedding
        self.children = []
        self.parent = None
        self.timestamp = time.time()
        self.image = None
        
    def add_child(self, child):
        if child not in self.children:
            self.children.append(child)
            child.parent = self
            
    def get_all_children_recursive(self):
        result = []
        for child in self.children:
            result.append(child)
            result.extend(child.get_all_children_recursive())
        return result
        
class SemanticForest:
    def __init__(self, spatial_weight=0.4):
        self.leaf_nodes = [] 
        self.all_nodes = {}
        self.levels = {0: []}
        self.root_nodes = []
        self.spatial_weight = spatial_weight
        self.last_update_time = 0
        self.update_interval = 2.0
        self.next_node_id = 0
    
    def add_node(self, node_data):
        
        node_id = f"L0_{self.next_node_id}"
        self.next_node_id += 1
        
        leaf_node = SemanticNode(
            node_id=node_id,
            level=0,
            caption=node_data['caption'],
            position=node_data['position'],
            embedding=node_data.get('embedding')
        )
        
        if 'image' in node_data:
            leaf_node.image = node_data['image']
        
        leaf_node.timestamp = node_data['timestamp']
        
        self.leaf_nodes.append(leaf_node)
        self.all_nodes[node_id] = leaf_node
        
        node = {
            'id': node_id,
            'level': 0,
            'position': node_data['position'],
            'caption': node_data['caption'],
            'timestamp': node_data['timestamp'],
            'embedding': node_data.get('embedding'),
            'image': node_data.get('image')
        }
        if len(self.levels[0]) <= len(self.leaf_nodes):
            self.levels[0].append(node)

        current_time = time.time()
        if len(self.leaf_nodes) > 3 and (current_time - self.last_update_time) > self.update_interval:
            self._update_hierarchy()
            self.last_update_time = current_time
            
        return node_id
    
    def _update_hierarchy(self):
        if len(self.leaf_nodes) < 3:
            return
            
        try:
            preserved_nodes = {}
            for node in self.leaf_nodes:
                preserved_nodes[node.id] = node
                node.parent = None  
                node.children = [] 
                
            self.all_nodes = preserved_nodes  
            self.levels = {0: self._convert_to_old_format(self.leaf_nodes, level=0)}
            self.root_nodes = []
            
            current_nodes = self.leaf_nodes.copy()
            
            level1_nodes = self._create_area_level(current_nodes)
            self.levels[1] = self._convert_to_old_format(level1_nodes, level=1)
            
            level2_nodes = self._create_room_level(level1_nodes)
            self.levels[2] = self._convert_to_old_format(level2_nodes, level=2)
            
            home_node = self._create_home_level(level2_nodes)
            self.levels[3] = self._convert_to_old_format([home_node], level=3)

            self.root_nodes = [home_node]
            
        except Exception as e:
            logging.error(f"Error updating hierarchy: {e}")
            import traceback
            traceback.print_exc()
            
    def _create_area_level(self, leaf_nodes):

        if len(leaf_nodes) < 2:
            area_nodes = []
            for i, node in enumerate(leaf_nodes):
                area_node_id = f"L1_solo_{i}"
                area_name = self._extract_area_type([node.caption])
                area_node = SemanticNode(
                    node_id=area_node_id,
                    level=1,
                    caption=f"Area: {area_name}",
                    position=node.position.copy() if isinstance(node.position, np.ndarray) else node.position
                )
                self.all_nodes[area_node_id] = area_node
                area_node.add_child(node)
                area_nodes.append(area_node)
            return area_nodes
        
        positions = []
        embeddings = []
        for node in leaf_nodes:
            positions.append(node.position)
            if node.embedding is not None:
                embeddings.append(node.embedding)
        
        spatial_dist = distance.pdist(np.array(positions), 'euclidean')
        
        semantic_dist = np.zeros_like(spatial_dist)
        if len(embeddings) == len(leaf_nodes):
            semantic_dist = distance.pdist(np.array(embeddings), 'cosine')

        w = self.spatial_weight
        combined_dist = w * spatial_dist + (1-w) * semantic_dist
        

        dist_threshold = 1.0
        if len(leaf_nodes) > 20:
            dist_threshold = 1.5
        elif len(leaf_nodes) < 10:
            dist_threshold = 0.8
        
        Z = linkage(combined_dist, method='complete')
        clusters = fcluster(Z, dist_threshold, criterion='distance')
        
        area_nodes = []
        cluster_map = {}
        
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_map:
                node = leaf_nodes[i]

                area_name = self._extract_area_type([node.caption])
                
                area_node_id = f"L1_{cluster_id}"
                area_node = SemanticNode(
                    node_id=area_node_id,
                    level=1,
                    caption=f"Area: {area_name}",
                    position=node.position
                )
                
                self.all_nodes[area_node_id] = area_node
                area_nodes.append(area_node)
                cluster_map[cluster_id] = area_node
            
            cluster_map[cluster_id].add_child(leaf_nodes[i])
        
        for area_node in area_nodes:
            if area_node.children:
                positions = [child.position for child in area_node.children]
                area_node.position = np.mean(np.array(positions), axis=0).tolist()
        
        logging.info(f"Created {len(area_nodes)} area nodes from {len(leaf_nodes)} leaf nodes")
        
        return area_nodes

    def _extract_area_type(self, captions):
        """全面的HM3D数据集房间类型识别"""
        area_types = {
            # 住宅区域
            "kitchen": ["kitchen", "stove", "fridge", "refrigerator", "sink", "counter", "cabinets", "microwave", "oven"],
            "living room": ["living room", "sofa", "couch", "tv", "television", "coffee table", "living area", "lounge"],
            "bedroom": ["bedroom", "bed", "dresser", "closet", "nightstand", "mattress"],
            "bathroom": ["bathroom", "toilet", "shower", "bathtub", "sink", "restroom", "washroom"],
            "hallway": ["hallway", "corridor", "passage", "entrance", "foyer", "entryway"],
            "dining room": ["dining room", "dining table", "chairs", "dinner", "dining area"],
            "office": ["office", "desk", "computer", "bookshelf", "workspace", "study", "workstation"],
            "storage": ["storage", "closet", "cabinet", "pantry", "wardrobe"],
            "laundry": ["laundry", "washer", "dryer", "washing machine", "utility room"],

            "restaurant": ["restaurant", "cafe", "dining area", "bar", "cafeteria", "taco bell", "mcdonald", "fast food"],
            "store": ["store", "shop", "retail", "merchandise", "market", "supermarket", "mall"],
            "hotel room": ["hotel", "hotel room", "suite", "motel"],
            "reception": ["reception", "lobby", "front desk", "waiting area", "entrance hall"],

            "hospital room": ["hospital room", "patient room", "ward", "bed", "hospital bed", "medical"],
            "examination room": ["exam room", "examination", "doctor office", "medical equipment"],
            "operating room": ["operating room", "surgery", "surgical", "operating theatre"],
            "waiting room": ["waiting room", "waiting area", "reception area"],

            "conference room": ["conference room", "meeting room", "boardroom", "meeting table"],
            "cubicle": ["cubicle", "workstation", "office desk", "office space"],
            "break room": ["break room", "lunch room", "coffee area", "lounge"],

            "cabin": ["cabin", "ship room", "cruise room", "stateroom"],
            "deck": ["deck", "outdoor area", "balcony", "patio", "terrace"],
            "gym": ["gym", "fitness", "exercise", "workout", "equipment", "treadmill"],
            "theater": ["theater", "cinema", "auditorium", "stage", "seating area"],
            "library": ["library", "books", "bookshelves", "reading area", "study area"],
            "classroom": ["classroom", "school", "lecture", "desks", "education"]
        }
        
        type_counts = {area_type: 0 for area_type in area_types}
        
        for caption in captions:
            caption_lower = caption.lower()
            for area_type, keywords in area_types.items():
                for keyword in keywords:
                    if keyword in caption_lower:
                        type_counts[area_type] += 1

        if sum(type_counts.values()) == 0:
            return "unknown area"

        return max(type_counts.items(), key=lambda x: x[1])[0]

    def _create_room_level(self, area_nodes):
        if len(area_nodes) < 2:
            area_name = area_nodes[0].caption.split(": ")[1] if area_nodes and ": " in area_nodes[0].caption else "unknown area"
            room_node_id = f"L2_single_{area_nodes[0].id.split('_')[-1] if area_nodes else '0'}"
            room_node = SemanticNode(
                node_id=room_node_id,
                level=2,
                caption=f"Room: {area_name}",
                position=area_nodes[0].position if area_nodes else [0, 0, 0]
            )
            self.all_nodes[room_node_id] = room_node
            if area_nodes:
                room_node.add_child(area_nodes[0])
            return [room_node]

        positions = np.array([node.position for node in area_nodes])

        spatial_threshold = 3.0  
        spatial_Z = linkage(distance.pdist(positions), 'complete')
        spatial_clusters = fcluster(spatial_Z, spatial_threshold, criterion='distance')
        
        spatial_groups = {}
        for i, cluster_id in enumerate(spatial_clusters):
            if cluster_id not in spatial_groups:
                spatial_groups[cluster_id] = []
            spatial_groups[cluster_id].append(area_nodes[i])

        room_nodes = []

        for group_id, group_nodes in spatial_groups.items():

            area_types = {}
            for node in group_nodes:
                area_type = node.caption.split(": ")[1] if ": " in node.caption else "unknown"
                key_type = area_type.split()[0] 

                if "kitchen" in area_type.lower() or "dining" in area_type.lower():
                    room_type = "kitchen"
                elif "living" in area_type.lower() or "sofa" in area_type.lower():
                    room_type = "living_room"
                elif "bed" in area_type.lower() or "bedroom" in area_type.lower():
                    room_type = "bedroom"
                elif "bath" in area_type.lower() or "toilet" in area_type.lower():
                    room_type = "bathroom"
                else:
                    room_type = key_type
                    
                if room_type not in area_types:
                    area_types[room_type] = []
                area_types[room_type].append(node)

            for room_type, nodes in area_types.items():
                room_node_id = f"L2_{group_id}_{room_type}"
                room_node = SemanticNode(
                    node_id=room_node_id,
                    level=2,
                    caption=f"Room: {room_type.replace('_', ' ')}",
                    position=np.mean([node.position for node in nodes], axis=0).tolist()
                )
                self.all_nodes[room_node_id] = room_node
                
                for node in nodes:
                    room_node.add_child(node)
                    
                room_nodes.append(room_node)
        
        return room_nodes

    def _create_home_level(self, room_nodes):
        home_node_id = "L3_HOME"
        home_node = SemanticNode(
            node_id=home_node_id,
            level=3,
            caption="Complete home environment",
            position=np.mean([node.position for node in room_nodes], axis=0).tolist() if room_nodes else [0, 0, 0]
        )
        
        self.all_nodes[home_node_id] = home_node

        for node in room_nodes:
            home_node.add_child(node)
        
        return home_node

    def _convert_to_old_format(self, nodes, level):
        result = []
        for node in nodes:
            old_format = {
                'id': node.id,
                'level': node.level,
                'position': node.position,
                'caption': node.caption,
                'timestamp': node.timestamp,
                'members': [child.id for child in node.children]
            }
            result.append(old_format)
        return result
    
    def retrieve(self, query, current_pos=None, text_encoder=None, top_k=8):
        if not self.leaf_nodes:
            return []
            
        try:
            query_embedding = None
            if text_encoder:
                query_embedding = text_encoder.encode(query, show_progress_bar=False)

            candidates = []

            if self.root_nodes:

                frontier = self.root_nodes.copy() 
                visited_ids = set()

                while frontier:
                    current_node = frontier.pop(0)
                    visited_ids.add(current_node.id)

                    if current_node.level == 0:
                        candidates.append(current_node)
                        continue

                    scored_children = []
                    for child in current_node.children:
                        if child.id not in visited_ids: 
                            score = self._score_node(child, query, query_embedding, current_pos)
                            scored_children.append((child, score))

                    scored_children.sort(key=lambda x: x[1], reverse=True)

                    branching_factor = min(3, len(scored_children))
                    for i in range(branching_factor):
                        if i < len(scored_children):
                            frontier.append(scored_children[i][0])

            if len(candidates) < top_k:
                candidates = self.leaf_nodes

            scored_candidates = []
            for node in candidates:
                score = self._score_node(node, query, query_embedding, current_pos)
                scored_candidates.append((node, score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)

            result = []
            for node, _ in scored_candidates[:top_k]:
                old_format = {
                    'id': node.id,
                    'level': node.level,
                    'position': node.position,
                    'caption': node.caption,
                    'timestamp': node.timestamp,
                    'image': node.image
                }
                result.append(old_format)
                
            return result
            
        except Exception as e:
            logging.error(f"Error in retrieval: {e}")
            import traceback
            traceback.print_exc()
            return self._get_old_format_nodes()[:min(top_k, len(self.leaf_nodes))]
            
    def _score_node(self, node, query, query_embedding, current_pos):

        semantic_score = 0
        if query_embedding is not None and hasattr(node, 'embedding') and node.embedding is not None:
            semantic_score = np.dot(query_embedding, node.embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(node.embedding))

        spatial_score = 0
        if current_pos is not None and node.position is not None:
            dist = np.linalg.norm(np.array(current_pos) - np.array(node.position))
            spatial_score = 1.0 / (1.0 + dist)

        keyword_score = 0
        if query and node.caption:
            keywords = query.lower().split()
            caption_lower = node.caption.lower()
            matches = sum(keyword in caption_lower for keyword in keywords)
            keyword_score = matches / max(1, len(keywords))

            target_object = keywords[-1] 
            if target_object in caption_lower:
                keyword_score *= 1.5
        
        time_score = 0
        if hasattr(node, 'timestamp') and hasattr(self, 'step_ndx'):
            steps_ago = self.step_ndx - node.timestamp
            time_score = 1.0 / (1.0 + steps_ago * 0.1) 

        total_score = (
            0.35 * semantic_score +
            0.30 * spatial_score +
            0.20 * keyword_score +  
            0.15 * time_score  
        )
        
        return total_score

    def _get_old_format_nodes(self):
        result = []
        for node in self.leaf_nodes:
            old_format = {
                'id': node.id,
                'level': node.level,
                'position': node.position,
                'caption': node.caption,
                'timestamp': node.timestamp,
                'image': node.image
            }
            result.append(old_format)
        return result
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

    def _construct_prompt(self, **kwargs):
        raise NotImplementedError
    
    def _choose_action(self, obs):
        raise NotImplementedError

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

    def _enhanced_visual_analysis(self, image, goal):
        """更深入分析视觉内容，检测目标相关特征"""
        goal_name = goal['name'] if isinstance(goal, dict) else goal
        
        # 使用现有的VLM来判断图像中的目标相关线索
        analysis_prompt = (
            f"Analyze this image carefully for any signs of {goal_name} or paths leading to it. "
            f"Identify any indicators of {goal_name}'s location - doorways, room types, relevant objects. "
            f"Return: {{\"visible\": <0-1>, \"partial\": <0-1>, \"direction\": \"description\", \"confidence\": <0-1>}}"
        )
        
        try:
            response = self.actionVLM.call([image], analysis_prompt)
            result = self._eval_response(response)
            return result
        except:
            return {"visible": 0, "partial": 0, "direction": "unknown", "confidence": 0}
            
    def _preprocessing_module(self, obs: dict):
        """Excutes the navigability, action_proposer and projection submodules."""
        agent_state = obs['agent_state']
        images = {'color_sensor': obs['color_sensor'].copy()}
        if not self.cfg['project']:
            # Actions for the w/o proj baseline
            a_final = {
                (self.cfg['max_action_dist'], -0.28 * np.pi): 1,
                (self.cfg['max_action_dist'], 0): 2,
                (self.cfg['max_action_dist'], 0.28 * np.pi): 3,
            }
            if hasattr(self, 'precision_mode') and self.precision_mode:
                cv2.putText(
                    images['color_sensor'], 
                    "PRECISION MODE ACTIVE", 
                    (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.8, 
                    (0, 255, 0), 
                    2
                )
                a_final = {
                    (self.cfg['max_action_dist'] * self.precision_scale, -0.28 * np.pi): 1,
                    (self.cfg['max_action_dist'] * self.precision_scale, 0): 2,
                    (self.cfg['max_action_dist'] * self.precision_scale, 0.28 * np.pi): 3,
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

            if hasattr(self, 'precision_mode') and self.precision_mode:
                a_final = [(dist * self.precision_scale, theta) for dist, theta in a_final]
        
        else:
            a_initial = self._navigability(obs)
            a_final = self._action_proposer(a_initial, agent_state)

            if hasattr(self, 'precision_mode') and self.precision_mode:
                a_final = [(dist * self.precision_scale, theta) for dist, theta in a_final]

        a_final_projected = self._projection(a_final, images, agent_state)
        images['voxel_map'] = self._generate_voxel(a_final_projected, agent_state=agent_state)
        return a_final_projected, images

    def _stopping_module(self, stopping_images: list[np.array], goal):
        """Determines if the agent should stop."""
        stopping_prompt = self._construct_prompt(
            goal=goal, 
            prompt_type='stopping'
        )
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

        min_safe_distance = max(0.3, self.cfg['min_action_dist'] * 1.2) 
        
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
            is_unexplored = score < 3
            

            search_multiplier = 1.0
            if hasattr(self, 'searched_areas') and hasattr(self, '_get_area_id'):

                dest_x = agent_state.position[0] + mag*np.sin(theta)
                dest_z = agent_state.position[2] - mag*np.cos(theta)
                target_pos = [dest_x, agent_state.position[1], dest_z]
                
                area_id = self._get_area_id(target_pos)
                if area_id in self.searched_areas:
                    area_info = self.searched_areas[area_id]
                    
                    if area_info["found_target"]:
                        search_multiplier = 1.5
                        is_unexplored = True 
                    else:

                        steps_since_search = self.step_ndx - area_info["timestamp"]
                        if steps_since_search < 30:
                            search_multiplier = 0.3  
                        elif steps_since_search < 60:
                            search_multiplier = 0.6  
                        else:
                            search_multiplier = 0.8  
            arrowData.append([clip_frac*mag*search_multiplier, theta, is_unexplored])
        
        arrowData.sort(key=lambda x: x[1])
        thetas = set()
        out = []
        filter_thresh = 0.75
        filtered = list(filter(lambda x: x[0] > filter_thresh, arrowData))
        quality_filtered = []
        for mag, theta, is_unexplored in filtered:
            if mag < min_safe_distance:
                continue
            quality_filtered.append([mag, theta, is_unexplored])
        if quality_filtered:
            filtered = quality_filtered

        filtered.sort(key=lambda x: x[1])
        if filtered == []:
            return []

        if explore:
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

        if not out or (len(out) > 0 and max(out, key=lambda x: x[0])[0] < min_safe_distance):
            if (self.step_ndx - self.turned) < self.cfg['turn_around_cooldown']:
                return self._get_default_arrows()
            else:
                return [(0.5, np.pi)] 
        
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

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict):
        """
        Prompting component of VLMNav. Constructs the textual prompt and calls the action model.
        Parses the response for the chosen action number.
        """
        prompt_type = 'action' if self.cfg['project'] else 'no_project'
        action_prompt = self._construct_prompt(goal, prompt_type, num_actions=len(a_final))

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])
        
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

    def _get_navigability_mask(self, rgb_image, depth_image, agent_state, sensor_state):
        if self.cfg['navigability_mode'] == 'segmentation':
            navigability_mask = self.segmentor.get_navigability_mask(rgb_image)
        else:
            thresh = 0.8 if self.cfg['navigability_mode'] == 'depth_estimate' else \
                    min(0.8, self.cfg['navigability_height_threshold'])
            height_map = depth_to_height(depth_image, self.fov, sensor_state.position, sensor_state.rotation)
            
            navigability_mask = abs(height_map - (agent_state.position[1] - 0.04)) < thresh

            kernel = np.ones((3, 3), np.uint8)
            navigability_mask = cv2.erode(navigability_mask.astype(np.uint8), kernel, iterations=1).astype(bool)

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
                local_to_global(sensor_state.position, sensor_state.rotation, camera_coords)
            )
            r_i = np.linalg.norm([local_coords[0], local_coords[2]])

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

    def _get_optimal_turn_angle(self):

        default_turn_angle = np.pi

        if not hasattr(self, 'memory_nodes') or len(self.memory_nodes) < 6:
            return default_turn_angle
        
        try:
            recent_positions = [node['position'] for node in self.memory_nodes[-6:]]
            
            directions = []
            for i in range(1, len(recent_positions)):
                prev_pos = np.array(recent_positions[i-1])
                curr_pos = np.array(recent_positions[i])
                if np.linalg.norm(curr_pos - prev_pos) > 0.1:
                    direction = curr_pos - prev_pos
                    directions.append(direction[:3])
            
            if len(directions) >= 2:
                avg_direction = np.mean(directions, axis=0)

                if np.linalg.norm(avg_direction) > 0.5:

                    avg_direction = avg_direction / np.linalg.norm(avg_direction)

                    turn_angle = np.arctan2(avg_direction[0], avg_direction[2]) + np.pi

                    jitter = np.random.uniform(-0.2, 0.2)
                    return turn_angle + jitter
        
        except Exception as e:
            logging.error(f"Error in _get_optimal_turn_angle: {e}")

        return default_turn_angle + np.random.uniform(-0.3, 0.3)

    def _is_stuck_in_small_area(self, goal_info=None):
        """检测代理是否在小区域停留过长时间且没有目标信息"""
        if len(self.memory_nodes) < 15:  
            return False
            
        recent_positions = np.array([node['position'] for node in self.memory_nodes[-15:]])

        max_distance = 0
        for i in range(len(recent_positions)):
            for j in range(i+1, len(recent_positions)):
                dist = np.linalg.norm(recent_positions[i] - recent_positions[j])
                max_distance = max(max_distance, dist)

        start_pos = recent_positions[0]
        end_pos = recent_positions[-1]
        net_movement = np.linalg.norm(end_pos - start_pos)

        has_target_info = False
        goal_name = self._extract_goal_name()
        
        for i in range(1, 16): 
            if len(self.memory_nodes) >= i:
                node = self.memory_nodes[-i]
                if 'caption' in node and goal_name and goal_name.lower() in node['caption'].lower():
                    caption = node['caption'].lower()
                    goal_pos = caption.find(goal_name.lower())
                    context_before = caption[max(0, goal_pos-20):goal_pos] if goal_pos >= 0 else ""
                    
                    negation_terms = ["no", "not", "don't see", "cannot see", "couldn't find"]
                    is_negated = any(neg in context_before for neg in negation_terms)
                    
                    if not is_negated:
                        has_target_info = True
                        break
                        
        is_stuck = (max_distance < 1.5 and 
                    net_movement < 0.8 and 
                    not has_target_info)
        
        if is_stuck and not hasattr(self, 'stuck_areas'):
            self.stuck_areas = []
        
        if is_stuck:

            area_center = np.mean(recent_positions, axis=0)
            self.stuck_areas.append({
                'center': area_center,
                'timestamp': self.step_ndx,
                'radius': max(max_distance, 1.5)  
            })
            
            logging.warning(f"Stuck! navigastion range:{max_distance:.2f}m, movement:{net_movement:.2f}m, distance:{target_distance:.2f}m")
        
        return is_stuck

    def _generate_escape_from_area(self):
        if not hasattr(self, 'stuck_areas') or not self.stuck_areas:
            return PolarAction(1.0, np.random.uniform(-np.pi, np.pi))
        
        current_pos = np.array(self.memory_nodes[-1]['position'])
        closest_area = None
        min_dist = float('inf')
        
        for area in self.stuck_areas:
            dist = np.linalg.norm(current_pos - area['center'])
            if dist < min_dist:
                min_dist = dist
                closest_area = area
        
        if not closest_area or min_dist > 3.0:  
            return PolarAction(1.0, np.random.uniform(-np.pi, np.pi))

        escape_vector = current_pos - closest_area['center']

        if np.linalg.norm(escape_vector) < 0.5:
            angle = np.random.uniform(0, 2*np.pi)
            escape_vector = np.array([np.cos(angle), 0, np.sin(angle)])
        
        escape_vector = escape_vector / np.linalg.norm(escape_vector)
        escape_angle = np.arctan2(escape_vector[0], -escape_vector[2])

        for area in self.stuck_areas:
            if area == closest_area:
                continue

            future_pos = current_pos + 2.0 * np.array([np.sin(escape_angle), 0, -np.cos(escape_angle)])
            dist_to_area = np.linalg.norm(future_pos - area['center'])

            if dist_to_area < area['radius'] + 1.0:
                escape_angle += np.pi/2
                break

        distance = min(2.0, self.cfg['max_action_dist'])
        logging.info(f"generate escaping action: distance={distance:.2f}m, degrees={np.degrees(escape_angle):.1f}°")
        
        return PolarAction(distance, escape_angle)

    def _is_agent_stuck(self):
        if not hasattr(self, 'memory_nodes') or len(self.memory_nodes) < 10:
            return False
            
        recent_positions = np.array([node['position'] for node in self.memory_nodes[-10:]])
        recent_rotations = [node.get('rotation', [0,0,0,1]) for node in self.memory_nodes[-10:]]

        path_length = 0
        for i in range(1, len(recent_positions)):
            path_length += np.linalg.norm(recent_positions[i] - recent_positions[i-1])

        direct_distance = np.linalg.norm(recent_positions[-1] - recent_positions[0])

        path_efficiency = direct_distance / max(0.001, path_length)

        centroid = np.mean(recent_positions, axis=0)

        avg_distance_to_centroid = np.mean([np.linalg.norm(p - centroid) for p in recent_positions])

        rotation_changes = 0
        for i in range(1, len(recent_rotations)):
            q1 = recent_rotations[i-1]
            q2 = recent_rotations[i]
            dot_product = min(1.0, abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]))
            rotation_changes += np.arccos(dot_product) * 2  

        if not hasattr(self, 'stuck_scores'):
            self.stuck_scores = []

        current_stuck_score = 0
        if path_efficiency < 0.25: 
            current_stuck_score += 4
        if avg_distance_to_centroid < 0.35:
            current_stuck_score += 3
        if rotation_changes > 2.0 and path_length < 0.5:
            current_stuck_score += 3

        self.stuck_scores.append(current_stuck_score)
        if len(self.stuck_scores) > 5:
            self.stuck_scores.pop(0)

        avg_stuck_score = sum(self.stuck_scores) / len(self.stuck_scores)

        is_stuck = avg_stuck_score > 6.0
        
        if is_stuck:
            self.last_stuck_position = recent_positions[-1].copy()
            self.last_stuck_time = self.step_ndx
            
            details = {
                "path_efficiency": f"{path_efficiency:.3f}",
                "area_density": f"{avg_distance_to_centroid:.3f}m",
                "rotation_changes": f"{rotation_changes:.2f}",
                "stuck_score": f"{avg_stuck_score:.1f}/10"
            }
            logging.warning(f"STUCK DETECTION: Agent appears to be stuck! {details}")
        
        return is_stuck
        
    def _get_escape_action(self):

        narrow_passage = False
        corner_trap = False
        obstacle_ahead = False
        
        if hasattr(self, 'memory_nodes') and len(self.memory_nodes) > 0:
            recent_caption = self.memory_nodes[-1].get('caption', '').lower()

            if any(term in recent_caption for term in ['narrow', 'tight', 'small passage', 'doorway']):
                narrow_passage = True
            if any(term in recent_caption for term in ['corner', 'dead end', 'wall', 'blocked']):
                corner_trap = True
            if any(term in recent_caption for term in ['table', 'chair', 'obstacle', 'furniture']):
                obstacle_ahead = True

        if corner_trap:
            logging.info("Corner trap detected, performing 180 degree turn")
            return PolarAction(0, np.pi + np.random.uniform(-0.2, 0.2))
            
        elif narrow_passage:
            logging.info("Narrow passage detected, backing up slightly")
            return PolarAction(-0.3, np.random.uniform(-0.4, 0.4))
            
        elif obstacle_ahead:
            angle = np.random.choice([np.pi/4, np.pi/2, -np.pi/4, -np.pi/2])
            logging.info(f"Obstacle detected, turning {int(np.degrees(angle))} degrees")

        successful_directions = []
        if hasattr(self, 'memory_nodes') and len(self.memory_nodes) >= 10:
            positions = np.array([node['position'] for node in self.memory_nodes[-10:]])
            for i in range(1, len(positions)):
                move_vector = positions[i] - positions[i-1]
                dist = np.linalg.norm(move_vector)
                if dist > 0.3: 
                    successful_directions.append(move_vector[:3] / dist) 

        if successful_directions:
            main_dir = np.mean(successful_directions, axis=0)

            perpendicular = np.array([-main_dir[2], 0, main_dir[0]])

            if np.random.random() > 0.5:
                perpendicular = -perpendicular
                
            escape_angle = np.arctan2(perpendicular[0], -perpendicular[2])
            logging.info(f"Using perpendicular escape angle: {np.degrees(escape_angle):.1f}°")
            return PolarAction(0, escape_angle)

        random_angle = np.random.uniform(-np.pi, np.pi)
        logging.info(f"Using random escape angle: {np.degrees(random_angle):.1f}°")
        return PolarAction(0, random_angle)
        
    def _action_number_to_polar(self, action_number: int, a_final: list):
        try:
            action_number = int(action_number)

            is_stuck = self._is_agent_stuck() if hasattr(self, '_is_agent_stuck') else False
            if is_stuck:
                logging.warning("转换动作: 检测到卡住状态，使用逃脱策略")
                if action_number == 0:
                    if hasattr(self, '_get_escape_action'):
                        return self._get_escape_action()
                    else:

                        random_turn = np.random.uniform(-np.pi, np.pi)
                        return PolarAction(0, random_turn)
                        
                if action_number <= len(a_final) and action_number > 0:
                    r, theta = a_final[action_number - 1]
                    r = max(r, 0.3)
                    theta += np.random.uniform(-0.15, 0.15) 
                    return PolarAction(r, -theta)
                    
            if action_number <= len(a_final) and action_number > 0:
                r, theta = a_final[action_number - 1]

                if r < 0.2:
                    r = 0.2
                    
                return PolarAction(r, -theta)
                
            if action_number == 0:
                turn_angle = self._get_optimal_turn_angle()
                return PolarAction(0, turn_angle)
                
        except ValueError:
            pass

        return self._get_smart_fallback_action(a_final)
    
    def _get_smart_fallback_action(self, a_final):

        if self.step_ndx > 10:
            recent_positions = [np.array(node['position']) 
                            for node in self.memory_nodes[-5:]] if hasattr(self, 'memory_nodes') else []
            if recent_positions and len(recent_positions) >= 3:
                movements = [np.linalg.norm(recent_positions[i] - recent_positions[i-1]) 
                        for i in range(1, len(recent_positions))]
                avg_movement = np.mean(movements)

                if avg_movement < 0.1:
                    logging.warning("Detected possible stuck state, executing evasive turn")
                    return PolarAction(0, np.pi * 0.7)

        if a_final:
            longest = max(a_final, key=lambda x: x[0])
            return PolarAction(longest[0], -longest[1])

        return PolarAction(0.2, np.pi/4)  

    def _eval_response(self, response):
        try:
            try:
                result = json.loads(response)
                return result
            except json.JSONDecodeError:
                pass

            match = re.search(r'\{[^{}]*\}', response)
            if match:
                json_str = match.group(0).replace("'", '"')
                try:
                    result = json.loads(json_str)
                    return result
                except json.JSONDecodeError:
                    pass

            match = re.search(r'"?action"?\s*:\s*(\d+)', response)
            if match:
                action_num = int(match.group(1))
                return {"action": action_num}

            match = re.search(r'(\d+)', response)
            if match:
                action_num = int(match.group(1))
                return {"action": action_num}
                
            logging.error(f"Could not extract action from response: {response}")
            return {"action": -10}
            
        except Exception as e:
            logging.error(f"Error parsing response: {e}")
            return {"action": -10}

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
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images
        }
        return agent_action, metadata
    
    def _construct_prompt(self, goal: dict, prompt_type: str, num_actions=0):
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
            stopping_prompt = (f"The agent has the following navigation task: \n{task}\n. The agent has sent you an image taken from its current location{' as well as the goal image. ' if goal['mode'] == 'image' else '. '} "
                                f'Your job is to determine whether the agent is close to the specified {goal["name"].upper()}, And you have to be careful that it must be very close and that there are no obstacles in the middle 0.1m distance blocking it, or else it will be FALSE POSITIVE.'
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
                f"TASK: NAVIGATE TO THE NEAREST {goal['name'].upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. "
                f"{'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                "Let's solve this navigation task step by step:\\n"
                "1. Current State: What do you observe in the environment? What objects and pathways are visible?\\n"
                "2. Goal Analysis: Based on the target and home layout knowledge, where is the {goal} likely to be?\\n"
                "3. Path Planning: What's the most promising direction to reach the target? Consider available paths and typical room layouts.\\n"
                "4. Action Decision: Which numbered arrow best serves your plan? Return your choice as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS."
            )
            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

    def reset_goal(self):
        """Called after every subtask of GOAT. Notably does not reset the voxel map, only resets all areas to be unexplored"""
        self.stopping_calls = [self.step_ndx-2]
        self.explored_map = np.zeros_like(self.explored_map)
        self.turned = self.step_ndx - self.cfg['turn_around_cooldown']

class EnhancedGOATAgent(VLMNavAgent):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.area_tracker = AreaOverlapTracker(
            grid_resolution=0.2, 
            vision_radius=0.6,  
            map_size=1000
        )

        try:
            model_path = "models/clip-ViT-B-32"
            if os.path.exists(model_path):
                self.text_encoder = SentenceTransformer(model_path)
                logging.info(f"Loaded text encoder from local path: {model_path}")
            else:
                self.text_encoder = SentenceTransformer('clip-ViT-B-32')
                self.text_encoder.save(model_path)
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer: {e}")
            self.text_encoder = None

        self.topological_map = TopologicalMap()
        self.semantic_forest = SemanticForest()
        self.memory_nodes = []
        self.memory_enabled = True
        self.caption_cache = {}
        self.memory_update_interval = 3
        self.memory_distance_threshold = 0.3

        self.precision_mode = False
        self.precision_scale = 0.1
        self.searched_areas = {}
        self.area_grid_size = 2.0

        self.last_analyzed_area = None
        self.last_visual_analysis_step = -5
        self.visual_analysis_interval = 5
        self.enable_visual_analysis = True
        self.visual_analysis_cache = {}
        self.visual_analysis_cache_ttl = 20

        self.aori_values = []
    
    def reset(self):

        super().reset()

        if hasattr(self, 'memory_nodes'):
            logging.info(f"reset memory: 清clear {len(self.memory_nodes)} nodes")
        
        self.topological_map = TopologicalMap()
        self.semantic_forest = SemanticForest()
        self.memory_nodes = []
        self.caption_cache = {}

        self.last_analyzed_area = None
        self.last_visual_analysis_step = -5
        self.visual_analysis_cache = {}

        self.aori_values = []
        if hasattr(self, 'area_tracker'):
            self.area_tracker.reset()
        
        logging.info("set complete")
    
    def reset_goal(self):
        self.stopping_calls = [self.step_ndx-2]
        self.explored_map = np.zeros_like(self.explored_map)
        self.turned = self.step_ndx - self.cfg['turn_around_cooldown']
        
        self.precision_mode = False
        
        if hasattr(self, 'area_tracker'):
            self.area_tracker.reset()
        
        logging.info(f"set，keep {len(self.memory_nodes)} nodes")
    
    def step(self, obs: dict):
        try:
            agent_state = obs['agent_state']

            if hasattr(self, 'area_tracker'):
                position = agent_state.position
                current_aori = self.area_tracker.update(position)
                if not hasattr(self, 'aori_values'):
                    self.aori_values = []
                self.aori_values.append(current_aori)

            should_update_memory = False

            if self.step_ndx % self.memory_update_interval == 0:
                should_update_memory = True

            if len(self.memory_nodes) > 0:
                last_pos = self.memory_nodes[-1]['position']
                current_pos = agent_state.position.tolist()
                distance_moved = np.linalg.norm(np.array(current_pos) - np.array(last_pos))
                if distance_moved > 0.5:
                    should_update_memory = True
            
            if should_update_memory:
                try:
                    self._build_memory(obs)
                except Exception as mem_e:
                    logging.error(f"Memory building error: {mem_e}")
                    import traceback
                    traceback.print_exc()

            agent_action, metadata = super().step(obs)

            if hasattr(self, 'area_tracker') and hasattr(self, 'aori_values') and len(self.aori_values) > 0:
                if 'logging_data' not in metadata:
                    metadata['logging_data'] = {}
                metadata['logging_data']['AORI'] = self.aori_values[-1]

            try:
                if self.memory_enabled and len(self.memory_nodes) > 0:
                    if 'images' not in metadata:
                        metadata['images'] = {}
                    
                    memory_map = self._visualize_memory()
                    if memory_map is not None:
                        metadata['images']['memory_map'] = memory_map
                        
                if hasattr(self, 'area_tracker') and hasattr(self.area_tracker, 'total_observed_cells'):
                    if 'images' not in metadata:
                        metadata['images'] = {}
                    metadata['images']['area_coverage'] = self.area_tracker.generate_visualization()
            except Exception as vis_e:
                logging.error(f"Visualization error: {vis_e}")
                
            return agent_action, metadata
            
        except Exception as e:
            logging.error(f"Error in step: {str(e)}")
            import traceback
            traceback.print_exc()

            agent_action = PolarAction(0, 0)
            metadata = {
                'step_metadata': {'success': 0, 'error': str(e)},
                'logging_data': {},
                'images': {'color_sensor': obs.get('color_sensor', np.zeros((100, 100, 3), dtype=np.uint8))}
            }
            return agent_action, metadata
    
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

        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
            if hasattr(self, '_mark_area_as_searched'):
                self._mark_area_as_searched(agent_state.position, found_target=True)
                
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {'REASON': 'Model confirmed stop'}

            if hasattr(self, 'precision_mode') and self.precision_mode:
                logging.info("Navigation complete - deactivating precision mode")
                self.precision_mode = False

        elif hasattr(self, 'precision_mode') and not self.precision_mode and len(self.stopping_calls) >= 1 and self.stopping_calls[-1] == self.step_ndx - 1:
            if hasattr(self, '_mark_area_as_searched'):
                self._mark_area_as_searched(agent_state.position, found_target=True)
                
            logging.info(f"Activating precision navigation mode (step length scaled to {self.precision_scale})")
            self.precision_mode = True
            step_metadata['precision_mode'] = True

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
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

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
                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

                if hasattr(self, '_is_agent_stuck') and self._is_agent_stuck():
                    logging.warning("stuck! escape")
                    if hasattr(self, '_get_escape_action'):
                        agent_action = self._get_escape_action()
                        logging_data['STUCK_ESCAPE'] = "Used escape strategy"

        logging_data['STOPPING RESPONSE'] = stopping_response
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images
        }
        return agent_action, metadata
        
    
    def _build_memory(self, obs):
        try:
            agent_state = obs['agent_state']
            image = self._ensure_rgb_image(obs['color_sensor'])
            goal = obs['goal']
            goal_name = goal['name'] if isinstance(goal, dict) else str(goal)
            
            full_caption = self._generate_caption(image)

            target_mention, target_confidence = self._analyze_target_mention(full_caption, goal_name)
            
            position = agent_state.position.tolist()
            rotation = [agent_state.rotation.x, agent_state.rotation.y, 
                     agent_state.rotation.z, agent_state.rotation.w]
            timestamp = self.step_ndx

            embedding = None
            if self.text_encoder is not None:
                try:
                    words = full_caption.split()
                    if len(words) > 30:
                        clip_caption = " ".join(words[:30])
                    else:
                        clip_caption = full_caption
                        
                    embedding = self.text_encoder.encode(clip_caption, show_progress_bar=False)
                except Exception as e:
                    logging.error(f"Error encoding caption: {e}")

            node_data = {
                'position': position,
                'rotation': rotation,
                'timestamp': timestamp,
                'image': image.copy() if image is not None else None,
                'caption': full_caption,
                'embedding': embedding,
                'goal': goal,
                'target_mention': target_mention,
                'target_confidence': target_confidence
            }

            self.topological_map.add_node(position, rotation, timestamp, 
                                       image.copy() if image is not None else None, 
                                       full_caption)

            self.semantic_forest.add_node(node_data)

            self.memory_nodes.append(node_data)

            is_near_goal = len(self.stopping_calls) > 0 and self.stopping_calls[-1] == self.step_ndx - 1

            if is_near_goal and len(self.semantic_forest.leaf_nodes) >= 3:
                logging.info("near target, refresh memory")
                self.semantic_forest._update_hierarchy()
                self.semantic_forest.last_update_time = time.time()

        except Exception as e:
            logging.error(f"Error building memory: {e}")
    
    def _ensure_rgb_image(self, image):
        if image is None:
            return None
            
        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] != 3:
            return image[:, :, :3]
        return image
    
    def _generate_caption(self, image):
        if image is None:
            return "No image available"

        image_hash = hash(str(image.tobytes())) if hasattr(image, 'tobytes') else hash(str(image))
        if image_hash in self.caption_cache:
            return self.caption_cache[image_hash]
        
        try:
            caption_prompt = (
                "Describe this scene focusing on: 1) The type of room and visible area, "
                "2) Key objects and furniture, 3) Visible doorways and openings, "
                "4) Spatial layout and landmarks that would help with navigation decisions."
            )
            
            caption = self.actionVLM.call([image], caption_prompt)
            caption = caption.strip()
            self.caption_cache[image_hash] = caption
            return caption
        except Exception as e:
            logging.error(f"Error generating caption: {e}")
            return "Indoor environment"
    
    def _analyze_target_mention(self, caption, goal_name):
        caption_lower = caption.lower()
        goal_lower = goal_name.lower()

        confirmation_patterns = [
            f"clearly see {goal_lower}", f"can see {goal_lower}", f"there is {goal_lower}",
            f"visible {goal_lower}", f"{goal_lower} is present", f"{goal_lower} in the"
        ]
        for pattern in confirmation_patterns:
            if pattern in caption_lower:
                return 'confirmed', 0.9

        negation_patterns = [
            f"no {goal_lower}", f"not {goal_lower}", f"cannot see {goal_lower}",
            f"don't see {goal_lower}", f"{goal_lower} is not visible", f"{goal_lower} is absent"
        ]
        for pattern in negation_patterns:
            if pattern in caption_lower:
                return 'negative', 0.8

        uncertainty_patterns = [
            "possibly", "might be", "could be", "perhaps", "appears to be", 
            "looks like", "maybe", "what seems to be"
        ]
        
        if goal_lower in caption_lower:
            for pattern in uncertainty_patterns:
                if pattern in caption_lower:
                    pattern_pos = caption_lower.find(pattern)
                    term_pos = caption_lower.find(goal_lower)
                    if abs(pattern_pos - term_pos) < 30:
                        return 'uncertain', 0.4

            return 'mentioned', 0.6
        return 'none', 0.0
        
    def _mark_area_as_searched(self, position, found_target=False):
        if position is None:
            return

        area_x = int(position[0] / self.area_grid_size)
        area_z = int(position[2] / self.area_grid_size)
        area_id = f"{area_x}_{area_z}"
        
        if area_id not in self.searched_areas:
            self.searched_areas[area_id] = {
                "timestamp": self.step_ndx,
                "found_target": found_target,
                "position": position.tolist() if isinstance(position, np.ndarray) else position
            }
            
        elif not self.searched_areas[area_id]["found_target"] and found_target:
            self.searched_areas[area_id]["found_target"] = True
            self.searched_areas[area_id]["timestamp"] = self.step_ndx
            
    def _get_area_id(self, position):
        area_x = int(position[0] / self.area_grid_size)
        area_z = int(position[2] / self.area_grid_size)
        return f"{area_x}_{area_z}"
    
    def _is_agent_stuck(self):
        if not hasattr(self, 'memory_nodes') or len(self.memory_nodes) < 10:
            return False
            
        recent_positions = np.array([node['position'] for node in self.memory_nodes[-10:]])
        recent_rotations = [node.get('rotation', [0,0,0,1]) for node in self.memory_nodes[-10:]]

        path_length = 0
        for i in range(1, len(recent_positions)):
            path_length += np.linalg.norm(recent_positions[i] - recent_positions[i-1])
        
        direct_distance = np.linalg.norm(recent_positions[-1] - recent_positions[0])
        path_efficiency = direct_distance / max(0.001, path_length)

        centroid = np.mean(recent_positions, axis=0)
        avg_distance_to_centroid = np.mean([np.linalg.norm(p - centroid) for p in recent_positions])

        rotation_changes = 0
        for i in range(1, len(recent_rotations)):
            q1 = recent_rotations[i-1]
            q2 = recent_rotations[i]
            dot_product = min(1.0, abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3]))
            rotation_changes += np.arccos(dot_product) * 2

        current_stuck_score = 0
        if path_efficiency < 0.25: 
            current_stuck_score += 4
        if avg_distance_to_centroid < 0.35: 
            current_stuck_score += 3
        if rotation_changes > 2.0 and path_length < 0.5:
            current_stuck_score += 3

        if not hasattr(self, 'stuck_scores'):
            self.stuck_scores = []
            
        self.stuck_scores.append(current_stuck_score)
        if len(self.stuck_scores) > 5:
            self.stuck_scores.pop(0)

        avg_stuck_score = sum(self.stuck_scores) / len(self.stuck_scores)
        return avg_stuck_score > 6.0
    
    def _get_escape_action(self):
        successful_directions = []
        if hasattr(self, 'memory_nodes') and len(self.memory_nodes) >= 10:
            positions = np.array([node['position'] for node in self.memory_nodes[-10:]])
            for i in range(1, len(positions)):
                move_vector = positions[i] - positions[i-1]
                dist = np.linalg.norm(move_vector)
                if dist > 1:  
                    successful_directions.append(move_vector[:3] / dist)

        if successful_directions:
            main_dir = np.mean(successful_directions, axis=0)
            perpendicular = np.array([-main_dir[2], 0, main_dir[0]])
            
            if np.random.random() > 0.5:
                perpendicular = -perpendicular
                
            escape_angle = np.arctan2(perpendicular[0], -perpendicular[2])
            return PolarAction(0, escape_angle)

        return PolarAction(0, np.random.uniform(-np.pi, np.pi))
    
    def _visualize_memory(self):
        if len(self.memory_nodes) < 2:
            return None

        map_size = 800
        panel_height = 60
        memory_map = np.ones((map_size, map_size, 3), dtype=np.uint8) * 245

        target_name = "Unknown"
        if len(self.memory_nodes) > 0:
            latest_node = self.memory_nodes[-1]
            if 'goal' in latest_node:
                goal_info = latest_node['goal']
                if isinstance(goal_info, dict) and 'name' in goal_info:
                    target_name = goal_info['name']
                elif isinstance(goal_info, str):
                    target_name = goal_info

        info_panel = np.ones((panel_height, map_size, 3), dtype=np.uint8) * 240
        cv2.putText(info_panel, f"Memory Map (Nodes: {len(self.memory_nodes)})", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
        cv2.putText(info_panel, f"Target: {target_name}", 
                   (map_size - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

        positions = np.array([node['position'] for node in self.memory_nodes])
        min_x, max_x = np.min(positions[:, 0]), np.max(positions[:, 0])
        min_z, max_z = np.min(positions[:, 2]), np.max(positions[:, 2])

        margin = 2.0
        min_x, max_x = min_x - margin, max_x + margin
        min_z, max_z = min_z - margin, max_z + margin
        
        scale_x = (map_size - 40) / max(1.0, (max_x - min_x))
        scale_z = (map_size - 40) / max(1.0, (max_z - min_z))
        scale = min(scale_x, scale_z)

        for i in range(len(self.memory_nodes)):
            pos = self.memory_nodes[i]['position']
            x = int(20 + (pos[0] - min_x) * scale)
            y = int(20 + (pos[2] - min_z) * scale)

            cv2.circle(memory_map, (x, y), 6, (100, 100, 100), -1)

            if i > 0:
                prev_pos = self.memory_nodes[i-1]['position']
                prev_x = int(20 + (prev_pos[0] - min_x) * scale)
                prev_y = int(20 + (prev_pos[2] - min_z) * scale)
                cv2.line(memory_map, (prev_x, prev_y), (x, y), (150, 150, 150), 2)

        memory_map_with_panel = np.vstack([info_panel, memory_map])
        
        return memory_map_with_panel

    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict):

        memory_context = ""

        has_memory = False
        if self.memory_enabled and len(self.memory_nodes) >= 3:
            try:
                goal_name = goal['name'] if isinstance(goal, dict) else str(goal)
                query = f"Find {goal_name}"

                relevant_memories = self.semantic_forest.retrieve(
                    query, step_metadata.get('agent_location'), self.text_encoder, top_k=3
                )
                
                if relevant_memories:

                    context_parts = ["## Memory Context:"]
                    for i, memory in enumerate(relevant_memories):
                        steps_ago = self.step_ndx - memory.get('timestamp', 0)
                        context_parts.append(f"- {steps_ago} steps ago: {memory.get('caption', '')[:100]}...")
                    
                    memory_context = "\n".join(context_parts)
                    has_memory = True
            except Exception as e:
                logging.error(f"Error retrieving memories: {e}")

        prompt_type = 'action' if self.cfg['project'] else 'no_project'

        action_prompt = self._construct_prompt(goal, prompt_type, num_actions=len(a_final))

        if memory_context:
            action_prompt += f"\n\n{memory_context}"

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.actionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except Exception as e:
            logging.error(f'Error parsing response: {str(e)}')
            step_metadata['success'] = 0
        
        logging_data = {
            'PROMPT': action_prompt,
            'RESPONSE': response
        }
        
        return step_metadata, logging_data, response
        
    def _construct_prompt(self, goal: dict, prompt_type: str, num_actions=0, has_memory=False):
        goal_name = goal['name'] if isinstance(goal, dict) else str(goal)
        
        if prompt_type == 'stopping':
            obj_features = self._get_object_visual_features(goal_name) if hasattr(self, '_get_object_visual_features') else ""
            stopping_prompt = (f"The agent has been tasked with navigating to a {goal_name.upper()}. The agent has sent you an image taken from its current location, the {goal_name} typically looks like: {obj_features}"
                            f'Your job is to determine whether the agent is VERY CLOSE to a {goal_name}. Note a chair is NOT sofa which is NOT a bed. '
                            f"First, tell me what you see in the image, and tell me if there is a {goal_name}. Second, return 1 if the agent is VERY CLOSE to the {goal_name} - make sure the object you see is ACTUALLY a {goal_name}, Return 0 if if there is no {goal_name}, or if it is far away, or if you are not sure. Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt

        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: NAVIGATE TO THE NEAREST {goal_name.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                            "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                            f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal_name.upper()}. Second, tell me which general direction you should go in. "
                            f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
                            )
            return baseline_prompt

        if prompt_type == 'pivot':
            pivot_prompt = f"NAVIGATE TO THE NEAREST {goal_name.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            return pivot_prompt

        if prompt_type == 'action':
            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal_name.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                f"There are {num_actions} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. "
                f"{'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                "Let's solve this navigation task step by step:\n"
                "1. Current State: What do you observe in the environment? What objects and pathways are visible? Look carefully for the target object, even if it's partially visible or at a distance.\n"
                f"2. Goal Analysis: Based on the target and home layout knowledge, where is the {goal_name} likely to be?\n"
            )

            if has_memory:
                action_prompt += (
                    "3. Memory Integration: Review the memory context below for clues about target location.\n"
                    "   - Pay special attention to memories containing or near the target object\n"
                    "   - Use recent memories (fewer steps ago) over older ones\n"
                    "   - Consider action recommendations based on memory\n\n"
                )

            section_num = 4 if has_memory else 3
            action_prompt += (
                f"{section_num}. Scene Assessment: Quickly evaluate if {goal_name} could reasonably exist in this type of space:\n"
                f"   - If you're in an obviously incompatible room (e.g., looking for a {goal_name} but in a clearly different room type), choose action 0 to TURN AROUND immediately\n"
            )
            
            section_num += 1
            action_prompt += (
                f"{section_num}. Path Planning: What's the most promising direction to reach the target? Avoid revisiting previously explored areas unless necessary. Consider:\n"
                "   - Available paths and typical room layouts\n"
                "   - Areas you haven't explored yet\n"
            )
            
            section_num += 1
            action_prompt += (
                f"{section_num}. Action Decision: Which numbered arrow best serves your plan? Return your choice as {{'action': <action_key>}}. Note:\n"
                "   - You CANNOT GO THROUGH CLOSED DOORS, It doesn't make any sense to go near a closed door.\n"
                "   - You CANNOT GO THROUGH WINDOWS AND MIRRORS\n"
                "   - You DO NOT NEED TO GO UP OR DOWN STAIRS\n"
                f"   - Please try to avoid actions that will lead you to a dead end to avoid affecting subsequent actions, unless the dead end is very close to the {goal_name}\n"
                "   - If you see the target object, even partially, choose the action that gets you closest to it\n"
            )
            
            if has_memory and hasattr(self, 'searched_areas') and self.searched_areas:
                action_prompt += (
                    "- IMPORTANT: Prioritize actions that lead to UNEXPLORED areas\n"
                    "- Avoid revisiting areas that have been thoroughly searched without finding the target\n"
                    "- If you see evidence of the target, prioritize getting closer even if the area was visited before\n"
                )

            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

class ObjectNavAgent(VLMNavAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.area_tracker = AreaOverlapTracker(
            grid_resolution=0.2,  
            vision_radius=0.6,   
            map_size=1000
        )
        try:

            model_path = "models/clip-ViT-B-32"
            if os.path.exists(model_path):
                self.text_encoder = SentenceTransformer(model_path)
                logging.info(f"Loaded text encoder from local path: {model_path}")
            else:
                self.text_encoder = SentenceTransformer('clip-ViT-B-32')
                self.text_encoder.save(model_path)
        except Exception as e:
            logging.error(f"Failed to load SentenceTransformer: {e}")
            self.text_encoder = None

        self.topological_map = TopologicalMap()
        self.semantic_forest = SemanticForest()
        self.memory_nodes = []
        self.memory_enabled = True
        self.caption_cache = {}
        self.memory_update_interval = 3
        self.memory_distance_threshold = 0.3
        self.precision_mode = False 
        self.precision_scale = 0.1 
        self.searched_areas = {} 
        self.area_grid_size = 2.0  
        self.last_analyzed_area = None
        self.last_visual_analysis_step = -5  
        self.visual_analysis_interval = 5
        self.enable_visual_analysis = True  
        self.visual_analysis_cache = {} 
        self.visual_analysis_cache_ttl = 20 

        self.aori_values = []
        self.trustworthy_metrics = {}
    def reset(self):

        super().reset()

        if hasattr(self, 'memory_nodes'):
            logging.info(f"reset memory: clear {len(self.memory_nodes)} nodes")
        
        self.topological_map = TopologicalMap() 
        self.semantic_forest = SemanticForest() 
        self.memory_nodes = [] 
        self.caption_cache = {}  

        self.last_analyzed_area = None
        self.last_visual_analysis_step = -5
        self.visual_analysis_cache = {}
        
        self.aori_values = []
        self.trustworthy_metrics = {}
        if hasattr(self, 'area_tracker'):
            self.area_tracker.reset()
        logging.info("memory all reset")

    def _mark_area_as_searched(self, position, found_target=False):

        if position is None:
            return

        area_x = int(position[0] / self.area_grid_size)
        area_z = int(position[2] / self.area_grid_size)
        area_id = f"{area_x}_{area_z}"

        if area_id not in self.searched_areas:
            self.searched_areas[area_id] = {
                "timestamp": self.step_ndx,
                "found_target": found_target,
                "position": position.tolist() if isinstance(position, np.ndarray) else position
            }

            if found_target:
                logging.info(f"note area {area_id} find target")
            else:
                logging.info(f"note area {area_id} have searched，no target")
        elif not self.searched_areas[area_id]["found_target"] and found_target:
            self.searched_areas[area_id]["found_target"] = True
            self.searched_areas[area_id]["timestamp"] = self.step_ndx
            logging.info(f"refresh {area_id}：find target")
                
    def _get_area_search_status(self, position):
        if not position:
            return "unknown"
            
        area_id = self._get_area_id(position)
        if area_id in self.searched_areas:
            return "found_target" if self.searched_areas[area_id]["found_target"] else "searched_no_target"
        return "not_searched"
        
    def _get_area_id(self, position):
        area_x = int(position[0] / self.area_grid_size)
        area_z = int(position[2] / self.area_grid_size)
        return f"{area_x}_{area_z}"
                        
    def step(self, obs: dict):
        try:
            agent_state = obs['agent_state']
            image = obs.get('color_sensor')

            if hasattr(self, 'area_tracker'):
                position = agent_state.position
                current_aori = self.area_tracker.update(position)
                if not hasattr(self, 'aori_values'):
                    self.aori_values = []
                self.aori_values.append(current_aori)

            should_update_memory = False

            if self.step_ndx % self.memory_update_interval == 0:
                should_update_memory = True

            if len(self.memory_nodes) > 0:
                last_pos = self.memory_nodes[-1]['position']
                current_pos = agent_state.position.tolist()
                distance_moved = np.linalg.norm(np.array(current_pos) - np.array(last_pos))
                if distance_moved > 0.5: 
                    should_update_memory = True

            goal_name = None
            if isinstance(obs.get('goal'), dict) and 'name' in obs['goal']:
                goal_name = obs['goal']['name'].lower()
            elif isinstance(obs.get('goal'), str):
                goal_name = obs['goal'].lower()
                
            if goal_name and image is not None:

                pass

            if should_update_memory:
                self._build_memory(obs)

            agent_action, metadata = super().step(obs)

            if hasattr(self, 'area_tracker') and hasattr(self, 'aori_values') and len(self.aori_values) > 0:
                if 'logging_data' not in metadata:
                    metadata['logging_data'] = {}

                metadata['logging_data']['AORI'] = self.aori_values[-1]

                if self.step_ndx % 2 == 0:
                    aori_metrics = self.area_tracker.get_metrics()
                    metadata['logging_data']['AORI_METRICS'] = aori_metrics
        except Exception as e:
            logging.error(f"Error in step: {e}")
            if not locals().get('agent_action') or not locals().get('metadata'):
                agent_action = PolarAction(0, 0)
                metadata = {
                    'step_metadata': {'success': 0, 'error': str(e)},
                    'logging_data': {},
                    'images': {'color_sensor': obs.get('color_sensor', np.zeros((100, 100, 3), dtype=np.uint8))}
                }

        if self.memory_enabled and len(self.memory_nodes) > 0:
            if 'images' not in metadata:
                metadata['images'] = {}

            memory_map = self._visualize_memory()
            if memory_map is not None:
                metadata['images']['memory_map'] = memory_map

        if hasattr(self, 'area_tracker') and hasattr(self.area_tracker, 'total_observed_cells') and self.area_tracker.total_observed_cells > 0:
            if 'images' not in metadata:
                metadata['images'] = {}
            metadata['images']['area_coverage'] = self.area_tracker.generate_visualization()
                
        return agent_action, metadata
        
    def calculate_trustworthy_metrics(self):
        aori_metrics = self.area_tracker.get_metrics() if hasattr(self, 'area_tracker') else {}

        has_found_target = any(area_info.get("found_target", False) for area_info in self.searched_areas.values()) if hasattr(self, 'searched_areas') else False

        exploration_effectiveness = 0.0
        if hasattr(self, 'memory_nodes') and len(self.memory_nodes) > 0:
            unique_areas = len(set(self._get_area_id(node['position']) for node in self.memory_nodes))
            explored_ratio = unique_areas / max(1, len(self.memory_nodes))
            exploration_effectiveness = explored_ratio

        trustworthy_metrics = {
            **aori_metrics,
            "has_found_target": has_found_target,
            "exploration_effectiveness": exploration_effectiveness,
            "unique_area_percentage": 100.0 - aori_metrics.get("repeat_percentage", 0.0),
            "trustworthy_score": (1.0 - aori_metrics.get("aori", 0.5)) * (
                1.0 + 0.5 * has_found_target) * exploration_effectiveness
        }
        
        self.trustworthy_metrics = trustworthy_metrics
        
        return trustworthy_metrics
        
    def _analyze_target_mention(self, caption, goal_name):

        caption_lower = caption.lower()
        goal_lower = goal_name.lower()

        confirmation_patterns = [
            f"clearly see {goal_lower}", f"can see {goal_lower}", f"there is {goal_lower}",
            f"visible {goal_lower}", f"{goal_lower} is present", f"{goal_lower} in the"
        ]
        for pattern in confirmation_patterns:
            if pattern in caption_lower:
                return 'confirmed', 0.9

        negation_patterns = [
            f"no {goal_lower}", f"not {goal_lower}", f"cannot see {goal_lower}",
            f"don't see {goal_lower}", f"{goal_lower} is not visible", f"{goal_lower} is absent"
        ]
        for pattern in negation_patterns:
            if pattern in caption_lower:
                return 'negative', 0.8

        uncertainty_patterns = [
            "possibly", "might be", "could be", "perhaps", "appears to be", 
            "looks like", "maybe", "what seems to be"
        ]
        
        if goal_lower in caption_lower:
            for pattern in uncertainty_patterns:
                if pattern in caption_lower:
                    pattern_pos = caption_lower.find(pattern)
                    term_pos = caption_lower.find(goal_lower)
                    if abs(pattern_pos - term_pos) < 30:
                        return 'uncertain', 0.4

            return 'mentioned', 0.6

        return 'none', 0.0

    def _build_memory(self, obs):

        try:
            agent_state = obs['agent_state']
            image = self._ensure_rgb_image(obs['color_sensor'])
            goal = obs['goal']
            goal_name = goal['name'] if isinstance(goal, dict) else str(goal)

            full_caption = self._generate_caption(image)

            target_mention, target_confidence = self._analyze_target_mention(full_caption, goal_name)
            
            position = agent_state.position.tolist()
            rotation = [agent_state.rotation.x, agent_state.rotation.y, 
                    agent_state.rotation.z, agent_state.rotation.w]
            timestamp = self.step_ndx

            embedding = None
            if self.text_encoder is not None:
                try:

                    words = full_caption.split()
                    if len(words) > 30: 
                        clip_caption = " ".join(words[:30])
                    else:
                        clip_caption = full_caption
                        
                    embedding = self.text_encoder.encode(clip_caption, show_progress_bar=False)
                except Exception as e:
                    logging.error(f"Error encoding caption: {e}")

            node_data = {
                'position': position,
                'rotation': rotation,
                'timestamp': timestamp,
                'image': image.copy() if image is not None else None,
                'caption': full_caption,
                'embedding': embedding,
                'goal': goal,
                'target_mention': target_mention, 
                'target_confidence': target_confidence, 
                'search_status': self._get_area_search_status(position)
            }

            if target_mention == 'confirmed':
                self._mark_area_as_searched(agent_state.position, found_target=True)
                logging.info(f"comfirmed memory：find target  {goal_name} (confidence: {target_confidence:.2f})")
            elif target_mention == 'negative':
                self._mark_area_as_searched(agent_state.position, found_target=False)

            self.topological_map.add_node(position, rotation, timestamp, 
                                        image.copy() if image is not None else None, 
                                        full_caption)

            self.semantic_forest.add_node(node_data)

            self.memory_nodes.append(node_data)
            
            logging.info(f"Built memory node {len(self.memory_nodes)}: {full_caption[:30]}...")

            is_near_goal = len(self.stopping_calls) > 0 and self.stopping_calls[-1] == self.step_ndx - 1

            if is_near_goal and len(self.semantic_forest.leaf_nodes) >= 3:
                logging.info("接近目标，强制更新记忆层次结构")
                self.semantic_forest._update_hierarchy()
                self.semantic_forest.last_update_time = time.time()

        except Exception as e:
            logging.error(f"Error building memory: {e}")
    
    def _ensure_rgb_image(self, image):
        if image is None:
            return None

        if len(image.shape) == 2:
            return np.stack([image] * 3, axis=2)
        elif len(image.shape) == 3 and image.shape[2] != 3:
            return image[:, :, :3]
        return image
    
    def _generate_caption(self, image):
        """Generate navigation-relevant descriptions to support action selection"""
        if image is None:
            return "No image available"
                
        # Check cache
        image_hash = hash(str(image.tobytes())) if hasattr(image, 'tobytes') else hash(str(image))
        if image_hash in self.caption_cache:
            return self.caption_cache[image_hash]
        
        try:
            caption_prompt = (
                "Describe this scene focusing on: 1) The type of room and visible area, "
                "2) Key objects and furniture, 3) Visible doorways and openings, "
                "4) Spatial layout and landmarks that would help with navigation decisions."
            )
            
            caption = self.actionVLM.call([image], caption_prompt)
            caption = caption.strip()
            self.caption_cache[image_hash] = caption
            return caption
        except Exception as e:
            logging.error(f"Error generating caption: {e}")
            return "Indoor environment"
    
    def _prompting(self, goal, a_final: list, images: dict, step_metadata: dict):
        memory_context = ""
        action_guidance = []

        agent_state = step_metadata.get('agent_state', None)
        current_pos = agent_state.position.tolist() if agent_state else None

        if self.memory_enabled and len(self.memory_nodes) >= 3:
            try:
                goal_name = goal['name'] if isinstance(goal, dict) else str(goal)
                query = f"Find {goal_name}"

                relevant_memories = self.semantic_forest.retrieve(
                    query, current_pos, self.text_encoder, top_k=8
                )

                relevant_memories = self._filter_relevant_memories(relevant_memories)     

                if relevant_memories:
                    memory_context = self._generate_memory_context(relevant_memories)

                    action_guidance = self._generate_action_guidance(relevant_memories, a_final)
            except Exception as e:
                logging.error(f"Error retrieving memories: {e}")

        prompt_type = 'action' if self.cfg['project'] else 'no_project'
        has_memory = self.memory_enabled and len(self.memory_nodes) >= 3 and relevant_memories
        action_prompt = self._construct_prompt(goal, prompt_type, num_actions=len(a_final), has_memory=has_memory)

        if self.memory_nodes and a_final:
            directions = self._get_action_directions(a_final)
            if directions:
                action_prompt += "\n\nAvailable action directions:\n" + directions

        if action_guidance:
            action_prompt += "\n\nAction suggestions based on memory:\n" + "\n".join(action_guidance)

        if memory_context:
            action_prompt += f"\n\n# Memory Based Navigation Context\n{memory_context}"

        prompt_images = [images['color_sensor']]
        if 'goal_image' in images:
            prompt_images.append(images['goal_image'])

        response = self.actionVLM.call_chat(self.cfg['context_history'], prompt_images, action_prompt)

        try:
            response_dict = self._eval_response(response)
            step_metadata['action_number'] = int(response_dict['action'])
        except Exception as e:
            logging.error(f'Error parsing response: {str(e)}')
            step_metadata['success'] = 0
        
        logging_data = {
            'PROMPT': action_prompt,
            'RESPONSE': response
        }
        
        if memory_context:
            logging_data['MEMORY_CONTEXT'] = memory_context
        
        return step_metadata, logging_data, response

    def _generate_action_guidance(self, relevant_memories, a_final):
        if not relevant_memories or not a_final:
            return []
            
        guidance = []
        goal_name = self._extract_goal_name()
        if not goal_name:
            return []
        
        current_pos = self.memory_nodes[-1]['position'] if self.memory_nodes else [0, 0, 0]

        confirmed_memories = []
        uncertain_memories = []
        negative_memories = []
        
        for memory in relevant_memories:
            if 'target_mention' in memory:
                mention_type = memory['target_mention']
                confidence = memory.get('target_confidence', 0.5)
            else:
                mention_type, confidence = self._analyze_target_mention(memory.get('caption', ''), goal_name)

            if mention_type == 'confirmed':
                confirmed_memories.append((memory, confidence))
            elif mention_type in ['uncertain', 'mentioned']:
                uncertain_memories.append((memory, confidence))
            elif mention_type == 'negative':
                negative_memories.append((memory, confidence))

        if confirmed_memories:
            confirmed_memories.sort(key=lambda x: x[1], reverse=True)
            best_memory, _ = confirmed_memories[0]
            
            guidance.append(f"Memory CONFIRMS {goal_name} was seen in the environment.")

            if 'position' in best_memory:
                memory_pos = best_memory['position']
                memory_vec = [memory_pos[0] - current_pos[0], memory_pos[2] - current_pos[2]]
                memory_angle = np.arctan2(memory_vec[0], -memory_vec[1])

                for (r, theta) in a_final:
                    angle_diff = abs(theta - memory_angle) % (2 * np.pi)
                    angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                    
                    if angle_diff < 0.4:
                        action_num = a_final.get((r, theta))
                        if action_num is not None:
                            guidance.append(f"Action {action_num} leads toward CONFIRMED {goal_name} location.")
                            break
        elif uncertain_memories:
            uncertain_memories.sort(key=lambda x: x[1], reverse=True)
            best_memory, confidence = uncertain_memories[0]
            
            guidance.append(f"Memory suggests {goal_name} MIGHT BE in the environment, but this is UNCERTAIN.")
            
            if 'position' in best_memory:
                memory_pos = best_memory['position']
                steps_ago = self.step_ndx - best_memory.get('timestamp', 0)
                
                guidance.append(f"Consider checking location seen {steps_ago} steps ago, but priority should be on new areas.")

        if negative_memories:
            negative_areas = []
            for memory, _ in negative_memories[:2]:
                if 'position' in memory:
                    memory_pos = memory['position']
                    memory_vec = [memory_pos[0] - current_pos[0], memory_pos[2] - current_pos[2]]
                    memory_angle = np.arctan2(memory_vec[0], -memory_vec[1])

                    for (r, theta) in a_final:
                        angle_diff = abs(theta - memory_angle) % (2 * np.pi)
                        angle_diff = min(angle_diff, 2 * np.pi - angle_diff)
                        
                        if angle_diff < 0.4: 
                            action_num = a_final.get((r, theta))
                            if action_num is not None and action_num not in negative_areas:
                                negative_areas.append(action_num)
            
            if negative_areas:
                guidance.append(f"AVOID Actions {', '.join(map(str, negative_areas))} - confirmed NO {goal_name} in those areas.")
        
        return guidance

    def _quaternion_to_yaw(self, quaternion):

        x, y, z, w = quaternion

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        
        return np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    def _get_direction_description(self, current_pos, target_pos, agent_rotation=None):
        dx = target_pos[0] - current_pos[0]
        dz = target_pos[2] - current_pos[2]

        angle = np.arctan2(dx, -dz)
        degrees = np.degrees(angle)

        if agent_rotation is not None:
            agent_angle = self._quaternion_to_yaw(agent_rotation) 
            relative_degrees = (degrees - agent_angle) % 360
        else:
            relative_degrees = degrees % 360

        direction_names = {
            0: "forward",
            45: "forward-right",
            90: "right",
            135: "backward-right",
            180: "backward",
            225: "backward-left",
            270: "left",
            315: "forward-left"
        }

        closest_direction = min(direction_names.keys(), key=lambda x: abs((relative_degrees - x + 180) % 360 - 180))
        direction_name = direction_names[closest_direction]

        dist = np.sqrt(dx**2 + dz**2)

        return {
            "text": direction_name,
            "angle_degrees": relative_degrees,
            "distance": dist,
            "is_diagonal": closest_direction % 90 != 0  
        }

    def _filter_relevant_memories(self, relevant_memories):
        filtered = []
        current_position = self.memory_nodes[-1]['position'] if self.memory_nodes else None
        current_time = self.step_ndx if hasattr(self, 'step_ndx') else None
        
        if not current_position or not current_time:
            return relevant_memories
            
        for memory in relevant_memories:
            if memory.get('timestamp') == current_time:
                continue

            if (current_time - memory.get('timestamp', 0) <= 2 and 
                'position' in memory and 
                np.linalg.norm(np.array(memory['position']) - np.array(current_position)) < 0.2):
                continue
                
            filtered.append(memory)

        if not filtered and relevant_memories:
            return [m for m in relevant_memories if m.get('timestamp') != current_time]
            
        return filtered

    def _get_object_synonyms(self, object_name):

        synonyms = {
            "chair": ["seat", "stool", "office chair", "dining chair"],
            "sofa": ["couch", "settee", "loveseat"],
            "bed": ["mattress", "sleeping place"],
            "tv_monitor": ["television", "tv", "monitor", "screen", "display"],
            "toilet": ["bathroom fixture", "commode", "lavatory", "water closet"],
        }
        
        object_name = object_name.lower()
        return synonyms.get(object_name, [])
        
    def _generate_memory_context(self, relevant_memories):

        if not relevant_memories:
            return ""

        goal_name = self._extract_goal_name()
        current_pos = self.memory_nodes[-1]['position'] if self.memory_nodes else [0, 0, 0]
        current_rotation = self.memory_nodes[-1].get('rotation') if self.memory_nodes else None

        confirmed_memories = []
        uncertain_memories = []
        negative_memories = []
        general_memories = []
        
        for memory in relevant_memories:

            if 'target_mention' in memory:
                mention_type = memory['target_mention']
            else:
                mention_type, _ = self._analyze_target_mention(memory.get('caption', ''), goal_name)

            direction_info = self._get_direction_description(current_pos, memory.get('position', [0,0,0]), current_rotation)
            steps_ago = self.step_ndx - memory.get('timestamp', 0)
            
            memory_desc = f"{direction_info['text']} direction ({direction_info['distance']:.1f}m, {steps_ago} steps ago): {memory.get('caption', '')[:100]}..."

            if mention_type == 'confirmed' and goal_name.lower() in memory.get('caption', '').lower():
                confirmed_memories.append((memory_desc, direction_info, steps_ago))
            elif mention_type == 'negative' and goal_name.lower() in memory.get('caption', '').lower():
                negative_memories.append((memory_desc, direction_info, steps_ago))
            elif mention_type in ['uncertain', 'mentioned'] and goal_name.lower() in memory.get('caption', '').lower():
                uncertain_memories.append((memory_desc, direction_info, steps_ago))
            else:
                general_memories.append((memory_desc, direction_info, steps_ago))

        context_parts = ["## Previous Observations:"]

        if confirmed_memories:
            context_parts.append("## CONFIRMED Target Observations (High Confidence)")
            confirmed_memories.sort(key=lambda x: (x[1]['distance'], x[2]))
            for memory_info in confirmed_memories[:2]:
                context_parts.append(f"- {memory_info[0]}")
            context_parts.append(f"**PRIORITY GUIDANCE**: Target {goal_name} was CONFIRMED in the {confirmed_memories[0][1]['text']} direction")

        if uncertain_memories:
            context_parts.append("## UNCERTAIN Target Observations")
            uncertain_memories.sort(key=lambda x: (x[1]['distance'], x[2]))
            for memory_info in uncertain_memories[:2]:
                context_parts.append(f"- {memory_info[0]}")
            context_parts.append(f"*Note: These observations express UNCERTAINTY about seeing {goal_name}*")

        if negative_memories:
            context_parts.append("## CONFIRMED Areas WITHOUT Target")
            negative_memories.sort(key=lambda x: x[2]) 
            for memory_info in negative_memories[:2]:
                context_parts.append(f"- {memory_info[0]}")
            context_parts.append(f"**AVOID these directions - CONFIRMED NO {goal_name} present**")
        
        context_parts.append("### Environmental Context")
        general_memories.sort(key=lambda x: x[2]) 
        if general_memories:
            for memory_info in general_memories[:4]:
                context_parts.append(f"- {memory_info[0]}")
        else:
            all_memories = confirmed_memories + uncertain_memories + negative_memories
            all_memories.sort(key=lambda x: x[2])
            for memory_info in all_memories[:3]:
                context_parts.append(f"- {memory_info[0]}")
            if not all_memories:
                context_parts.append("- No previous environmental observations available yet")

        context_parts.append("\n## Navigation Recommendations")

        if confirmed_memories:
            context_parts.append("- **HIGHEST PRIORITY**: Move toward area where target was CONFIRMED")
        elif uncertain_memories:
            context_parts.append("- Consider checking areas where target might have been seen, but with CAUTION")
        
        if negative_memories:
            context_parts.append("- AVOID areas where target was CONFIRMED NOT present")
        
        if not confirmed_memories and not uncertain_memories:
            room_suggestion = self._get_likely_room_for_object(goal_name)
            if room_suggestion:
                context_parts.append(f"- {room_suggestion}")
        
        return "\n".join(context_parts)
    
    def _get_likely_room_for_object(self, object_name):
        if not object_name:
            return None

        object_room_mapping = {
            "refrigerator": "kitchen",
            "fridge": "kitchen",
            "microwave": "kitchen", 
            "oven": "kitchen",
            "stove": "kitchen",
            "sink": "kitchen or bathroom",
            "counter": "kitchen",
            "dishwasher": "kitchen",
            "kettle": "kitchen",
            "toaster": "kitchen",
            "pot": "kitchen",
            "pan": "kitchen",
            "bowl": "kitchen",
            "plate": "kitchen",
            "cup": "kitchen",
            "knife": "kitchen",
            "fork": "kitchen",
            "spoon": "kitchen",
            "sofa": "living room",
            "couch": "living room",
            "television": "living room",
            "tv": "living room",
            "coffee table": "living room",
            "armchair": "living room",
            "entertainment center": "living room",
            "console": "living room",
            "bed": "bedroom",
            "pillow": "bedroom",
            "dresser": "bedroom",
            "wardrobe": "bedroom",
            "nightstand": "bedroom",
            "lamp": "bedroom or living room",
            "alarm clock": "bedroom",
            "toilet": "bathroom",
            "shower": "bathroom",
            "bathtub": "bathroom",
            "towel": "bathroom",
            "toothbrush": "bathroom",
            "shampoo": "bathroom",
            "dining table": "dining room",
            "chair": "dining room or any room",
            "desk": "office or bedroom",
            "computer": "office or bedroom",
            "laptop": "office or anywhere",
            "bookshelf": "office or living room",
            "book": "office or living room",
        }

        clean_name = object_name.lower().strip()
        if clean_name.endswith('s'):
            clean_name = clean_name[:-1]

        if clean_name in object_room_mapping:
            room = object_room_mapping[clean_name]
            return f"Based on common household layouts, a {clean_name} is typically found in the {room}"

        for obj, room in object_room_mapping.items():
            if obj in clean_name or clean_name in obj:
                return f"Based on common household layouts, {clean_name} might be found in the {room}"

        return f"Consider exploring common areas like living room, kitchen, or bedrooms to find the {clean_name}"

    def _visualize_memory(self):
        """生成美观且信息丰富的记忆可视化地图，集成层次结构视图"""
        try:
            if len(self.memory_nodes) < 2:
                return None

            map_size = 800
            panel_height = 60
            thumb_panel_height = 150
            hierarchy_panel_height = 240
            memory_map = np.ones((map_size, map_size, 3), dtype=np.uint8) * 245

            target_name = "Unknown"
            if len(self.memory_nodes) > 0:
                latest_node = self.memory_nodes[-1]
                if 'goal' in latest_node:
                    goal_info = latest_node['goal']
                    if isinstance(goal_info, dict) and 'name' in goal_info:
                        target_name = goal_info['name']
                    elif isinstance(goal_info, str):
                        target_name = goal_info

            info_panel = np.ones((panel_height, map_size, 3), dtype=np.uint8) * 240
            cv2.putText(info_panel, f"Memory Map (Nodes: {len(self.memory_nodes)})", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)
            cv2.putText(info_panel, f"Target: {target_name}", 
                    (map_size - 250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

            if len(self.memory_nodes) >= 2:
                positions = np.array([node['position'] for node in self.memory_nodes])
                min_x, max_x = np.min(positions[:, 0]), np.max(positions[:, 0])
                min_z, max_z = np.min(positions[:, 2]), np.max(positions[:, 2])
                margin = 2.0
                min_x, max_x = min_x - margin, max_x + margin
                min_z, max_z = min_z - margin, max_z + margin
                
                scale_x = (map_size - 40) / max(1.0, (max_x - min_x))
                scale_z = (map_size - 40) / max(1.0, (max_z - min_z))
                scale = min(scale_x, scale_z)

                for i in range(len(self.memory_nodes)):
                    pos = self.memory_nodes[i]['position']
                    x = int(20 + (pos[0] - min_x) * scale)
                    y = int(20 + (pos[2] - min_z) * scale)
                    node_color = (100, 100, 100) 
                    cv2.circle(memory_map, (x, y), 6, node_color, -1)
                    if i > 0:
                        prev_pos = self.memory_nodes[i-1]['position']
                        prev_x = int(20 + (prev_pos[0] - min_x) * scale)
                        prev_y = int(20 + (prev_pos[2] - min_z) * scale)
                        cv2.line(memory_map, (prev_x, prev_y), (x, y), (150, 150, 150), 2)
            if hasattr(self, 'searched_areas') and self.searched_areas:
                for area_id, area_info in self.searched_areas.items():
                    if 'position' not in area_info:
                        continue
                        
                    pos = area_info['position']
                    x = int(20 + (pos[0] - min_x) * scale)
                    y = int(20 + (pos[2] - min_z) * scale)

                    if area_info['found_target']:
                        cv2.rectangle(memory_map, (x-12, y-12), (x+12, y+12), (0, 200, 0), 2)
                        cv2.putText(memory_map, "T", (x-5, y+5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1)
                    else:
                        cv2.line(memory_map, (x-10, y-10), (x+10, y+10), (0, 0, 200), 2)
                        cv2.line(memory_map, (x+10, y-10), (x-10, y+10), (0, 0, 200), 2)

            thumbnail_panel = np.ones((thumb_panel_height, map_size, 3), dtype=np.uint8) * 235
            important_nodes = []

            if target_name != "Unknown":
                for i, node in enumerate(self.memory_nodes):
                    if 'caption' in node and target_name.lower() in node['caption'].lower():
                        important_nodes.append(i)
                        if len(important_nodes) >= 3:
                            break

            if important_nodes:
                cv2.putText(thumbnail_panel, "Important Observations:", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 1)

                thumbnail_width = map_size // min(len(important_nodes), 3)
                for idx, node_idx in enumerate(important_nodes[:3]):
                    node = self.memory_nodes[node_idx]
                    if 'image' in node and node['image'] is not None:
                        start_x = idx * thumbnail_width
                        thumb = cv2.resize(node['image'], (thumbnail_width - 10, 100))
                        thumbnail_panel[40:140, start_x+5:start_x+thumbnail_width-5] = thumb

                        caption = node['caption']
                        short_caption = caption[:40] + "..." if len(caption) > 40 else caption
                        cv2.putText(thumbnail_panel, short_caption, 
                                (start_x+10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                0.4, (0, 0, 0), 1)
            hierarchy_panel = np.ones((hierarchy_panel_height, map_size, 3), dtype=np.uint8) * 250
            cv2.rectangle(hierarchy_panel, (0, 0), (map_size-1, hierarchy_panel_height-1), (200, 200, 200), 1)

            cv2.putText(hierarchy_panel, "Memory Hierarchy", (20, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 50, 50), 2)

            if self.semantic_forest.root_nodes:
                self._draw_hierarchy_tree(hierarchy_panel)
            else:
                cv2.putText(hierarchy_panel, "Memory hierarchy not yet built", 
                        (map_size//2 - 150, hierarchy_panel_height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
            memory_map_with_panel = np.vstack([info_panel, memory_map])
            if important_nodes:
                memory_map_with_panel = np.vstack([memory_map_with_panel, thumbnail_panel])
            memory_map_with_panel = np.vstack([memory_map_with_panel, hierarchy_panel])
            
            return memory_map_with_panel
        
        except Exception as e:
            logging.error(f"Error generating memory visualization: {e}")
            import traceback
            traceback.print_exc()
            return None
            
    def _draw_hierarchy_tree(self, panel):
        try:
            panel_height, panel_width = panel.shape[:2]
            root_nodes = self.semantic_forest.root_nodes
            
            if not root_nodes:
                cv2.putText(panel, "No hierarchy yet", (panel_width//2-50, panel_height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 100, 100), 1)
                return
                
            leaf_count = len(self.semantic_forest.leaf_nodes)
            cv2.putText(panel, f"Leaf nodes: {leaf_count}", (panel_width-150, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)
                    
            home_node = root_nodes[0]
            room_count = len(home_node.children)
            cv2.putText(panel, f"Rooms: {room_count}", (panel_width-150, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 1)

            level_heights = {
                3: 50, 
                2: 110,
                1: 170, 
                0: 220
            }
            level_names = {
                3: "HOME",
                2: "ROOMS",
                1: "AREAS",
                0: "OBSERVATIONS"
            }
            
            for level, y_pos in level_heights.items():
                cv2.putText(panel, level_names[level], (10, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

            home_node = root_nodes[0]
            home_x = int(panel_width // 2)
            home_y = int(level_heights[3])
            node_size = 15
            cv2.circle(panel, (home_x, home_y), node_size, (50, 120, 180), -1)
            cv2.putText(panel, "HOME", (home_x - 22, home_y - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            room_nodes = home_node.children
            if room_nodes:
                room_width = min(700, panel_width - 60)
                room_start_x = home_x - room_width // 2
                room_spacing = room_width // max(len(room_nodes), 1)
                
                for i, room in enumerate(room_nodes):
                    room_x = int(room_start_x + i * room_spacing + room_spacing // 2)
                    room_y = int(level_heights[2]) 
                    room_name = room.caption.split(": ")[1] if ": " in room.caption else "Room"
                    room_color = self._get_room_color(room_name)
                    cv2.circle(panel, (room_x, room_y), node_size - 3, room_color, -1)
                    text_width = len(room_name) * 5
                    cv2.putText(panel, room_name, (room_x - text_width//2, room_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

                    cv2.line(panel, (home_x, home_y), (room_x, room_y), (120, 120, 120), 1)

                    max_areas_per_room = min(3, len(room.children))
                    area_nodes = room.children[:max_areas_per_room]

                    if area_nodes:
                        area_width = min(room_spacing * 0.8, 150)
                        area_start_x = room_x - area_width // 2
                        area_spacing = area_width // max(len(area_nodes), 1)
                        
                        for j, area in enumerate(area_nodes):
                            area_x = int(area_start_x + j * area_spacing + area_spacing // 2) 
                            area_y = int(level_heights[1])  
                            area_color = self._lighten_color(room_color)
                            cv2.circle(panel, (area_x, area_y), node_size - 5, area_color, -1)
                            
                            cv2.line(panel, (room_x, room_y), (area_x, area_y), (150, 150, 150), 1)

                            if area.children:
                                leaf_node = area.children[0]
                                leaf_x = int(area_x) 
                                leaf_y = int(level_heights[0])
                                cv2.circle(panel, (leaf_x, leaf_y), 3, (100, 100, 100), -1)
                                cv2.line(panel, (area_x, area_y), (leaf_x, leaf_y), (180, 180, 180), 1, cv2.LINE_AA)
                                leaf_count = len(area.children)
                                if leaf_count > 1:
                                    cv2.putText(panel, f"{leaf_count}", (leaf_x + 5, leaf_y + 5), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (80, 80, 80), 1)
        
        except Exception as e:
            logging.error(f"Error drawing hierarchy: {e}")
            import traceback
            traceback.print_exc
            cv2.putText(panel, "Error visualizing hierarchy", (20, panel.shape[0]//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    def _get_room_color(self, room_name):
        color_map = {
            "kitchen": (70, 130, 180),
            "living": (110, 160, 90),
            "bedroom": (180, 120, 100),
            "bathroom": (100, 100, 180),
            "hallway": (140, 140, 80),
            "dining": (160, 100, 130),
            "unknown": (120, 120, 120)
        }
        
        for key in color_map.keys():
            if key in room_name.lower():
                return color_map[key]
        return color_map["unknown"]
        
    def _lighten_color(self, color):
        return tuple(min(c + 50, 255) for c in color)

    def _extract_goal_name(self):
        if not hasattr(self, 'memory_nodes') or not self.memory_nodes:
            return ""
        
        latest_node = self.memory_nodes[-1]
        if 'goal' not in latest_node:
            return ""
            
        goal_info = latest_node['goal']
        
        if isinstance(goal_info, dict) and 'name' in goal_info:
            return goal_info['name']
        elif isinstance(goal_info, str):
            return goal_info
        
        return ""
        
    def _evaluate_exploration_status(self):
        explored_pixels = np.sum(np.all(self.explored_map == self.explored_color, axis=-1))
        visible_map_pixels = np.sum(np.any(self.voxel_map > 0, axis=-1))

        exploration_coverage = float(explored_pixels) / max(1.0, float(visible_map_pixels))

        exploration_stagnant = False
        if not hasattr(self, 'exploration_history'):
            self.exploration_history = []
        
        if len(self.exploration_history) >= 10:
            recent_change = exploration_coverage - self.exploration_history[-10]
            exploration_stagnant = recent_change < 0.01 
        
        self.exploration_history.append(exploration_coverage)
        if len(self.exploration_history) > 20:
            self.exploration_history.pop(0)
        
        return {
            'coverage': exploration_coverage,
            'sufficient': bool(exploration_coverage > 0.7),  
            'stagnant': exploration_stagnant
        }

    def _identify_unexplored_frontiers(self):
        explored_binary = np.all(self.explored_map == self.explored_color, axis=-1).astype(np.uint8)

        kernel = np.ones((5, 5), np.uint8)
        explored_dilated = cv2.dilate(explored_binary, kernel, iterations=2)
        frontier_mask = explored_dilated - explored_binary

        frontier_points = np.where(frontier_mask > 0)
        frontiers = []

        current_pos_grid = self._global_to_grid(self.memory_nodes[-1]['position'])

        for i in range(len(frontier_points[0])):
            y, x = frontier_points[0][i], frontier_points[1][i]
            grid_pos = (x, y)
            global_pos = self._grid_to_global(grid_pos)
            grid_dist = np.sqrt((current_pos_grid[0] - x)**2 + (current_pos_grid[1] - y)**2)
            frontiers.append((global_pos, grid_dist))

        frontiers.sort(key=lambda f: f[1])
        return frontiers[:5]  

    def _grid_to_global(self, grid_pos):
        x, y = grid_pos
        resolution = self.voxel_map.shape
        dx = (x - resolution[1] // 2) / self.scale
        dz = (y - resolution[0] // 2) / self.scale
        return np.array([self.init_pos[0] + dx, self.init_pos[1], self.init_pos[2] + dz])

    def _check_revisiting(self):
        if len(self.memory_nodes) < 10:
            return False

        recent_positions = [node['position'] for node in self.memory_nodes[-20:]]

        grid_positions = [tuple(self._global_to_grid(pos)) for pos in recent_positions]

        unique_positions = set(grid_positions)
        diversity_ratio = len(unique_positions) / len(grid_positions)

        position_counts = {}
        for pos in grid_positions:
            position_counts[pos] = position_counts.get(pos, 0) + 1
        
        max_revisits = max(position_counts.values()) if position_counts else 0

        return diversity_ratio < 0.4 or max_revisits > 4

    def _has_target_memory_clues(self, goal):
        if not hasattr(self, 'memory_nodes') or not self.memory_nodes:
            return False

        goal_name = goal['name'] if isinstance(goal, dict) else str(goal)
        goal_name = goal_name.lower()

        target_keywords = [goal_name]
        if goal_name == "tv_monitor":
            target_keywords.extend(["tv", "television", "monitor", "screen"])
        elif goal_name == "sofa":
            target_keywords.extend(["couch", "settee"])
        elif goal_name == "chair":
            target_keywords.extend(["seat", "stool"])
        elif goal_name == "plant":
            target_keywords.extend(["Dried/artificial floral arrangement"])        
        for node in self.memory_nodes:
            if 'caption' in node:
                caption = node['caption'].lower()
                for keyword in target_keywords:
                    if keyword in caption:
                        negation_terms = ["no", "not", "don't see", "cannot see", "couldn't find", 
                                        "没有", "看不到", "找不到", "无法看到"]
                        keyword_pos = caption.find(keyword)

                        context_before = caption[max(0, keyword_pos-20):keyword_pos]
                        is_negated = any(neg in context_before for neg in negation_terms)
                        
                        if not is_negated:
                            return True
        
        return False

    def _get_object_visual_features(self, object_name):
        features = {
            "bed": "Large rectangular furniture with mattress, usually of low height, possibly with pillows and covers",
            "toilet": "White porcelain unit, round or oval seat, with water tank",
            "sofa": "large upholstered seat, usually with armrests and backrests, accommodating more than one person",
            "chair": "Single seat with backrest, possibly with armrests, various materials",
            "tv_monitor": "Rectangular screen, may be on a cabinet or wall mounted, black or gray border"
        }
        return features.get(object_name, "")

    def _score_action_with_search_history(self, action_position, distance, angle):
        search_score = 1.0
        
        x = action_position[0] + distance * np.sin(angle)
        z = action_position[2] - distance * np.cos(angle)
        target_pos = [x, action_position[1], z]

        area_id = self._get_area_id(target_pos)

        if area_id in self.searched_areas:
            area_info = self.searched_areas[area_id]
            
            if area_info["found_target"]:

                search_score = 1.5
            else:

                steps_since_search = self.step_ndx - area_info["timestamp"]
                if steps_since_search < 30:
                    search_score = 0.3  
                elif steps_since_search < 60:
                    search_score = 0.6  
                else:
                    search_score = 0.8 
        
        return search_score

    def _enhanced_visual_analysis(self, image, goal):

        goal_name = goal['name'] if isinstance(goal, dict) else goal
        
        analysis_prompt = (
            f"Analyze this image carefully for any signs of {goal_name} or paths leading to it. "
            f"Identify any indicators of {goal_name}'s location - doorways, room types, relevant objects. "
            f"Return: {{\"visible\": <0-1>, \"partial\": <0-1>, \"direction\": \"description\", \"confidence\": <0-1>}}"
        )
        
        try:
            response = self.actionVLM.call([image], analysis_prompt)
            result = self._eval_response(response)
            return result
        except:
            return {"visible": 0, "partial": 0, "direction": "unknown", "confidence": 0}

    def _generate_refined_actions(self, direction, base_actions):
        refined = []
        
        candidate_action = None
        direction_lower = direction.lower()
        
        for mag, theta in base_actions:
            action_dir = self._angle_to_direction(theta)
            if action_dir.lower() in direction_lower:
                candidate_action = (mag, theta)
                break

        if not candidate_action:
            return []

        base_mag, base_theta = candidate_action

        refined.append((base_mag * 0.6, base_theta)) 
        refined.append((base_mag * 0.8, base_theta + 0.1))
        refined.append((base_mag * 0.8, base_theta - 0.1)) 
        
        return refined

    def _choose_action(self, obs: dict):
        logging_data = {}
        
        agent_state = obs['agent_state']
        goal = obs['goal']

        a_final, images, step_metadata, stopping_response = self._run_threads(obs, [obs['color_sensor']], goal)
        step_metadata['object'] = goal

        do_visual_analysis = False
        visual_analysis = None

        if self.enable_visual_analysis and hasattr(self, '_enhanced_visual_analysis'):
            current_area_id = self._get_area_id(agent_state.position)
            if self.last_analyzed_area != current_area_id:
                do_visual_analysis = True
                logging.info(f"visual anaylysis：area ({self.last_analyzed_area} -> {current_area_id})")
                self.last_analyzed_area = current_area_id

            if self.step_ndx - self.last_visual_analysis_step >= self.visual_analysis_interval:
                do_visual_analysis = True
                logging.info(f"visual anaylysis (step after prev: {self.step_ndx - self.last_visual_analysis_step}steps)")

            if step_metadata['called_stopping'] and not current_area_id in self.searched_areas:
                do_visual_analysis = True
                logging.info("visual anaylysis: target")

            if current_area_id in self.visual_analysis_cache:
                cache_entry = self.visual_analysis_cache[current_area_id]
                if self.step_ndx - cache_entry['timestamp'] < self.visual_analysis_cache_ttl:
                    visual_analysis = cache_entry['result']
                    logging.info(f"visual anaylysis before: (stored before{self.step_ndx - cache_entry['timestamp']}steps)")
                    do_visual_analysis = False

            if do_visual_analysis:
                try:
                    visual_analysis = self._enhanced_visual_analysis(obs['color_sensor'], goal)
                    step_metadata['visual_analysis'] = visual_analysis
                    self.last_visual_analysis_step = self.step_ndx

                    self.visual_analysis_cache[current_area_id] = {
                        'result': visual_analysis,
                        'timestamp': self.step_ndx
                    }
                    
                    if visual_analysis and visual_analysis.get('visible', 0) > 0.6:
                        self._mark_area_as_searched(agent_state.position, found_target=True)

                    logging.info(f"visual anaylysis: visible={visual_analysis.get('visible', 0):.2f}, confidence={visual_analysis.get('confidence', 0):.2f}")
                except Exception as e:
                    logging.error(f"visual anaylysis wrong: {e}")
                    visual_analysis = {"visible": 0, "partial": 0, "direction": "unknown", "confidence": 0}
            else:
                if visual_analysis is None: 
                    visual_analysis = {"visible": 0, "partial": 0, "direction": "unknown", "confidence": 0}
                
            step_metadata['visual_analysis'] = visual_analysis

        current_frame_detected_target = step_metadata['called_stopping']

        if self.precision_mode and not current_frame_detected_target:
            self._mark_area_as_searched(agent_state.position, found_target=False)
            logging.info("Target no longer visible - deactivating precision navigation mode")
            self.precision_mode = False

        if len(self.stopping_calls) >= 2 and self.stopping_calls[-2] == self.step_ndx - 1:
            self._mark_area_as_searched(agent_state.position, found_target=True)
            step_metadata['action_number'] = -1
            agent_action = PolarAction.stop
            logging_data = {'REASON': 'Model confirmed stop'}

            if self.precision_mode:
                logging.info("Navigation complete - deactivating precision navigation mode")
                self.precision_mode = False

        elif not self.precision_mode and len(self.stopping_calls) >= 1 and self.stopping_calls[-1] == self.step_ndx - 1:
            self._mark_area_as_searched(agent_state.position, found_target=True)
            logging.info(f"Activating precision navigation mode (step length scaled to {self.precision_scale})")
            self.precision_mode = True
            step_metadata['precision_mode'] = True

            if hasattr(self, '_generate_refined_actions') and visual_analysis and visual_analysis.get('direction', '') != 'unknown':

                refined_actions = self._generate_refined_actions(
                    visual_analysis['direction'], 
                    list(a_final) if isinstance(a_final, dict) else a_final
                )
                if refined_actions:
                    logging.info("Generated precision actions based on visual analysis")
                    if isinstance(a_final, dict):
                        for i, action in enumerate(refined_actions[:2]):
                            key = i + 1
                            a_final[(action[0], action[1])] = key
                    else:
                        for i, action in enumerate(refined_actions[:2]):
                            if i+1 < len(a_final):
                                a_final[i+1] = action

            step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
            agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))
        
        else:

            stuck_in_area = False
            if self.step_ndx > 30:
                stuck_in_area = self._is_stuck_in_small_area(goal)
            
            if stuck_in_area:
                logging.warning(f"detect stuck, escape (step: {self.step_ndx})")
                agent_action = self._generate_escape_from_area()
                step_metadata['action_number'] = 999 
                logging_data = {'REASON': 'Forced escape from stuck area'}
            else:
                if self.step_ndx % 10 == 0:
                    self._mark_area_as_searched(agent_state.position, found_target=False)
                if self.step_ndx > 500 and hasattr(self, 'memory_nodes') and len(self.memory_nodes) >= 8:

                    exploration = self._evaluate_exploration_status()
                    is_revisiting = self._check_revisiting()

                    has_target_memory = self._has_target_memory_clues(goal)

                    is_sufficient = bool(exploration['sufficient'])
                    is_stagnant = bool(exploration['stagnant'])
                    has_memory = bool(has_target_memory)
                    is_revisit = bool(is_revisiting)
                    if ((is_sufficient and is_stagnant and not has_memory) or is_revisit):
                        logging.info(f"Stopping search - Exploration: {exploration['coverage']:.2f}, Has clues: {has_target_memory}, Revisiting: {is_revisiting}")
                        
                        step_metadata['action_number'] = -1
                        agent_action = PolarAction.stop
                        logging_data = {'REASON': 'Exploration complete or repeated area'}

                        metadata = {
                            'step_metadata': step_metadata,
                            'logging_data': logging_data,
                            'a_final': a_final,
                            'images': images
                        }
                        return agent_action, metadata

                step_metadata, logging_data, _ = self._prompting(goal, a_final, images, step_metadata)
                agent_action = self._action_number_to_polar(step_metadata['action_number'], list(a_final))

                if hasattr(self, '_is_agent_stuck') and self._is_agent_stuck():
                    logging.warning("detect stuck, escape")
                    if hasattr(self, '_get_escape_action'):
                        agent_action = self._get_escape_action()
                        logging_data['STUCK_ESCAPE'] = "Used escape strategy"

        logging_data['STOPPING RESPONSE'] = stopping_response
        
        if 'visual_analysis' in step_metadata:
            logging_data['VISUAL_ANALYSIS'] = step_metadata['visual_analysis']

        if hasattr(self, 'last_visual_analysis_step'):
            analysis_stats = {
                'last_analysis': self.last_visual_analysis_step,
                'steps_since_analysis': self.step_ndx - self.last_visual_analysis_step,
                'cache_entries': len(self.visual_analysis_cache) if hasattr(self, 'visual_analysis_cache') else 0
            }
            logging_data['ANALYSIS_STATS'] = analysis_stats

        if hasattr(self, 'stuck_areas') and self.stuck_areas:
            metadata['stuck_areas'] = self.stuck_areas
        
        metadata = {
            'step_metadata': step_metadata,
            'logging_data': logging_data,
            'a_final': a_final,
            'images': images
        }
        return agent_action, metadata
        
    def _get_action_directions(self, a_final):
        try:
            directions = []
            for i, (dist, angle) in enumerate(a_final):
                action_num = i + 1
                direction = self._angle_to_direction(angle)
                directions.append(f"Action {action_num}: {direction} ({dist:.1f}m)")
            return "\n".join(directions)
        except:
            return ""
            
    def _angle_to_direction(self, angle):
        """将弧度角转换为方向描述"""
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        
        if angle > -np.pi/8 and angle < np.pi/8:
            return "forward"
        elif angle >= np.pi/8 and angle < 3*np.pi/8:
            return "forward-right"
        elif angle >= 3*np.pi/8 and angle < 5*np.pi/8:
            return "right"
        elif angle >= 5*np.pi/8 and angle < 7*np.pi/8:
            return "backward-right"
        elif angle >= 7*np.pi/8 or angle < -7*np.pi/8:
            return "backward"
        elif angle >= -7*np.pi/8 and angle < -5*np.pi/8:
            return "backward-left"
        elif angle >= -5*np.pi/8 and angle < -3*np.pi/8:
            return "left"
        else:
            return "forward-left"

    def _construct_prompt(self, goal: str, prompt_type:str, num_actions: int=0, has_memory: bool=False):
        goal_name = goal['name'] if isinstance(goal, dict) else str(goal)
        if prompt_type == 'stopping':
            obj_features = self._get_object_visual_features(goal_name) 
            stopping_prompt = (f"The agent has has been tasked with navigating to a {goal_name.upper()}. The agent has sent you an image taken from its current location,the {goal_name} typically looks like: {obj_features}"
            f'Your job is to determine whether the agent is VERY CLOSE to a {goal_name}. And you have to be careful that it must be very close and that there are no obstacles in the middle 0.1m distance blocking it, or else it will be FALSE POSITIVE. '
            f"First, tell me what you see in the image, and tell me if there is a {goal_name}. Second, return 1 if the agent is VERY CLOSE to the {goal_name} - make sure the object you see is ACTUALLY a {goal_name}, Return 0 if if there is no {goal_name}, or if it is far away, or if you are not sure. Format your answer in the json {{'done': <1 or 0>}}")
            return stopping_prompt
        if prompt_type == 'no_project':
            baseline_prompt = (f"TASK: NAVIGATE TO THE NEAREST {goal_name.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
                        "You have four possible actions: {0: Turn completely around, 1: Turn left, 2: Move straight ahead, 3: Turn right}. "
                        f"First, tell me what you see in your sensor observation, and if you have any leads on finding the {goal_name.upper()}. Second, tell me which general direction you should go in. "
                        f"Lastly, explain which action acheives that best, and return it as {{'action': <action_key>}}. Note you CANNOT GO THROUGH CLOSED DOORS, and you DO NOT NEED TO GO UP OR DOWN STAIRS"             
            )
            return baseline_prompt
        if prompt_type == 'pivot':
            pivot_prompt = f"NAVIGATE TO THE NEAREST {goal_name.upper()} and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. "
            return pivot_prompt

        if prompt_type == 'action':
            action_prompt = (
                f"TASK: NAVIGATE TO THE NEAREST {goal_name.upper()}, and get as close to it as possible. Use your prior knowledge about where items are typically located within a home. Please note that a chair and a sofa are not the same thing."
                f"There are {num_actions - 1} red arrows superimposed onto your observation, which represent potential actions. " 
                f"These are labeled with a number in a white circle, which represent the location you would move to if you took that action. "
                f"{'NOTE: choose action 0 if you want to TURN AROUND or DONT SEE ANY GOOD ACTIONS. ' if self.step_ndx - self.turned >= self.cfg['turn_around_cooldown'] else ''}"
                "Let's solve this navigation task step by step:\n"
                "1. Current State: What do you observe in the environment? What objects and pathways are visible? Look carefully for the target object, even if it's partially visible or at a distance.\n"
                f"2. Goal Analysis: Based on the target and home layout knowledge, where is the {goal_name} likely to be?\n"
            )
            if has_memory:
                action_prompt += (
                    "3. Memory Integration: Review the memory context below for clues about target location.\n"
                    "   - Pay special attention to memories containing or near the target object\n"
                    "   - Use recent memories (fewer steps ago) over older ones\n"
                    "   - Consider action recommendations based on memory\n\n"
                )
            action_prompt += (
            f"{'4' if has_memory else '3'}. Scene Assessment: Quickly evaluate if {goal_name} could reasonably exist in this type of space:\n"
            f"   - If you're in an obviously incompatible room (e.g., looking for a {goal_name} but in a clearly different room type), choose action 0 to TURN AROUND immediately\\n"
            )
            action_prompt += (
                f"{'5' if has_memory else '4'}. Path Planning: What's the most promising direction to reach the target? Avoid revisiting previously explored areas unless necessary. Consider:\n"
                "   - Available paths and typical room layouts\n"
                "   - Areas you haven't explored yet\n"
                f"{'6' if has_memory else '5'}. Action Decision: Which numbered arrow best serves your plan? Return your choice as {{'action': <action_key>}}. Note:\n"
                "   - You CANNOT GO THROUGH CLOSED DOORS, It doesn't make any sense to go near a closed door.\n"
                "   - You CANNOT GO THROUGH WINDOWS AND MIRRORS\n"
                "   - You DO NOT NEED TO GO UP OR DOWN STAIRS\n"
                f"   - Please try to avoid actions that will lead you to a dead end to avoid affecting subsequent actions, unless the dead end is very close to the {goal_name} \n"
                "   - If you see the target object, even partially, choose the action that gets you closest to it\n"
            )
            if has_memory and hasattr(self, 'searched_areas') and self.searched_areas:
                action_prompt += (
                    "- IMPORTANT: Prioritize actions that lead to UNEXPLORED areas\n"
                    "- Avoid revisiting areas that have been thoroughly searched without finding the target\n"
                    "- If you see evidence of the target, prioritize getting closer even if the area was visited before\n"
                )

            return action_prompt

        raise ValueError('Prompt type must be stopping, pivot, no_project, or action')

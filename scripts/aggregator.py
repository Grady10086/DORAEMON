import traceback
import threading
import time
import argparse
import wandb
import sys
import os
from flask import Flask, request, jsonify
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取 scripts 目录
project_root = os.path.dirname(current_dir)  # 获取项目根目录
src_path = os.path.join(project_root, 'src')  # 获取 src 目录路径
sys.path.insert(0, src_path)  # 添加到 Python 路径
from utils import *

# Initialize Flask app
app = Flask(__name__)

# Threading components
lock = threading.Lock()
terminate_event = threading.Event()

# Aggregated metrics
episode_data = []  # Stores episode data
episodes_completed = set()  # Tracks completed episodes
cumulative_metrics = {'episodes_completed': 0}  # Tracks cumulative metrics for all episodes
total_episodes = [1]  # Total episodes to be completed
spend_per_instance = {}  # Stores spend per instance
task_log = {}  # Logs data for each task

@app.route('/terminate', methods=['POST'])
def terminate():
    """Endpoint to receive termination signal and shutdown the server."""
    with lock:
        print("Received termination signal.")
    terminate_event.set()
    logging_thread.join()
    return jsonify({'status': 'terminating'}), 200

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点，返回当前状态"""
    return jsonify({
        'status': 'running',
        'episodes_completed': len(episodes_completed),
        'instances_connected': len(spend_per_instance)
    }), 200
    

@app.route('/log', methods=['POST'])
def log_metrics():
    """Endpoint to log metrics for each instance."""
    data = request.json
    required_keys = ['instance', 'episode_ndx', 'total_episodes', 'spend', 'task', 'task_data']
    
    # Check for missing keys
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        return jsonify({
            'status': 'error',
            'message': f'Missing key(s): {", ".join(missing_keys)}'
        }), 400

    # Log the data within a lock to ensure thread safety
    with lock:
        instance = data['instance']
        spend_per_instance[instance] = data['spend']
        total_episodes[0] = data['total_episodes']

        # Log task-specific data
        task = data['task']
        task_log.setdefault(task, []).append(data['task_data'])

        # Log unique episode data
        if data['episode_ndx'] not in episodes_completed:
            episodes_completed.add(data['episode_ndx'])
            episode_data.append(data)
            cumulative_metrics['episodes_completed'] += 1

            # Update cumulative metrics with the new episode data
            for key, value in data.items():
                if key not in required_keys:
                    cumulative_metrics[key] = cumulative_metrics.get(key, 0) + value

    return jsonify({'status': 'success'}), 200


def log_task_data():
    """Logs task-specific data to WandB."""
    for task, task_entries in task_log.items():
        if task.lower() == 'goat':
            goal_rows = [goal for episode in episode_data for goal in episode['task_data'].get('goal_data', [])]

            # Compute goal-based metrics
            out_log = {
                'goals_completed': len(goal_rows),
                'success_rate': sum(row['goal_reached'] for row in goal_rows) / len(goal_rows),
                'spl': sum(row['spl'] for row in goal_rows) / len(goal_rows),
            }
            
            # 添加 trustworthy_score (转换 AORI，越低越好 -> 越高越好)
            if goal_rows and any('trustworthy_metrics' in row for row in goal_rows):
                out_log['trustworthy_score'] = 1.0 - sum(row.get('trustworthy_metrics', 0) 
                                                     for row in goal_rows) / len(goal_rows)
            
            wandb.log(out_log)
        # 添加 ObjectNav 处理
        elif task.lower() == 'objectnav':
            objectnav_episodes = [ep for ep in episode_data if ep['task'].lower() == 'objectnav']
            if objectnav_episodes:
                # 计算 ObjectNav 特定指标
                out_log = {
                    'objectnav_episodes': len(objectnav_episodes),
                    'objectnav_success_rate': sum(1 for ep in objectnav_episodes if ep.get('goal_reached', False)) / len(objectnav_episodes),
                    'objectnav_spl': sum(ep.get('spl', 0) for ep in objectnav_episodes) / len(objectnav_episodes),
                }
                
                # 添加 AORI 指标
                if any('trustworthy_metrics' in ep for ep in objectnav_episodes):
                    aori_values = [ep.get('trustworthy_metrics', 0.0) for ep in objectnav_episodes]
                    out_log['objectnav_trustworthy_score'] = 1.0 - sum(aori_values) / len(objectnav_episodes)
                
                wandb.log(out_log)


def wandb_logging(sleep_interval):
    """Thread that handles logging data to WandB at regular intervals."""
    while not terminate_event.is_set():
        time.sleep(sleep_interval)
        with lock:
            # Log aggregated data
            total_spend = sum(spend_per_instance.values())
            out_data = {
                'total_spend': total_spend,
                'episodes_completed': cumulative_metrics['episodes_completed'],
                'progress': cumulative_metrics['episodes_completed'] / total_episodes[0],
            }
            
            # Log additional metrics per episode
            for key, value in cumulative_metrics.items():
                if key != 'episodes_completed' and cumulative_metrics['episodes_completed'] > 0:
                    if key == 'trustworthy_metrics':
                        # AORI 是越低越好，转换为越高越好的 trustworthy_score
                        out_data['trustworthy_score'] = 1.0 - (value / cumulative_metrics['episodes_completed'])
                    else:
                        out_data[key] = value / cumulative_metrics['episodes_completed']
            
            wandb.log(out_data)
            
            # Log task-specific data
            try:
                log_task_data()
            except Exception as e:
                traceback.print_exc()
                print(f"Error logging task data: {e}")

            print("Logged to WandB.")
    # Final log when terminating
    time.sleep(1)
    with lock:
        total_spend = sum(spend_per_instance.values())
        out_data = {
            'total_spend': total_spend,
            'episodes_completed': cumulative_metrics['episodes_completed'],
            'progress': cumulative_metrics['episodes_completed'] / total_episodes[0],
        }
        for key, value in cumulative_metrics.items():
            if key != 'episodes_completed' and value > 0:
                if key == 'trustworthy_metrics':
                    # AORI 是越低越好，转换为越高越好的 trustworthy_score
                    out_data['trustworthy_score'] = 1.0 - (value / cumulative_metrics['episodes_completed'])
                else:
                    out_data[key] = value / cumulative_metrics['episodes_completed']
        wandb.log(out_data)
        print("Final log to WandB.")

    print("WandB logging thread terminating.")
    wandb.finish()
    print("Aggregator has shut down.")
    exit(0)


if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Aggregator for Parallel Workers')
    parser.add_argument('--name', type=str, required=True, help='Name for the WandB run group')
    parser.add_argument('--sleep', type=int, default=10, help='Sleep interval between WandB logs')
    parser.add_argument('--port', type=int, default=5000, help='Port number for the Flask server')
    args = parser.parse_args()

    # Initialize WandB
    task_group = args.name.split('_')[0]
    wandb.init(project='1V1EPO', group=task_group, name=args.name)
    print('Initialized WandB.')

    # Start WandB logging in a separate thread
    logging_thread = threading.Thread(target=wandb_logging, daemon=True, args=(args.sleep,))
    logging_thread.start()

    # Run Flask app
    try:
        app.run(host='0.0.0.0', port=args.port)
    except KeyboardInterrupt:
        print("Aggregator received KeyboardInterrupt. Shutting down.")
    finally:
        terminate_event.set()
        logging_thread.join()
        print("Aggregator has shut down.")
        exit(0)

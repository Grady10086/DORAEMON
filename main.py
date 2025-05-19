import argparse
import yaml
import sys
import os
from dotenv import load_dotenv
current_dir = os.path.dirname(os.path.abspath(__file__))  # 获取 scripts 目录
print('current_dir', current_dir)
project_root = os.path.dirname(current_dir)  # 获取项目根目录
src_path = os.path.join(project_root, 'src')  # 获取 src 目录路径
sys.path.insert(0, src_path)  # 添加到 Python 路径

from agent import *
from vlm import *
from env import *

"""# 确保只使用 GPU 2
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# 检查当前使用的 GPU
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
print('Current device:', torch.cuda.current_device())
print('Device name:', torch.cuda.get_device_name(torch.cuda.current_device()"""
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="Run a dynamic envmark")

    parser.add_argument('--config', type=str, default='ObjectNav', help='name of the YAML config file')
    parser.add_argument('-n', '--name', type=str, help='Name for the run (optional)')
    parser.add_argument('-lf', '--log_freq', type=int, help='Logging frequency (optional)')
    parser.add_argument('-ms', '--max_steps', type=int, help='Max steps per episode (optional)')
    parser.add_argument('-ne', '--num_episodes', type=int, help='Number of episodes to run (optional)')
    parser.add_argument('-pa', '--parallel', action='store_true', help='Enable parallel execution')
    parser.add_argument('--instances', type=int, help='Number of instances for parallel execution (optional)')
    parser.add_argument('--instance', type=int, help='Instance number for parallel execution (optional)')
    parser.add_argument('--port', type=int, help='port number for Flask server parallel execution (optional)')
    parser.add_argument('--start_from', type=int, default=0, 
                       help='Start from which instance (for resuming interrupted runs)')
    parser.add_argument('--total_instances', type=int, default=1000,
                       help='Total number of instances to run')

    args = parser.parse_args()

    # 检查已完成的实例
    log_dir = f"logs/ObjectNav_{args.name}"
    completed_instances = set()
    if os.path.exists(log_dir):
        for filename in os.listdir(log_dir):
            if filename.endswith('.txt'):
                instance_num = int(filename.split('_of_')[0])
                completed_instances.add(instance_num)
    
    # 打印进度信息
    print(f"Found {len(completed_instances)} completed instances")
    print(f"Starting from instance {args.start_from}")
    print(f"Target total instances: {args.total_instances}")

    # Load configuration from YAML file
    with open(f'config/{args.config}.yaml', 'r') as file:
        config = yaml.safe_load(file)
    # Override YAML config with command-line arguments if provided
    if args.name is not None:
        config['env_cfg']['name'] = args.name
    if args.log_freq is not None:
        config['env_cfg']['log_freq'] = args.log_freq
    if args.max_steps is not None:
        config['env_cfg']['max_steps'] = args.max_steps
    if args.num_episodes is not None:
        config['env_cfg']['num_episodes'] = args.num_episodes
    if args.instances is not None:
        config['env_cfg']['instances'] = args.instances
    if args.instance is not None:
        config['env_cfg']['instance'] = args.instance
    if args.port is not None:
        config['env_cfg']['port'] = args.port
    if args.parallel:
        config['env_cfg']['parallel'] = True

    # 添加断点续跑相关配置
    config['env_cfg']['start_from'] = args.start_from
    config['env_cfg']['total_instances'] = args.total_instances
    config['env_cfg']['completed_instances'] = completed_instances

    env_cls = globals()[config['env_cls']]
    env = env_cls(cfg=config)
    env.run_experiment()

if __name__ == '__main__':
    main()

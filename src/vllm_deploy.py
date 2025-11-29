#!/usr/bin/env python3
"""
Deploy vLLM workers with Nginx load balancer.

Usage:
    python src/vllm_deploy.py -m Qwen/Qwen3-4B-Instruct-2507
    python src/vllm_deploy.py -m /path/to/model --gpus 01,23,45,67
    python src/vllm_deploy.py -m my-model --gpus 01,23 --port 8080 --start-port 8000
"""

import os
import subprocess
import shutil
import sys
import atexit
import time
import socket
import argparse
import requests
from tabulate import tabulate


def parse_gpu_groups(gpu_str: str) -> list[str]:
    """
    Parse GPU string like '01,23,45,67' into ['0,1', '2,3', '4,5', '6,7'].
    Also supports '0,1,2,3' format (single digits separated by commas become individual groups).
    """
    groups = gpu_str.split(',')
    result = []
    for group in groups:
        group = group.strip()
        if len(group) > 1 and ',' not in group:
            # Multi-digit like '01' or '23' -> '0,1' or '2,3'
            result.append(','.join(list(group)))
        else:
            # Single GPU or already formatted
            result.append(group)
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description='Deploy vLLM workers with Nginx load balancer',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -m Qwen/Qwen3-4B-Instruct-2507
  %(prog)s -m /path/to/model --gpus 01,23,45,67
  %(prog)s -m my-model --gpus 01,23 --port 8080 --start-port 8000
  %(prog)s -m my-model --gpus 0,1 --tp 1  # Single GPU per worker
        """
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='Qwen/Qwen3-4B-Instruct-2507',
        help='Model name or path (default: Qwen/Qwen3-4B-Instruct-2507)'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default='01,23,45,67',
        help='GPU groups, e.g., "01,23,45,67" -> ["0,1", "2,3", "4,5", "6,7"] (default: 01,23,45,67)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8080,
        help='Nginx load balancer port (default: 8080)'
    )
    parser.add_argument(
        '--start-port',
        type=int,
        default=8000,
        help='Starting port for vLLM workers (default: 8000)'
    )
    parser.add_argument(
        '--tp',
        type=int,
        default=2,
        help='Tensor parallelism (default: 2)'
    )
    parser.add_argument(
        '--vllm-bin',
        type=str,
        default='/home/anh/projects/lighteval/.venv/bin/vllm',
        help='Path to vLLM binary (default: /home/anh/projects/lighteval/.venv/bin/vllm)'
    )
    parser.add_argument(
        '--extra-args',
        type=str,
        default='--reasoning-parser qwen3',
        help='Extra arguments to pass to vLLM (default: --reasoning-parser qwen3)'
    )
    return parser.parse_args()


# Parse arguments
ARGS = parse_args()

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
# Ports
NGINX_PORT = ARGS.port
START_PORT = ARGS.start_port

# GPU Allocation
GPU_GROUPS = parse_gpu_groups(ARGS.gpus)

# Logging
LOG_DIR = "/tmp/vllm_py"
NGINX_CONF = os.path.join(LOG_DIR, "nginx_vllm.conf")
NGINX_LOG_OUT = os.path.join(LOG_DIR, "nginx_stdout.log")
NGINX_LOG_ERR = os.path.join(LOG_DIR, "nginx_error.log")

# Stats tracking
STATS_INTERVAL = 5  # seconds between stats refresh
WORKER_STATS = {}  # port -> {'total_processed': 0, 'current_processing': 0}

# ==============================================================================
# 2. UTILITIES
# ==============================================================================

RUNNING_PROCESSES = []

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def cleanup_processes():
    print("\n🧹 Shutting down processes...")
    for proc in RUNNING_PROCESSES:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), 15) # SIGTERM
            except OSError:
                pass
    print("✅ Cleanup complete.")

atexit.register(cleanup_processes)


def get_worker_stats(port: int) -> dict:
    """Fetch stats from vLLM worker's /metrics endpoint."""
    try:
        resp = requests.get(f'http://127.0.0.1:{port}/metrics', timeout=2)
        if resp.status_code == 200:
            text = resp.text
            running = 0
            total = 0
            for line in text.split('\n'):
                if line.startswith('vllm:num_requests_running{'):
                    running = int(float(line.split()[-1]))
                elif line.startswith('vllm:request_success_total{'):
                    total = int(float(line.split()[-1]))
            return {'current_processing': running, 'total_processed': total}
    except Exception:
        pass
    return {'current_processing': '-', 'total_processed': '-'}


def print_stats_table(worker_ports: list):
    """Print a formatted table of worker stats."""
    rows = []
    for port in worker_ports:
        stats = get_worker_stats(port)
        rows.append([port, stats['current_processing'], stats['total_processed']])
    
    # Clear screen and print table
    print('\033[2J\033[H', end='')  # Clear screen and move cursor to top
    print('=' * 60)
    print('vLLM Worker Stats (Press Ctrl+C to stop)')
    print('=' * 60)
    headers = ['Port', 'Current Processing', 'Total Processed']
    print(tabulate(rows, headers=headers, tablefmt='grid'))
    print(f'\nLogs directory: {LOG_DIR}')
    print(f'Load balancer: http://127.0.0.1:{NGINX_PORT}')


def check_requirements():
    # 1. Check Nginx
    if not shutil.which("nginx"):
        print("❌ Error: 'nginx' binary not found. Please install it (sudo apt install nginx).")
        sys.exit(1)
        
    # 2. Check Ports
    if is_port_in_use(NGINX_PORT):
        print(f"❌ Error: Load Balancer Port {NGINX_PORT} is already in use.")
        print(f"   Try: 'lsof -i :{NGINX_PORT}' to see what's running.")
        sys.exit(1)
        
    for i in range(len(GPU_GROUPS)):
        port = START_PORT + i
        if is_port_in_use(port):
            print(f"❌ Error: Worker Port {port} is already in use.")
            sys.exit(1)

# ==============================================================================
# 3. NGINX CONFIG
# ==============================================================================

def generate_nginx_config(port_list):
    upstream_block = ""
    for port in port_list:
        upstream_block += f"        server 127.0.0.1:{port};\n"

    config = f"""
worker_processes auto;
pid {LOG_DIR}/nginx.pid;
error_log {NGINX_LOG_ERR} warn;
events {{ worker_connections 1024; }}
http {{
    access_log /dev/null;
    proxy_read_timeout 1200s;
    upstream vllm_backend {{
        least_conn;
{upstream_block}
    }}
    server {{
        listen {NGINX_PORT};
        location / {{
            proxy_pass http://vllm_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_buffering off; 
        }}
    }}
}}
"""
    with open(NGINX_CONF, 'w') as f:
        f.write(config)

# ==============================================================================
# 4. MAIN EXECUTION
# ==============================================================================

def main():
    check_requirements()
    
    # Clean logs
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Start vLLM Workers
    current_port = START_PORT
    worker_ports = []
    
    print(f"🚀 Starting {len(GPU_GROUPS)} vLLM workers...")
    print(f"   Model: {ARGS.model}")
    print(f"   GPU groups: {GPU_GROUPS}")
    print(f"   Tensor parallelism: {ARGS.tp}")
    
    for gpus in GPU_GROUPS:
        log_file = os.path.join(LOG_DIR, f"worker_{current_port}.log")
        cmd = [
            ARGS.vllm_bin, "serve", ARGS.model,
            "-tp", str(ARGS.tp),
            "--port", str(current_port)
        ]
        if ARGS.extra_args:
            cmd.extend(ARGS.extra_args.split())
        
        with open(log_file, 'w') as f:
            proc = subprocess.Popen(
                cmd,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": gpus},
                stdout=f, stderr=f,
                start_new_session=True
            )
            RUNNING_PROCESSES.append(proc)
            worker_ports.append(current_port)
            current_port += 1

    # Start Nginx
    generate_nginx_config(worker_ports)
    print(f"🌐 Starting Nginx on port {NGINX_PORT}...")
    
    with open(NGINX_LOG_OUT, 'w') as f_out:
        # Nginx writes startup errors to stderr if config is bad
        nginx_proc = subprocess.Popen(
            ["nginx", "-c", NGINX_CONF, "-g", "daemon off;"],
            stdout=f_out, 
            stderr=f_out, # Capture stderr to the same log file for debugging
            start_new_session=True
        )
        RUNNING_PROCESSES.append(nginx_proc)

    # Monitoring Loop
    print("\n✅ System Running. Press Ctrl+C to stop.")
    print(f"Logs directory: {LOG_DIR}")
    
    last_stats_time = 0
    try:
        while True:
            time.sleep(1)
            
            # Print stats periodically
            current_time = time.time()
            if current_time - last_stats_time >= STATS_INTERVAL:
                print_stats_table(worker_ports)
                last_stats_time = current_time
            
            # Check if Nginx died
            if nginx_proc.poll() is not None:
                print(f"\n🚨 CRITICAL: Nginx crashed (Exit Code: {nginx_proc.returncode})")
                print(f"📄 Log files:")
                print(f"   - Stdout/Stderr: {NGINX_LOG_OUT}")
                print(f"   - Error log: {NGINX_LOG_ERR}")
                print("--- Nginx Log Tail ---")
                
                # Print the reason Nginx died
                if os.path.exists(NGINX_LOG_OUT):
                    os.system(f"tail -n 10 {NGINX_LOG_OUT}")
                if os.path.exists(NGINX_LOG_ERR):
                     os.system(f"tail -n 10 {NGINX_LOG_ERR}")
                break

            # Check if any vLLM worker died
            for i, proc in enumerate(RUNNING_PROCESSES[:-1]): # Exclude nginx (last one)
                if proc.poll() is not None:
                    worker_port = worker_ports[i]
                    worker_log = os.path.join(LOG_DIR, f"worker_{worker_port}.log")
                    print(f"\n🚨 CRITICAL: vLLM Worker {i} (port {worker_port}) crashed.")
                    print(f"📄 Log file: {worker_log}")
                    print("--- Worker Log Tail ---")
                    if os.path.exists(worker_log):
                        os.system(f"tail -n 20 {worker_log}")
                    break

    except KeyboardInterrupt:
        print("\nStopping...")

if __name__ == "__main__":
    main()

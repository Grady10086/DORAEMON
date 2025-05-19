export GEMINI_API_KEY=
# 修改cleanup函数
cleanup() {
    echo "Caught interrupt signal. Cleaning up tmux sessions..."
    if [ -n "${SESSION_NAMES[*]}" ]; then
        for session in "${SESSION_NAMES[@]}"; do
            if [ -n "$session" ] && tmux has-session -t "$session" 2>/dev/null; then
                echo "Killing session: $session"
                tmux kill-session -t "$session"
            fi
        done
    fi
}

# 定义运行实验的函数
run_experiment() {
    local START_FROM=$1
    local NAME=$2
    local GPU_DEVICE=$3
    mkdir -p logs/${TASK}_${NAME}/
    # Configuration Variables
    NUM_GPU=4
    INSTANCES=8
    NUM_EPISODES_PER_INSTANCE=1
    MAX_STEPS_PER_EPISODE=500
    TOTAL_INSTANCES=1000
    TASK="ObjectNav"
    CFG="ObjectNav_15_pro"
    SLEEP_INTERVAL=10
    LOG_FREQ=1
    PORT=$(shuf -i 6000-7000 -n 1)
    VENV_NAME="vlm_nav"
    
    # 定义CMD变量
    CMD="python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT} --start_from ${START_FROM} --total_instances ${TOTAL_INSTANCES}"
    
    # Set memory limits
    ulimit -v 100000000  # Virtual memory limit: 100GB
    ulimit -m 80000000   # Physical memory limit: 80GB
    
    echo "=== Starting experiment: $NAME ==="
    echo "Number of instances: $INSTANCES"
    echo "Episodes per instance: $NUM_EPISODES_PER_INSTANCE"
    echo "Starting from episode: $START_FROM"
    echo "Memory limits:"
    echo "Virtual memory: $(ulimit -v)"
    echo "Physical memory: $(ulimit -m)"
    echo "========================================"
    
    # 重置会话列表
    SESSION_NAMES=()
    AGGREGATOR_SESSION="aggregator_${NAME}"

    echo "Starting aggregator..."
    # Start Aggregator Session with error checking
    tmux new-session -d -s "$AGGREGATOR_SESSION" \
    "bash -i -c 'source activate ${VENV_NAME} && CUDA_VISIBLE_DEVICES=3 python scripts/aggregator.py --name ${TASK}_${NAME} --sleep ${SLEEP_INTERVAL} --port ${PORT} 2>&1 | tee logs/${TASK}_${NAME}/aggregator.log'"

    # Wait for aggregator to start and verify it's responding
    sleep 5
    echo "Checking if aggregator is running..."
    if ! tmux has-session -t "$AGGREGATOR_SESSION" 2>/dev/null; then
        echo "Failed to start aggregator session!"
        return 1
    fi

    # Try to connect to aggregator
    echo "Testing aggregator connection..."
    for i in {1..5}; do
        if curl -s "http://localhost:${PORT}/health" >/dev/null 2>&1; then
            echo "Aggregator is responding!"
            break
        fi
        if [ $i -eq 5 ]; then
            echo "Aggregator is not responding after 5 attempts. Exiting..."
            cleanup
            return 1
        fi
        echo "Waiting for aggregator to start (attempt $i)..."
        sleep 2
    done

    SESSION_NAMES+=("$AGGREGATOR_SESSION")

    echo "Starting worker instances..."
    # Start Tmux Sessions for Each Instance
    FAILED_SESSIONS=0
    for instance_id in $(seq 0 $((INSTANCES - 1))); do
        GPU_ID=$(( (instance_id + GPU_DEVICE) % NUM_GPU))
        SESSION_NAME="${TASK}_${NAME}_${instance_id}_${INSTANCES}"
        
        echo "Starting instance $instance_id on GPU $GPU_ID..."
        tmux new-session -d -s "$SESSION_NAME" \
            "bash -i -c 'source activate ${VENV_NAME} && CUDA_VISIBLE_DEVICES=$GPU_ID $CMD --instance $instance_id'"
        
        if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
            echo "Failed to start session $SESSION_NAME!"
            ((FAILED_SESSIONS++))
            continue
        fi
        
        SESSION_NAMES+=("$SESSION_NAME")
        # Add small delay between instance starts to prevent overwhelming the system
        sleep 2
    done

    echo "All instances started. Monitoring progress..."
    # Monitor Tmux Sessions
    LAST_ACTIVE_COUNT=-1
    STALL_COUNT=0
    while true; do
        sleep $SLEEP_INTERVAL
        
        ALL_DONE=true
        ACTIVE_SESSIONS=0
        
        for instance_id in $(seq 0 $((INSTANCES - 1))); do
            SESSION_NAME="${TASK}_${NAME}_${instance_id}_${INSTANCES}"
            if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
                ALL_DONE=false
                ((ACTIVE_SESSIONS++))
            fi
        done
        
        echo "$(date): Active sessions: $ACTIVE_SESSIONS"
        
        # Check for stalled state
        if [ $ACTIVE_SESSIONS -eq $LAST_ACTIVE_COUNT ] && [ $ACTIVE_SESSIONS -ne 0 ]; then
            ((STALL_COUNT++))
        else
            STALL_COUNT=0
        fi
        LAST_ACTIVE_COUNT=$ACTIVE_SESSIONS
        
        if $ALL_DONE; then
            echo "All workers completed!"
            break
        fi
    done
    
    echo "$(date): Sending termination signal to aggregator..."
    curl -X POST http://localhost:${PORT}/terminate
    if [ $? -eq 0 ]; then
        echo "$(date): Termination signal sent successfully."
    else
        echo "$(date): Failed to send termination signal."
    fi
    
    sleep 10
    if tmux has-session -t "$AGGREGATOR_SESSION" 2>/dev/null; then
        tmux kill-session -t "$AGGREGATOR_SESSION"
        echo "Killed aggregator session"
    fi
}

# 设置中断信号处理
trap cleanup SIGINT

run_experiment 1088 "V1_1088_1095_EPO_RAG_15pro_AA" 0
run_experiment 1368 "V1_1368_1375_EPO_RAG_15pro_AA" 2

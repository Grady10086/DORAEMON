#!/bin/bash
export GEMINI_API_KEY=sk-LPDkdkU9gEXIcOuT3EsfhuTwyjLEQp4FSdCjwPvbN9RtlMXs

# Configuration Variables
NUM_GPU=4
INSTANCES=8
NUM_EPISODES_PER_INSTANCE=25
MAX_STEPS_PER_EPISODE=40
START_FROM=0  # Continue from where we left off
TOTAL_INSTANCES=1000
TASK="ObjectNav"
CFG="ObjectNav"
NAME="0_199_EPO_EmbodiedRAG_qwentb_0414"
SLEEP_INTERVAL=10
LOG_FREQ=1
PORT=5002
VENV_NAME="vlm_nav"

# Set memory limits
ulimit -v 100000000  # Virtual memory limit: 100GB
ulimit -m 80000000   # Physical memory limit: 80GB

echo "=== Starting script with configuration ==="
echo "Number of instances: $INSTANCES"
echo "Episodes per instance: $NUM_EPISODES_PER_INSTANCE"
echo "Starting from episode: $START_FROM"
echo "Memory limits:"
echo "Virtual memory: $(ulimit -v)"
echo "Physical memory: $(ulimit -m)"
echo "========================================"


CMD="CUDA_VISIBLE_DEVICES=0 python scripts/main.py --config ${CFG} -ms ${MAX_STEPS_PER_EPISODE} -ne ${NUM_EPISODES_PER_INSTANCE} --name ${NAME} --instances ${INSTANCES} --parallel -lf ${LOG_FREQ} --port ${PORT} --start_from ${START_FROM} --total_instances ${TOTAL_INSTANCES}"

# Tmux Session Names
SESSION_NAMES=()
AGGREGATOR_SESSION="aggregator_${NAME}"

echo "Starting aggregator..."
# Start Aggregator Session with error checking
tmux new-session -d -s "$AGGREGATOR_SESSION" \
  "bash -i -c 'source activate ${VENV_NAME} && CUDA_VISIBLE_DEVICES=0 python scripts/aggregator.py --name ${TASK}_${NAME} --sleep ${SLEEP_INTERVAL} --port ${PORT} 2>&1 | tee logs/${TASK}_${NAME}/aggregator.log'"

# Wait for aggregator to start and verify it's responding
sleep 5
echo "Checking if aggregator is running..."
if ! tmux has-session -t "$AGGREGATOR_SESSION" 2>/dev/null; then
    echo "Failed to start aggregator session!"
    exit 1
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
        exit 1
    fi
    echo "Waiting for aggregator to start (attempt $i)..."
    sleep 2
done

SESSION_NAMES+=("$AGGREGATOR_SESSION")

# Cleanup Function
cleanup() {
    echo "Caught interrupt signal. Cleaning up tmux sessions..."
    for session in "${SESSION_NAMES[@]}"; do
        if tmux has-session -t "$session" 2>/dev/null; then
            echo "Killing session: $session"
            tmux kill-session -t "$session"
        fi
    done
}

# Trap SIGINT to Run Cleanup
trap cleanup SIGINT

echo "Starting worker instances..."
# Start Tmux Sessions for Each Instance
FAILED_SESSIONS=0
for instance_id in $(seq 0 $((INSTANCES - 1))); do
    GPU_ID=$((instance_id % NUM_GPU))
    SESSION_NAME="${TASK}_${NAME}_${instance_id}_${INSTANCES}"
    
    echo "Starting instance $instance_id on GPU $GPU_ID..."
    tmux new-session -d -s "$SESSION_NAME" \
        "bash -i -c 'source activate ${VENV_NAME} && CUDA_VISIBLE_DEVICES=$GPU_ID $CMD --instance $instance_id'"
    
    if ! tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "Failed to start session $SESSION_NAME!"
        ((FAILED_SESSIONS++))
        if [ $FAILED_SESSIONS -gt 5 ]; then
            echo "Too many failed sessions. Cleaning up and exiting..."
            cleanup
            exit 1
        fi
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
        if [ $STALL_COUNT -ge 5 ]; then
            echo "Warning: Process appears to be stalled. No change in active sessions for 5 checks."
            echo "You may want to check the logs or restart the process."
        fi
    else
        STALL_COUNT=0
    fi
    LAST_ACTIVE_COUNT=$ACTIVE_SESSIONS
    
    if $ALL_DONE; then
        echo "All workers completed!"
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
        break
    fi
done


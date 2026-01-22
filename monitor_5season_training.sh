#!/bin/bash
# Monitor 5-Season Batch Training Progress

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 5-SEASON TRAINING MONITOR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check if process is running
if pgrep -f "train_models.py --seasons" > /dev/null; then
    PID=$(pgrep -f "train_models.py --seasons")
    echo "✅ Training Process: RUNNING (PID: $PID)"
    
    # Get CPU and memory usage
    CPU_MEM=$(ps -p $PID -o %cpu,%mem,etime | tail -1)
    echo "   $CPU_MEM"
    echo ""
    
    # Check log file
    if [ -f training_5seasons_186features.log ]; then
        LOG_SIZE=$(ls -lh training_5seasons_186features.log | awk '{print $5}')
        LOG_LINES=$(wc -l < training_5seasons_186features.log)
        echo "📊 Log File: training_5seasons_186features.log"
        echo "   Size: $LOG_SIZE"
        echo "   Lines: $LOG_LINES"
        echo ""
        
        if [ $LOG_LINES -gt 0 ]; then
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📝 Recent Log Output (last 30 lines):"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            tail -30 training_5seasons_186features.log
            echo ""
            
            # Show training progress if available
            if grep -q '\[train\]' training_5seasons_186features.log; then
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                echo "🎯 Training Progress:"
                echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
                grep '\[train\]' training_5seasons_186features.log | tail -10
            fi
        else
            echo "⚠️  Log file is empty (Python output buffering)"
            echo "   This is normal - output will appear once buffer flushes"
        fi
    else
        echo "⚠️  Log file not found"
    fi
else
    echo "❌ Training Process: NOT RUNNING"
    echo ""
    echo "Checking for recent completion or errors..."
    if [ -f training_5seasons_186features.log ]; then
        echo ""
        echo "Last 40 lines of log:"
        tail -40 training_5seasons_186features.log
    fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "💡 Commands:"
echo "   Watch live: tail -f training_5seasons_186features.log"
echo "   Re-check:   bash monitor_5season_training.sh"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

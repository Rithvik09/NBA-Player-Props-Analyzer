#!/bin/bash
echo "========================================"
echo "🏀 TRAINING MONITOR - 153 Features"
echo "========================================"
echo ""
echo "Batch training is running in the background..."
echo "This process will:"
echo "  1. Fetch historical game data from NBA API"
echo "  2. Calculate all 153 features for each game"
echo "  3. Train Random Forest models for each prop type"
echo "  4. Save trained models to models/ directory"
echo ""
echo "Expected duration: 10-30 minutes (depending on API speed)"
echo ""

while true; do
    # Check if process is still running
    if ps aux | grep -v grep | grep "train_models.py" > /dev/null; then
        # Get process stats
        PROC_INFO=$(ps aux | grep -v grep | grep "train_models.py" | awk '{print $3 " " $4 " " $10}')
        CPU=$(echo $PROC_INFO | awk '{print $1}')
        MEM=$(echo $PROC_INFO | awk '{print $2}')
        TIME=$(echo $PROC_INFO | awk '{print $3}')
        
        echo "[$(date '+%H:%M:%S')] Training running... CPU: ${CPU}% | Memory: ${MEM}% | Time: ${TIME}"
        
        # Check for model files
        MODEL_COUNT=$(ls -1 models/*.pkl 2>/dev/null | wc -l | xargs)
        echo "                      Models saved so far: ${MODEL_COUNT}"
        
        sleep 15
    else
        echo ""
        echo "========================================"
        echo "✅ Training process completed!"
        echo "========================================"
        
        # Show results
        if [ -d "models" ]; then
            echo ""
            echo "📦 Trained models:"
            ls -lh models/*.pkl 2>/dev/null | awk '{print "   " $9 " (" $5 ")"}'
            echo ""
            echo "Total models: $(ls -1 models/*.pkl 2>/dev/null | wc -l | xargs)"
        fi
        
        # Check log file
        if [ -f "training_output.log" ] && [ -s "training_output.log" ]; then
            echo ""
            echo "📄 Training log (last 20 lines):"
            tail -20 training_output.log
        fi
        
        break
    fi
done

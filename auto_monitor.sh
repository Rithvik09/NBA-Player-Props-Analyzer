#!/bin/bash
echo "═══════════════════════════════════════════════════════════════"
echo "🏀 AUTO-MONITORING 5-SEASON TRAINING"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "This will check progress every 2 minutes and show key updates."
echo "Press Ctrl+C to stop monitoring."
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""

START_TIME=$(date +%s)
LAST_LINE=""

while true; do
    if ! ps aux | grep -v grep | grep "train_models.py" > /dev/null; then
        echo ""
        echo "═══════════════════════════════════════════════════════════════"
        echo "✅ TRAINING COMPLETED!"
        echo "═══════════════════════════════════════════════════════════════"
        
        ELAPSED=$(( $(date +%s) - START_TIME ))
        ELAPSED_MIN=$(( ELAPSED / 60 ))
        
        echo ""
        echo "Total Duration: ${ELAPSED_MIN} minutes"
        echo ""
        echo "📊 Final Results:"
        tail -30 training_5seasons.log | grep -E "\[wf\]|\[train\] done|total examples"
        echo ""
        echo "📦 Models Created:"
        ls -lh models/*.joblib 2>/dev/null | wc -l | xargs echo "  Total:"
        echo ""
        break
    fi
    
    CURRENT_LINE=$(tail -1 training_5seasons.log)
    
    if [ "$CURRENT_LINE" != "$LAST_LINE" ]; then
        ELAPSED=$(( $(date +%s) - START_TIME ))
        ELAPSED_MIN=$(( ELAPSED / 60 ))
        
        # Show season completions and every 50 players
        if echo "$CURRENT_LINE" | grep -qE "season |total examples|players processed: (50|100|150|200)"; then
            echo "[$(date '+%H:%M:%S')] Elapsed: ${ELAPSED_MIN}m | $CURRENT_LINE"
        fi
        
        LAST_LINE="$CURRENT_LINE"
    fi
    
    sleep 30
done

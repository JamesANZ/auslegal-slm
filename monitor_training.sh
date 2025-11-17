#!/bin/bash
# Monitor training progress and test query when complete

echo "Monitoring training progress..."
echo "Logs are being saved to: training_log.txt"
echo ""

while ps aux | grep -q "[t]rain_slm.py"; do
    # Get current progress
    progress=$(tail -1 training_log.txt 2>/dev/null | grep -o '[0-9]*%' | head -1 || echo "0%")
    batch_info=$(tail -1 training_log.txt 2>/dev/null | grep -o '[0-9]*/[0-9]*' | head -1 || echo "")
    loss_info=$(tail -20 training_log.txt 2>/dev/null | grep -i "loss" | tail -1 || echo "")
    
    echo "$(date '+%H:%M:%S'): Training in progress... $progress $batch_info"
    if [ ! -z "$loss_info" ]; then
        echo "  Latest: $loss_info"
    fi
    echo ""
    
    sleep 300  # Check every 5 minutes
done

echo ""
echo "=== Training Complete! ==="
echo ""

# Wait a moment for files to be written
sleep 10

# Check if model files exist
if [ -f "models/legal_slm/pytorch_model.bin" ] || [ -f "models/legal_slm/model.safetensors" ]; then
    echo "✓ Model files found"
    echo ""
    echo "=== Step 3: Testing Query Interface ==="
    echo ""
    source venv/bin/activate
    python query_slm.py --question "What is negligence in Australian law?"
    echo ""
    echo ""
    echo "=== End-to-End Test Complete! ==="
    echo ""
    echo "Summary:"
    echo "  ✓ Data preparation: Complete"
    echo "  ✓ Model training: Complete"
    echo "  ✓ Query interface: Tested"
    echo ""
    echo "Training logs saved to: training_log.txt"
    echo "Model saved to: models/legal_slm/"
else
    echo "ERROR: Model files not found after training completed"
    echo "Check training_log.txt for errors"
    exit 1
fi


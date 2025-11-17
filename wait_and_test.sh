#!/bin/bash
# Wait for training to complete, then test querying

echo "Waiting for training to complete..."
while ps aux | grep -q "[t]rain_slm.py"; do
    progress=$(tail -1 training.log 2>/dev/null | grep -o '[0-9]*%' | head -1 || echo "0%")
    echo "$(date '+%H:%M:%S'): Training in progress... $progress"
    sleep 300  # Check every 5 minutes
done

echo ""
echo "=== Training Complete! ==="
echo ""

# Wait a moment for files to be written
sleep 5

# Check if model files exist
if [ -f "models/legal_slm/pytorch_model.bin" ] || [ -f "models/legal_slm/model.safetensors" ]; then
    echo "Model files found. Testing query..."
    echo ""
    source venv/bin/activate
    python query_slm.py --question "What is negligence in Australian law?"
    echo ""
    echo "=== End-to-end test complete! ==="
else
    echo "ERROR: Model files not found after training completed"
    exit 1
fi


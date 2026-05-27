#!/bin/bash
echo "============================================"
echo "  Diamond Price Predictor - Launcher"
echo "============================================"
echo

# Install packages if not already present
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install -r requirements.txt
    echo
fi

echo "Starting app... Your browser will open automatically."
echo "Press Ctrl+C to stop the app."
echo
streamlit run streamlit_app.py

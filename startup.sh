#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "🚀 Starting Chat with PDF application setup..."

# Check if python3 is available
if ! command -v python3 &> /dev/null
then
    echo "❌ python3 could not be found. Please install Python 3."
    exit 1
fi

# Check if pip is available
if ! command -v pip &> /dev/null
then
    echo "❌ pip could not be found. Please ensure pip is installed for Python 3."
    exit 1
fi

# Optional: Create and activate a virtual environment
# echo "🐍 Creating/Activating virtual environment 'venv'..."
# python3 -m venv venv
# source venv/bin/activate

echo "📦 Installing dependencies from requirements.txt..."
if pip install -r requirements.txt; then
    echo "✅ Dependencies installed successfully."
else
    echo "❌ Failed to install dependencies. Please check requirements.txt and your Python environment."
    exit 1
fi

# Check for .env file and create a template if it doesn't exist
if [ ! -f .env ]; then
    echo "⚠️ .env file not found. Creating a template: .env"
    echo "GOOGLE_API_KEY=\"YOUR_GOOGLE_API_KEY_HERE\"" > .env
    echo "🔑 Please edit the .env file and add your Google API Key."
fi

# Check if GOOGLE_API_KEY is set and not the placeholder
if grep -q 'GOOGLE_API_KEY="YOUR_GOOGLE_API_KEY_HERE"' .env || ! grep -q 'GOOGLE_API_KEY' .env;
then
    echo ""
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo "!!! WARNING: Google API Key is not configured or is a placeholder.     !!!"
    echo "!!! Please set your GOOGLE_API_KEY in the .env file.                 !!!"
    echo "!!! The application may prompt for it or features will be limited.     !!!"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    echo ""
fi

# Ensure static directory exists
mkdir -p static

echo "🌍 Launching Streamlit application on port 9000..."
echo "🔗 Access it at http://localhost:9000 (or your server's IP if deployed)"

# Run Streamlit
streamlit run app.py --server.port 9000 --server.headless true --server.enableCORS false

echo "👋 Application startup script finished."
# Deactivate virtual environment if one was activated
# if [ -n "$VIRTUAL_ENV" ]; then
#     deactivate
# fi
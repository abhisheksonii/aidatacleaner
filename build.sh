#!/usr/bin/env bash
# exit on error
set -o errexit

# Create uploads directory if it doesn't exist
mkdir -p uploads
chmod -R 755 uploads

# Install Python dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create instance directory for Flask
mkdir -p instance
chmod -R 755 instance
#!/bin/bash
# Install system dependencies
apt-get update && apt-get install -y python3-dev

# Install Python packages
pip install --upgrade pip
pip install -r requirements.txt

# Django setup
python manage.py collectstatic --noinput
python manage.py migrate
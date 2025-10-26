#!/bin/bash

#Exit immediately if a command exits with a non-zero status.

set -e

#Install Python packages

pip install -r requirements.txt

#Run the Flask command to create the database tables

flask create-db
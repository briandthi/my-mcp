#!/bin/bash
set -e

git pull

echo "🚀 Déploiement en mode PRODUCTION (docker-compose.yml)"
docker-compose down
docker-compose up --build -d

echo "✅ Stack production lancée !"
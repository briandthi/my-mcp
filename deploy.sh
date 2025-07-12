git pull

# Delete old processes
pm2 delete "mcp"

# deploy the backend side
source venv/bin/activate
pm2 start main.py --name "mcp" --interpreter python3


# Healthcare Data Generation System - Docker Setup

This guide will help you containerize and run the Healthcare Data Generation System using Docker.

## Prerequisites

1. **Install Docker Desktop**
   - Windows/Mac: Download from [docker.com](https://www.docker.com/products/docker-desktop/)
   - Linux: Install Docker Engine and Docker Compose

2. **Verify Installation**
   ```bash
   docker --version
   docker-compose --version
   ```

## Quick Start (Recommended)

### 1. Clone the Repository
```bash
git clone https://github.com/Shrirang-Zend/Sem7_MPR.git
cd Sem7_MPR
```

### 2. Build and Run with Docker Compose
```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode (background)
docker-compose up -d --build
```

### 3. Access the Application
- **Frontend (Streamlit)**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### 4. Stop the Services
```bash
# Stop all services
docker-compose down

# Stop and remove all data
docker-compose down -v
```

## Manual Docker Commands (Alternative)

### Backend API
```bash
# Build backend image
cd backend
docker build -t healthcare-backend .

# Run backend container
docker run -p 8000:8000 healthcare-backend
```

### Frontend App
```bash
# Build frontend image
cd frontend
docker build -t healthcare-frontend .

# Run frontend container
docker run -p 8501:8501 healthcare-frontend
```

## Project Structure
```
.
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── src/
├── frontend/
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
├── docker-compose.yml
└── README.md
```

## Docker Architecture

### Services
1. **Backend (Port 8000)**
   - FastAPI application
   - Handles data generation and query processing
   - Serves API endpoints

2. **Frontend (Port 8501)**
   - Streamlit web interface
   - User interaction and visualization
   - Connects to backend API

### Networking
- Both services run on a custom bridge network
- Frontend can communicate with backend via service name
- External access via mapped ports

## Environment Variables

### Backend
- `PYTHONPATH=/app`
- `PYTHONUNBUFFERED=1`

### Frontend
- `PYTHONPATH=/app`
- `PYTHONUNBUFFERED=1`

## Data Persistence

The following directories are mounted as volumes:
- `backend/logs` - Application logs
- `backend/models` - Trained ML models
- `backend/data` - Generated data

## Health Checks

Both services include health checks:
- **Backend**: `http://localhost:8000/health`
- **Frontend**: `http://localhost:8501/_stcore/health`

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000
   lsof -i :8501
   
   # Kill the process or use different ports
   ```

2. **Docker Build Fails**
   ```bash
   # Clean Docker cache
   docker system prune -a
   
   # Rebuild without cache
   docker-compose build --no-cache
   ```

3. **Services Can't Communicate**
   ```bash
   # Check network connectivity
   docker network ls
   docker network inspect <network-name>
   ```

4. **Memory Issues**
   ```bash
   # Increase Docker memory allocation
   # Docker Desktop -> Settings -> Resources -> Memory
   ```

### Logs and Debugging

```bash
# View service logs
docker-compose logs backend
docker-compose logs frontend

# Follow logs in real-time
docker-compose logs -f backend

# Access container shell
docker-compose exec backend bash
docker-compose exec frontend bash
```

## Development Mode

For development with hot reload:

```bash
# Frontend with volume mounting
docker run -p 8501:8501 -v $(pwd)/frontend:/app healthcare-frontend

# Backend with volume mounting
docker run -p 8000:8000 -v $(pwd)/backend:/app healthcare-backend
```

## Production Deployment

For production deployment:

1. **Use environment-specific compose files**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
   ```

2. **Set resource limits**
   ```yaml
   deploy:
     resources:
       limits:
         cpus: '0.5'
         memory: 512M
   ```

3. **Configure reverse proxy** (nginx, traefik)
4. **Set up SSL certificates**
5. **Configure logging and monitoring**

## Team Sharing

### Sharing Docker Images

1. **Build and tag images**
   ```bash
   docker build -t username/healthcare-backend:latest ./backend
   docker build -t username/healthcare-frontend:latest ./frontend
   ```

2. **Push to Docker Hub**
   ```bash
   docker push username/healthcare-backend:latest
   docker push username/healthcare-frontend:latest
   ```

3. **Team members can pull and run**
   ```bash
   docker pull username/healthcare-backend:latest
   docker pull username/healthcare-frontend:latest
   ```

### Sharing Docker Compose

Share the `docker-compose.yml` file and team members can run:
```bash
docker-compose up -d
```

## Support

For issues and questions:
1. Check the logs: `docker-compose logs`
2. Verify health checks: `docker-compose ps`
3. Check network connectivity
4. Review Docker documentation

## Next Steps

1. Set up CI/CD pipeline
2. Configure production environment
3. Add monitoring and logging
4. Set up backup and recovery
5. Scale with Docker Swarm or Kubernetes

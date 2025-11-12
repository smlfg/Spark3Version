# Spark Cluster - Agent Infrastructure

Multi-agent infrastructure system for deploying and managing Spark/DGX GPU clusters with unified shared interfaces.

## Overview

This repository provides a modular, agent-based infrastructure for enterprise-grade Spark cluster deployment:

- **Shared Infrastructure**: Unified base images, configuration schema, and utilities
- **Agent System**: Specialized agents for different infrastructure layers
- **GPU Optimization**: NCCL multi-GPU communication and CUDA support
- **Remote Access**: SSH, Tailscale VPN, and secure networking
- **Containerization**: Docker-based deployment with GPU support
- **Observability**: Health checks, monitoring, and validation tools

## Architecture

### Shared Infrastructure ✅

**Status:** Implemented

**Location:** `shared/`

The shared infrastructure provides common interfaces for all agents:

- **Base Docker Images**: Pre-configured images with CUDA, PyTorch, and dependencies
  - `base.Dockerfile`: Full image with GPU support (~15GB)
  - `base-minimal.Dockerfile`: Lightweight for non-GPU agents (~2GB)
- **Configuration Schema**: Unified YAML schema (`config_schema.yaml`)
- **Health Check API**: Standardized health checking (`health_check.py`)
- **Utilities**: Common Python modules for logging, networking, and GPU management

**Documentation:** [shared/README.md](shared/README.md)

---

### Agent 1: Infrastructure Foundation ✅

**Status:** Implemented

**Location:** `agents/agent1_infrastructure/`

**Scope:** Base networking and remote access

**Deliverables:**
- SSH setup scripts
- Tailscale VPN configuration
- Network validation tool
- Multi-node cluster orchestration

**Playbooks:**
- `connect-to-your-spark` - Single node setup
- `connect-two-sparks` - Multi-node cluster setup
- `tailscale` - VPN configuration
- `nccl` - Multi-GPU communication

**Interface Output:**
```python
from agents.agent1_infrastructure.agent1_output import get_cluster_config

CLUSTER_CONFIG = {
    "nodes": ["spark-001.local", "spark-002.local"],
    "ssh_keys": "/home/user/.ssh/dgx_key",
    "tailscale_ip": "100.x.x.x"
}
```

**Documentation:** [agents/agent1_infrastructure/README.md](agents/agent1_infrastructure/README.md)

---

### Agent 2: CUDA & Driver Setup (Coming Soon)

**Scope:** GPU drivers and CUDA toolkit installation

**Planned Deliverables:**
- NVIDIA driver installation
- CUDA toolkit setup
- GPU monitoring and validation

---

### Agent 3: Container Platform (Coming Soon)

**Scope:** Docker/container runtime setup

**Planned Deliverables:**
- Docker installation with GPU support
- Container registry setup
- Image management tools

---

## Quick Start

### Prerequisites

```bash
# Install shared dependencies
pip install -r shared/requirements.txt

# Install Ansible
pip install ansible

# System tools (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y openssh-client docker.io
```

### Option 1: Using Docker Compose

```bash
# Copy environment file
cp .env.example .env
# Edit .env with your configuration

# Build and start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f agent1-infrastructure
```

### Option 2: Deploy Agent 1 Directly

1. **Configure your cluster:**

```bash
cd agents/agent1_infrastructure
cp config/cluster.yaml.example config/cluster.yaml
# Edit cluster.yaml with your node details
```

2. **Deploy infrastructure:**

```bash
# Full deployment
python3 tools/cluster_orchestrator.py -c config/cluster.yaml deploy

# Or step-by-step:
python3 tools/cluster_orchestrator.py -c config/cluster.yaml multi-node
python3 tools/cluster_orchestrator.py -c config/cluster.yaml validate
```

3. **Verify setup:**

```bash
# Check cluster status
python3 tools/cluster_orchestrator.py -c config/cluster.yaml status

# Validate network
python3 tools/network_validator.py spark-001 spark-002

# Test interface
python3 agent1_output.py validate
```

## Repository Structure

```
spark-cluster/
├── README.md                           # This file
├── docker-compose.yml                  # Multi-agent orchestration
├── .env.example                        # Environment configuration template
├── .dockerignore                       # Docker ignore rules
├── .gitignore                          # Git ignore rules
│
├── shared/                             # Shared infrastructure ✅
│   ├── README.md                       # Shared components documentation
│   ├── requirements.txt                # Shared Python dependencies
│   ├── base.Dockerfile                 # Full GPU base image
│   ├── base-minimal.Dockerfile         # Minimal base image
│   ├── config_schema.yaml              # Unified configuration schema
│   ├── health_check.py                 # Health check API
│   ├── utils/                          # Shared utilities
│   │   ├── __init__.py
│   │   ├── config_loader.py            # Config loading & validation
│   │   ├── logger.py                   # Logging utilities
│   │   ├── network.py                  # Network utilities
│   │   └── gpu.py                      # GPU utilities
│   └── tests/                          # Shared component tests
│       ├── test_health_check.py
│       └── test_config_loader.py
│
├── agents/                             # Agent implementations
│   └── agent1_infrastructure/          # Agent 1 ✅
│       ├── README.md                   # Agent documentation
│       ├── agent1_output.py            # Interface module
│       ├── requirements.txt            # Agent dependencies
│       ├── playbooks/                  # Ansible playbooks
│       │   ├── connect-to-your-spark.yaml
│       │   ├── connect-two-sparks.yaml
│       │   ├── tailscale.yaml
│       │   └── nccl.yaml
│       ├── scripts/                    # Setup scripts
│       │   ├── setup_ssh.sh
│       │   └── setup_tailscale.sh
│       ├── tools/                      # Management tools
│       │   ├── network_validator.py
│       │   └── cluster_orchestrator.py
│       └── config/                     # Configuration examples
│           ├── cluster.yaml.example
│           └── inventory.ini.example
│
└── tests/                              # Integration tests
    ├── unit/                           # Unit tests
    └── integration/                    # Integration tests
        └── test_infrastructure.py
```

## Shared Interfaces

### Docker Base Images

All agents extend shared base images:

```dockerfile
# GPU-intensive agents
FROM spark-cluster/base:latest
# Agent-specific setup...

# Lightweight agents
FROM spark-cluster/minimal:latest
# Agent-specific setup...
```

Build base images:

```bash
# Full image (GPU support)
docker build -f shared/base.Dockerfile -t spark-cluster/base:latest .

# Minimal image
docker build -f shared/base-minimal.Dockerfile -t spark-cluster/minimal:latest .
```

### Configuration Schema

All agents follow the unified schema in `shared/config_schema.yaml`:

```yaml
agent:
  name: "agent_name"
  version: "1.0.0"
  docker:
    image: "spark-cluster/agent:latest"
    ports: [8000]
    gpu_required: true
  health_check:
    enabled: true
    port: 8000
    endpoint: "/health"
```

### Health Check API

All agents implement standardized health checks:

```python
from shared.health_check import HealthCheck

checker = HealthCheck()
health = checker.self_check()  # Returns health status
```

### Utilities

Common utilities available to all agents:

```python
# Configuration loading
from shared.utils import load_config, validate_config
config = load_config("agent.yaml", validate=True)

# Logging
from shared.utils import setup_logger
logger = setup_logger(__name__, level="INFO", json_format=True)

# Network operations
from shared.utils import NetworkUtils
if NetworkUtils.check_port_open("spark-001", 22):
    print("SSH available")

# GPU operations
from shared.utils import GPUUtils
if GPUUtils.is_available():
    gpus = GPUUtils.get_gpu_info()
```

## Agent Communication

Agents communicate through standardized Python interfaces:

```python
# Agent 1 provides cluster configuration
from agents.agent1_infrastructure.agent1_output import get_cluster_config

# Downstream agents consume this
cluster = get_cluster_config()
for node in cluster["nodes"]:
    # Deploy to each node
    deploy_to_node(node)
```

## Features

### Shared Infrastructure ✅

- ✅ Docker base images with CUDA 12.2 + PyTorch 24.10
- ✅ Unified configuration schema for all agents
- ✅ Health check API with GPU monitoring
- ✅ Utilities: config, logging, network, GPU
- ✅ Comprehensive documentation and tests

### Infrastructure (Agent 1) ✅

- ✅ Automated SSH key generation and distribution
- ✅ Tailscale VPN for secure cluster networking
- ✅ Multi-node cluster orchestration
- ✅ NCCL setup for multi-GPU communication
- ✅ Network validation and performance testing

### Planned Features

- 🔄 CUDA driver and toolkit installation (Agent 2)
- 🔄 Container platform setup (Agent 3)
- 🔄 Distributed training frameworks (Agent 4)
- 🔄 Monitoring and observability (Agent 5)

## Docker Compose

Deploy the entire stack:

```bash
# Start all agents
docker-compose up -d

# View logs
docker-compose logs -f

# Check health
docker-compose ps

# Stop all agents
docker-compose down
```

## Testing

```bash
# Test shared components
pytest shared/tests/ -v

# Test agents
pytest tests/integration/ -v

# Full test suite with coverage
pytest --cov=shared --cov=agents --cov-report=html

# Test specific agent
cd agents/agent1_infrastructure
./scripts/setup_ssh.sh --help
python3 tools/network_validator.py --help
```

## Documentation

- **Shared Infrastructure**: [shared/README.md](shared/README.md)
  - Base Docker images
  - Configuration schema
  - Health check API
  - Utility modules

- **Agent 1**: [agents/agent1_infrastructure/README.md](agents/agent1_infrastructure/README.md)
  - Network setup
  - SSH configuration
  - Tailscale VPN
  - NCCL multi-GPU

## Contributing

### Adding New Agents

1. Create agent directory: `agents/agentN_name/`
2. Extend shared base image
3. Follow configuration schema
4. Implement `agentN_output.py` interface
5. Add health check endpoint
6. Write comprehensive README.md
7. Include tests and examples

### Adding Shared Components

1. Add to appropriate `shared/` subdirectory
2. Update `config_schema.yaml` if needed
3. Write unit tests
4. Update `shared/README.md`
5. Ensure backward compatibility

## Requirements

- **Python**: 3.8+
- **Docker**: 20.10+ with nvidia-container-toolkit
- **Ansible**: 2.10+ (for playbooks)
- **NVIDIA GPU**: For GPU-enabled agents
- **SSH**: For cluster management

## Environment Variables

See `.env.example` for all configuration options:

```bash
# Copy and customize
cp .env.example .env
```

Key variables:
- `LOG_LEVEL`: Logging verbosity (DEBUG, INFO, WARNING, ERROR)
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `CLUSTER_NAME`: Cluster identifier
- `SSH_KEY_PATH`: Path to SSH private key

## License

MIT License

## Support

For issues and questions:
- **Shared Infrastructure**: See [shared/README.md](shared/README.md)
- **Agent 1**: See [agents/agent1_infrastructure/README.md](agents/agent1_infrastructure/README.md)
- **General Issues**: Open an issue in the repository

---

## Status Summary

| Component | Status | Documentation |
|-----------|--------|---------------|
| Shared Infrastructure | ✅ Complete | [shared/README.md](shared/README.md) |
| Agent 1: Infrastructure | ✅ Complete | [agents/agent1_infrastructure/README.md](agents/agent1_infrastructure/README.md) |
| Agent 2: CUDA Setup | 🔄 Planned | Coming Soon |
| Agent 3: Containers | 🔄 Planned | Coming Soon |

**Latest Update:** 2025-11-12 - Shared infrastructure and Agent 1 complete

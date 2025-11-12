# SparkTest - Spark Cluster Agent Infrastructure

Multi-agent infrastructure system for deploying and managing Spark/DGX GPU clusters.

## Overview

This repository contains a modular agent-based infrastructure for setting up and managing Spark clusters with support for:

- Remote access and networking
- Multi-node orchestration
- GPU communication (NCCL)
- Container management
- Monitoring and observability

## Agent Architecture

The infrastructure is divided into specialized agents, each handling specific aspects of cluster setup:

### Agent 1: Infrastructure Foundation ✅

**Status:** Implemented

**Scope:** Basis-Netzwerk + Remote Access

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
from agent1_infrastructure.agent1_output import get_cluster_config

CLUSTER_CONFIG = {
    "nodes": ["spark-001.local", "spark-002.local"],
    "ssh_keys": "/home/user/.ssh/dgx_key",
    "tailscale_ip": "100.x.x.x"
}
```

**Documentation:** [agent1_infrastructure/README.md](agent1_infrastructure/README.md)

---

### Agent 2: CUDA & Driver Setup (Coming Soon)

**Scope:** GPU drivers and CUDA toolkit installation

**Planned Deliverables:**
- NVIDIA driver installation
- CUDA toolkit setup
- GPU monitoring tools

---

### Agent 3: Container Platform (Coming Soon)

**Scope:** Docker/container runtime setup

**Planned Deliverables:**
- Docker installation
- Container registry setup
- GPU container support

---

## Quick Start

### Prerequisites

```bash
# Install Python dependencies
pip install -r agent1_infrastructure/requirements.txt

# Install Ansible
pip install ansible

# Install system tools (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y openssh-client
```

### Deploy Agent 1: Infrastructure Foundation

1. **Configure your cluster:**

```bash
cd agent1_infrastructure
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
SparkTest/
├── README.md                           # This file
├── agent1_infrastructure/              # Agent 1: Infrastructure Foundation
│   ├── README.md                       # Agent 1 documentation
│   ├── agent1_output.py               # Interface for downstream agents
│   ├── requirements.txt               # Python dependencies
│   ├── playbooks/                     # Ansible playbooks
│   │   ├── connect-to-your-spark.yaml
│   │   ├── connect-two-sparks.yaml
│   │   ├── tailscale.yaml
│   │   └── nccl.yaml
│   ├── scripts/                       # Setup scripts
│   │   ├── setup_ssh.sh
│   │   └── setup_tailscale.sh
│   ├── tools/                         # Management tools
│   │   ├── network_validator.py
│   │   └── cluster_orchestrator.py
│   └── config/                        # Configuration examples
│       ├── cluster.yaml.example
│       └── inventory.ini.example
└── [future agents]/                   # Additional agents TBD
```

## Agent Communication

Agents communicate through standardized Python interfaces:

```python
# Agent 1 provides cluster configuration
from agent1_infrastructure.agent1_output import get_cluster_config, get_connection_info

# Agent 2 would consume this configuration
cluster = get_cluster_config()
connection = get_connection_info()

# Use cluster info for CUDA installation
for node in cluster["nodes"]:
    install_cuda(node, connection["ssh"])
```

## Features

### Infrastructure (Agent 1)

- ✅ Automated SSH key generation and distribution
- ✅ Tailscale VPN for secure cluster networking
- ✅ Multi-node cluster orchestration
- ✅ NCCL setup for multi-GPU communication
- ✅ Network validation and performance testing
- ✅ Comprehensive documentation

### Planned Features

- 🔄 CUDA driver and toolkit installation (Agent 2)
- 🔄 Container platform setup (Agent 3)
- 🔄 Distributed training framework setup (Agent 4)
- 🔄 Monitoring and logging infrastructure (Agent 5)

## Documentation

- [Agent 1 Infrastructure Foundation](agent1_infrastructure/README.md)
- [Network Validation Guide](agent1_infrastructure/README.md#network-validation)
- [Cluster Orchestration Guide](agent1_infrastructure/README.md#tools)

## Testing

```bash
# Test SSH connectivity
cd agent1_infrastructure
./scripts/setup_ssh.sh

# Validate network
python3 tools/network_validator.py node1 node2

# Run NCCL tests
ssh node1 'sudo /usr/local/bin/nccl_test.sh'
```

## Contributing

Each agent is self-contained with its own:
- README.md with detailed documentation
- requirements.txt for dependencies
- Tests and validation tools
- Example configurations

When adding new agents:
1. Create agent directory: `agentN_name/`
2. Implement interface module: `agentN_output.py`
3. Add comprehensive README.md
4. Include example configurations
5. Provide validation tools

## Requirements

- Python 3.8+
- Ansible 2.10+
- SSH access to cluster nodes
- NVIDIA GPUs (for GPU-related features)

## License

MIT License

## Support

For issues and questions:
- Check agent-specific README files
- Review troubleshooting sections
- Open an issue in the repository

---

**Current Status:** Agent 1 (Infrastructure Foundation) - Complete ✅

**Next Steps:** Agent 2 (CUDA & Driver Setup) - Coming Soon

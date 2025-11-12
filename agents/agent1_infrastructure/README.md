# Agent 1: Infrastructure Foundation

Complete infrastructure setup for Spark cluster with remote access, networking, and multi-GPU communication.

## Overview

Agent 1 provides the foundational infrastructure components for Spark cluster deployment:

- **SSH Setup**: Automated SSH key generation and distribution
- **Tailscale VPN**: Secure cluster networking via Tailscale
- **Multi-Node Orchestration**: Cluster-wide configuration and management
- **NCCL Setup**: Multi-GPU communication for distributed training
- **Network Validation**: Comprehensive network testing and validation

## Directory Structure

```
agent1_infrastructure/
├── agent1_output.py          # Interface module for downstream agents
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── playbooks/                 # Ansible playbooks
│   ├── connect-to-your-spark.yaml    # Single-node setup
│   ├── connect-two-sparks.yaml       # Multi-node cluster setup
│   ├── tailscale.yaml                # Tailscale VPN setup
│   └── nccl.yaml                     # NCCL multi-GPU communication
├── scripts/                   # Setup scripts
│   ├── setup_ssh.sh          # SSH configuration script
│   └── setup_tailscale.sh    # Tailscale installation script
├── tools/                     # Validation and orchestration tools
│   ├── network_validator.py  # Network testing tool
│   └── cluster_orchestrator.py  # Cluster management tool
└── config/                    # Configuration files
    ├── cluster.yaml.example  # Example cluster configuration
    └── inventory.ini.example # Example Ansible inventory
```

## Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Ansible (if not already installed)
pip install ansible

# Install system dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install -y openssh-client sshpass
```

### 2. Configure Your Cluster

Create a cluster configuration file:

```bash
cp config/cluster.yaml.example config/cluster.yaml
# Edit cluster.yaml with your node details
```

Example `config/cluster.yaml`:

```yaml
cluster_name: spark-cluster
enable_tailscale: true
enable_nccl: true

nodes:
  - hostname: spark-001
    ip: 192.168.1.101
    user: ubuntu
  - hostname: spark-002
    ip: 192.168.1.102
    user: ubuntu

ssh_key_path: ~/.ssh/dgx_key
```

### 3. Deploy Infrastructure

```bash
# Deploy complete cluster infrastructure
python3 tools/cluster_orchestrator.py -c config/cluster.yaml deploy

# Or deploy individual components:

# Single node setup
python3 tools/cluster_orchestrator.py -c config/cluster.yaml single-node

# Multi-node cluster
python3 tools/cluster_orchestrator.py -c config/cluster.yaml multi-node

# Tailscale VPN (requires auth key)
export TAILSCALE_AUTH_KEY="tskey-..."
python3 tools/cluster_orchestrator.py -c config/cluster.yaml tailscale

# NCCL multi-GPU
python3 tools/cluster_orchestrator.py -c config/cluster.yaml nccl
```

## Playbooks

### connect-to-your-spark.yaml

Single-node Spark setup with SSH access.

**Features:**
- SSH server configuration
- SSH key deployment
- Security hardening
- Connection testing

**Usage:**
```bash
ansible-playbook playbooks/connect-to-your-spark.yaml -i config/inventory.ini
```

### connect-two-sparks.yaml

Multi-node cluster setup with bidirectional SSH access.

**Features:**
- Cluster-wide SSH key distribution
- /etc/hosts configuration
- NFS shared storage setup
- Inter-node connectivity testing

**Usage:**
```bash
ansible-playbook playbooks/connect-two-sparks.yaml -i config/inventory.ini
```

### tailscale.yaml

Tailscale VPN configuration for secure cluster networking.

**Features:**
- Tailscale installation
- Network authentication
- IP address assignment
- SSH configuration for Tailscale IPs

**Usage:**
```bash
export TAILSCALE_AUTH_KEY="tskey-..."
ansible-playbook playbooks/tailscale.yaml -i config/inventory.ini
```

### nccl.yaml

NVIDIA NCCL setup for multi-GPU communication.

**Features:**
- NCCL library installation
- Network interface detection
- InfiniBand configuration (if available)
- NCCL test suite deployment
- Performance benchmarking

**Usage:**
```bash
ansible-playbook playbooks/nccl.yaml -i config/inventory.ini
```

## Scripts

### setup_ssh.sh

Automated SSH key generation and distribution.

```bash
# Generate SSH key only
./scripts/setup_ssh.sh

# Generate and copy to hosts
./scripts/setup_ssh.sh spark-001.local spark-002.local

# Custom key path
SSH_KEY_PATH=~/.ssh/custom_key ./scripts/setup_ssh.sh host1 host2
```

### setup_tailscale.sh

Tailscale installation and configuration.

```bash
# Basic installation
./scripts/setup_tailscale.sh

# With authentication
./scripts/setup_tailscale.sh tskey-... spark-001

# As subnet router
./scripts/setup_tailscale.sh tskey-... spark-001 192.168.1.0/24
```

## Tools

### network_validator.py

Comprehensive network validation tool.

**Features:**
- SSH connectivity testing
- Latency measurement (ping)
- Bandwidth testing (iperf3)
- GPU topology detection

**Usage:**
```bash
# Run all tests
python3 tools/network_validator.py spark-001 spark-002

# Specific tests
python3 tools/network_validator.py spark-001 spark-002 --tests ssh ping

# Save results
python3 tools/network_validator.py spark-001 spark-002 --output results.json
```

### cluster_orchestrator.py

Cluster management and orchestration tool.

**Commands:**
```bash
# Deploy full cluster
python3 tools/cluster_orchestrator.py -c config/cluster.yaml deploy

# Get cluster status
python3 tools/cluster_orchestrator.py -c config/cluster.yaml status

# Generate inventory
python3 tools/cluster_orchestrator.py -c config/cluster.yaml inventory

# Validate network
python3 tools/cluster_orchestrator.py -c config/cluster.yaml validate

# Show cluster info
python3 tools/cluster_orchestrator.py -c config/cluster.yaml info
```

## Agent Interface (agent1_output.py)

The `agent1_output.py` module provides a standardized interface for downstream agents.

### Python API

```python
from agent1_infrastructure.agent1_output import get_cluster_config, get_connection_info

# Get cluster configuration
config = get_cluster_config()
print(config["nodes"])  # List of cluster nodes

# Get connection information
conn_info = get_connection_info()
print(conn_info["ssh"]["key_path"])  # SSH key path
```

### CLI Usage

```bash
# Get cluster configuration
python3 agent1_output.py config

# Get SSH configuration
python3 agent1_output.py ssh

# Get network configuration
python3 agent1_output.py network

# Get NCCL configuration
python3 agent1_output.py nccl

# Get connection endpoints
python3 agent1_output.py endpoints

# Validate setup
python3 agent1_output.py validate

# Get connection command
python3 agent1_output.py connect
python3 agent1_output.py connect spark-001

# Export complete configuration
python3 agent1_output.py export cluster_config.json
```

### Output Format

```python
CLUSTER_CONFIG = {
    "cluster_name": "spark-cluster",
    "nodes": ["spark-001.local", "spark-002.local"],
    "ssh_keys": "/home/user/.ssh/dgx_key",
    "tailscale_ip": "100.x.x.x",
    "primary_node": "spark-001.local",
    "node_count": 2
}
```

## Configuration Examples

### cluster.yaml

```yaml
cluster_name: my-spark-cluster
enable_tailscale: true
enable_nccl: true
tailscale_auth_key: tskey-...

nodes:
  - hostname: spark-001
    ip: 192.168.1.101
    user: ubuntu
  - hostname: spark-002
    ip: 192.168.1.102
    user: ubuntu

ssh_key_path: ~/.ssh/dgx_key
```

### inventory.ini

```ini
[spark_nodes]
spark-001 ansible_host=192.168.1.101 ansible_user=ubuntu
spark-002 ansible_host=192.168.1.102 ansible_user=ubuntu

[spark_cluster:children]
spark_nodes

[spark_cluster:vars]
cluster_name=spark-cluster
ssh_key_path=~/.ssh/dgx_key
```

## Testing

### Network Validation

```bash
# Basic connectivity test
python3 tools/network_validator.py spark-001 spark-002

# Full validation with bandwidth
python3 tools/network_validator.py spark-001 spark-002 \
    --tests ssh ping bandwidth gpu \
    --output validation_results.json
```

### NCCL Testing

```bash
# Single node NCCL test
ssh spark-001 'sudo /usr/local/bin/nccl_test.sh'

# Multi-node NCCL test
ssh spark-001 'sudo /usr/local/bin/nccl_multinode_test.sh'
```

## Troubleshooting

### SSH Issues

```bash
# Check SSH key permissions
ls -la ~/.ssh/dgx_key
# Should be 600

# Test SSH connection manually
ssh -i ~/.ssh/dgx_key -v user@spark-001

# Check SSH server status
ssh spark-001 'sudo systemctl status sshd'
```

### Tailscale Issues

```bash
# Check Tailscale status
ssh spark-001 'tailscale status'

# Get Tailscale IP
ssh spark-001 'tailscale ip -4'

# Check Tailscale logs
ssh spark-001 'sudo journalctl -u tailscaled'
```

### NCCL Issues

```bash
# Check NCCL installation
ssh spark-001 'dpkg -l | grep nccl'

# Test GPU visibility
ssh spark-001 'nvidia-smi'

# Check NCCL environment
ssh spark-001 'cat /etc/nccl/nccl.conf'

# Enable NCCL debug mode
ssh spark-001 'export NCCL_DEBUG=INFO && /usr/local/bin/nccl_test.sh'
```

## Requirements

- Python 3.8+
- Ansible 2.10+
- SSH client
- NVIDIA GPU (for NCCL)
- iperf3 (for bandwidth testing)

## Integration with Other Agents

Agent 1 provides outputs that can be consumed by downstream agents:

```python
# Agent 2: CUDA/Driver Setup
from agent1_infrastructure.agent1_output import get_cluster_config

config = get_cluster_config()
for node in config["nodes"]:
    # Deploy CUDA drivers to each node
    deploy_cuda(node)

# Agent 3: Container Setup
from agent1_infrastructure.agent1_output import get_connection_info

conn = get_connection_info()
ssh_key = conn["ssh"]["key_path"]
# Use SSH key for container deployment
```

## License

MIT License

## Support

For issues and questions, please open an issue in the repository.

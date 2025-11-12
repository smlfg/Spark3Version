# Shared Infrastructure

Common interfaces, utilities, and base images for all Spark cluster agents.

## Overview

The `shared/` directory contains reusable components that provide a unified foundation for all agents:

- **Base Docker Images**: Pre-configured images with CUDA, PyTorch, and common dependencies
- **Configuration Schema**: Unified YAML schema for agent configuration
- **Health Check API**: Standardized health checking across all services
- **Utilities**: Common Python utilities for logging, networking, and GPU management

## Directory Structure

```
shared/
├── README.md                   # This file
├── requirements.txt            # Shared Python dependencies
├── base.Dockerfile            # Full base image with GPU support
├── base-minimal.Dockerfile    # Minimal base image for non-GPU agents
├── config_schema.yaml         # Unified configuration schema
├── health_check.py            # Health check API
├── utils/                     # Utility modules
│   ├── __init__.py
│   ├── config_loader.py       # Configuration loading and validation
│   ├── logger.py              # Logging utilities
│   ├── network.py             # Network utilities
│   └── gpu.py                 # GPU utilities
└── tests/                     # Unit tests
    ├── test_health_check.py
    └── test_config_loader.py
```

## Docker Base Images

### base.Dockerfile (Full Image)

Comprehensive image for GPU-intensive agents with:
- NVIDIA PyTorch 24.10
- CUDA 12.2 + Toolkit
- NCCL for multi-GPU
- InfiniBand support
- Common ML libraries

**Use for**: Training agents, GPU compute agents

```dockerfile
FROM spark-cluster/base:latest
```

**Size**: ~15GB
**GPU**: Required

### base-minimal.Dockerfile (Minimal Image)

Lightweight image for non-GPU agents with:
- Ubuntu 22.04 + CUDA base
- Python 3.10
- Minimal dependencies

**Use for**: API services, monitoring, orchestration

```dockerfile
FROM spark-cluster/minimal:latest
```

**Size**: ~2GB
**GPU**: Optional

## Configuration Schema

All agents follow a unified configuration schema defined in `config_schema.yaml`.

### Example Agent Configuration

```yaml
agent:
  name: "agent1_infrastructure"
  version: "1.0.0"
  description: "Infrastructure foundation agent"

  dependencies: []

  docker:
    image: "spark-cluster/agent1:latest"
    ports: [8000, 8001]
    volumes:
      - "/data:/app/data"
      - "shared-logs:/logs"
    environment:
      CUDA_VISIBLE_DEVICES: "0,1"
      LOG_LEVEL: "INFO"
    gpu_required: false

  resources:
    cpu:
      limit: "4.0"
      reservation: "2.0"
    memory:
      limit: "16G"
      reservation: "8G"

  health_check:
    enabled: true
    endpoint: "/health"
    port: 8000
    interval: 30
    timeout: 10
    retries: 3
```

### Schema Sections

- **agent**: Core agent metadata and configuration
- **playbook**: Ansible playbook configuration
- **service**: Service endpoint configuration
- **cluster**: Multi-node cluster configuration

See `config_schema.yaml` for complete schema documentation.

## Health Check API

### Python API

```python
from shared.health_check import HealthCheck, check_service

# Create health checker
checker = HealthCheck()

# Check service health
if checker.check_service(port=8000, endpoint="/health"):
    print("Service is healthy")

# Comprehensive self-check
health = checker.self_check()
print(f"Status: {health['status']}")
print(f"CPU: {health['system']['cpu']['percent']}%")
print(f"Memory: {health['system']['memory']['percent']}%")

# Check GPU
gpu_info = checker.check_gpu()
if gpu_info and gpu_info['available']:
    print(f"GPUs: {gpu_info['count']}")
```

### CLI Usage

```bash
# Self health check
python3 shared/health_check.py --self-check

# Check specific service
python3 shared/health_check.py --port 8000 --endpoint /health

# Check GPU
python3 shared/health_check.py --check-gpu

# JSON output
python3 shared/health_check.py --self-check --json

# Check processes
python3 shared/health_check.py --check-process nginx python3
```

### Docker Health Check

The base images include automatic health checks:

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python3 /usr/local/bin/health_check.py --self-check || exit 1
```

## Utilities

### Configuration Loader

```python
from shared.utils import ConfigLoader, load_config, validate_config

# Load and validate config
config = load_config("agent.yaml", validate=True, config_type="agent")

# Manual validation
loader = ConfigLoader()
errors = loader.validate_config(config, "agent")
if errors:
    print("Validation errors:", errors)

# Merge configs
merged = loader.merge_configs(base_config, override_config)

# Apply defaults
config_with_defaults = loader.apply_defaults(config, "agent")
```

### Logger

```python
from shared.utils import setup_logger, get_logger

# Setup logger
logger = setup_logger(
    name=__name__,
    level="INFO",
    log_file="agent.log",
    json_format=True
)

logger.info("Service started")
logger.error("Error occurred", extra={"request_id": "123"})

# Get existing logger
logger = get_logger(__name__)
```

### Network Utilities

```python
from shared.utils import NetworkUtils

# Get local IP
ip = NetworkUtils.get_local_ip()

# Check if port is open
if NetworkUtils.check_port_open("spark-001", 22):
    print("SSH is available")

# Wait for service
if NetworkUtils.wait_for_port("localhost", 8000, timeout=60):
    print("Service is ready")

# Find available port
port = NetworkUtils.get_open_port(start_port=8000, end_port=9000)

# Ping host
result = NetworkUtils.ping("spark-001")
if result['success']:
    print(f"Latency: {result['latency_ms']['avg']}ms")

# Test bandwidth (requires iperf3)
result = NetworkUtils.test_bandwidth("spark-001", duration=5)
if result['success']:
    print(f"Bandwidth: {result['bandwidth_gbps']:.2f} Gbps")
```

### GPU Utilities

```python
from shared.utils import GPUUtils

# Check GPU availability
if GPUUtils.is_available():
    print(f"GPUs available: {GPUUtils.get_gpu_count()}")

# Get GPU info
gpus = GPUUtils.get_gpu_info()
for gpu in gpus:
    print(f"GPU {gpu['id']}: {gpu['name']}")
    print(f"  Memory: {gpu['memory']['used_mb']:.0f}/{gpu['memory']['total_mb']:.0f} MB")
    print(f"  Utilization: {gpu['utilization']['gpu_percent']}%")
    print(f"  Temperature: {gpu['temperature_c']}°C")

# Get CUDA version
cuda_version = GPUUtils.get_cuda_version()
print(f"CUDA: {cuda_version}")

# Set visible devices
GPUUtils.set_visible_devices([0, 1])

# Get GPU topology
topology = GPUUtils.get_gpu_topology()
print(topology)

# Check NCCL
if GPUUtils.check_nccl():
    print("NCCL is available")

# Get processes using GPU
processes = GPUUtils.get_processes(gpu_id=0)
for proc in processes:
    print(f"PID {proc['pid']}: {proc['name']} ({proc['memory_mb']:.0f} MB)")
```

## Building Base Images

### Build Full Image

```bash
# Build
docker build -f shared/base.Dockerfile -t spark-cluster/base:latest .

# Test
docker run --rm --gpus all spark-cluster/base:latest nvidia-smi
```

### Build Minimal Image

```bash
# Build
docker build -f shared/base-minimal.Dockerfile -t spark-cluster/minimal:latest .

# Test
docker run --rm spark-cluster/minimal:latest python3 --version
```

## Integration with Agents

Agents should:

1. **Extend base images**:
   ```dockerfile
   FROM spark-cluster/base:latest
   # Agent-specific setup
   ```

2. **Use configuration schema**:
   ```yaml
   # agent.yaml
   agent:
     name: "my_agent"
     version: "1.0.0"
     # ... follow schema
   ```

3. **Import shared utilities**:
   ```python
   from shared.utils import setup_logger, GPUUtils
   from shared.health_check import HealthCheck
   ```

4. **Implement health checks**:
   ```python
   @app.get("/health")
   def health():
       checker = HealthCheck()
       return checker.self_check()
   ```

5. **Follow agent interface pattern**:
   ```python
   # agentN_output.py
   def get_config():
       """Return agent configuration"""
       return {...}

   def get_status():
       """Return agent status"""
       return {...}
   ```

## Testing

Run shared component tests:

```bash
# Unit tests
pytest shared/tests/ -v

# Integration tests
pytest tests/integration/ -v

# All tests with coverage
pytest --cov=shared --cov-report=html
```

## Requirements

Install shared dependencies:

```bash
pip install -r shared/requirements.txt
```

## Contributing

When adding new shared components:

1. Update `config_schema.yaml` if adding new config options
2. Add utility functions to appropriate module
3. Write unit tests in `shared/tests/`
4. Update this README with usage examples
5. Ensure backward compatibility with existing agents

## Version History

- **1.0.0** (2025-11-12): Initial release
  - Base Docker images
  - Configuration schema
  - Health check API
  - Core utilities (config, logger, network, GPU)

## License

MIT License

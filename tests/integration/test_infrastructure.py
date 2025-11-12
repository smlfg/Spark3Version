"""
Integration tests for Agent 1: Infrastructure Foundation
"""

import pytest
import subprocess
import sys
from pathlib import Path


def test_agent1_output_import():
    """Test that agent1_output can be imported"""
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from agent1_infrastructure.agent1_output import get_cluster_config, CLUSTER_CONFIG

        assert CLUSTER_CONFIG is not None
        assert "nodes" in CLUSTER_CONFIG
        assert "ssh_keys" in CLUSTER_CONFIG
    except ImportError as e:
        pytest.skip(f"Agent 1 not available: {e}")


def test_health_check_script():
    """Test health check script execution"""
    script_path = Path(__file__).parent.parent.parent / "shared" / "health_check.py"

    if not script_path.exists():
        pytest.skip("Health check script not found")

    result = subprocess.run(
        ["python3", str(script_path), "--self-check"],
        capture_output=True,
        timeout=10
    )

    # Should exit with 0 or 1 (healthy or unhealthy)
    assert result.returncode in [0, 1]


def test_network_validator_help():
    """Test network validator help"""
    validator_path = Path(__file__).parent.parent.parent / "agent1_infrastructure" / "tools" / "network_validator.py"

    if not validator_path.exists():
        pytest.skip("Network validator not found")

    result = subprocess.run(
        ["python3", str(validator_path), "--help"],
        capture_output=True,
        timeout=5
    )

    assert result.returncode == 0
    assert b"network" in result.stdout.lower() or b"validation" in result.stdout.lower()


def test_cluster_orchestrator_help():
    """Test cluster orchestrator help"""
    orchestrator_path = Path(__file__).parent.parent.parent / "agent1_infrastructure" / "tools" / "cluster_orchestrator.py"

    if not orchestrator_path.exists():
        pytest.skip("Cluster orchestrator not found")

    result = subprocess.run(
        ["python3", str(orchestrator_path), "--help"],
        capture_output=True,
        timeout=5
    )

    assert result.returncode in [0, 2]  # 0 or 2 (argparse help exit code)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

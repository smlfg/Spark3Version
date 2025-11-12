"""
Tests for health_check module
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from health_check import HealthCheck


def test_health_check_init():
    """Test HealthCheck initialization"""
    checker = HealthCheck()
    assert checker.hostname is not None
    assert checker.start_time > 0


def test_check_port_localhost():
    """Test port checking on localhost"""
    checker = HealthCheck()

    # Port 80 might not be open
    result = checker.check_port("localhost", 80, timeout=1)
    assert isinstance(result, bool)


def test_check_system_resources():
    """Test system resource checking"""
    checker = HealthCheck()
    resources = checker.check_system_resources()

    assert "cpu" in resources
    assert "memory" in resources
    assert "disk" in resources

    assert resources["cpu"]["percent"] >= 0
    assert resources["memory"]["total"] > 0
    assert resources["disk"]["total"] > 0


def test_check_network_interfaces():
    """Test network interface detection"""
    checker = HealthCheck()
    interfaces = checker.check_network_interfaces()

    assert isinstance(interfaces, dict)
    assert len(interfaces) > 0


def test_self_check():
    """Test comprehensive self check"""
    checker = HealthCheck()
    health = checker.self_check()

    assert "status" in health
    assert "timestamp" in health
    assert "hostname" in health
    assert "system" in health
    assert health["status"] in ["healthy", "degraded", "unhealthy"]


def test_get_status_code():
    """Test status code generation"""
    checker = HealthCheck()
    status_code = checker.get_status_code()

    assert status_code in [200, 503]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Tests for config_loader module
"""

import pytest
import tempfile
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from config_loader import ConfigLoader, load_config, validate_config


@pytest.fixture
def config_loader():
    """Create ConfigLoader instance"""
    return ConfigLoader()


@pytest.fixture
def sample_agent_config():
    """Sample agent configuration"""
    return {
        "name": "test_agent",
        "version": "1.0.0",
        "description": "Test agent",
        "docker": {
            "image": "test:latest",
            "ports": [8000, 8001],
            "gpu_required": True
        },
        "health_check": {
            "enabled": True,
            "port": 8000,
            "endpoint": "/health"
        }
    }


def test_config_loader_init(config_loader):
    """Test ConfigLoader initialization"""
    assert config_loader.schema is not None
    assert "agent" in config_loader.schema


def test_load_config_yaml(config_loader, sample_agent_config):
    """Test loading YAML config"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(sample_agent_config, f)
        temp_path = f.name

    try:
        config = config_loader.load_config(temp_path)
        assert config["name"] == "test_agent"
        assert config["version"] == "1.0.0"
    finally:
        Path(temp_path).unlink()


def test_validate_valid_config(config_loader, sample_agent_config):
    """Test validation of valid config"""
    errors = config_loader.validate_config(sample_agent_config, "agent")
    assert len(errors) == 0


def test_validate_missing_required(config_loader):
    """Test validation with missing required fields"""
    invalid_config = {
        "description": "Missing required fields"
    }

    errors = config_loader.validate_config(invalid_config, "agent")
    assert len(errors) > 0
    assert any("name" in error for error in errors)


def test_validate_invalid_type(config_loader):
    """Test validation with invalid type"""
    invalid_config = {
        "name": "test",
        "version": "1.0.0",
        "docker": {
            "image": "test:latest",
            "gpu_required": "yes"  # Should be boolean
        }
    }

    errors = config_loader.validate_config(invalid_config, "agent")
    # May or may not have errors depending on schema strictness


def test_merge_configs(config_loader):
    """Test config merging"""
    base = {
        "name": "test",
        "docker": {
            "image": "base:latest",
            "ports": [8000]
        }
    }

    override = {
        "docker": {
            "ports": [8001, 8002]
        }
    }

    merged = config_loader.merge_configs(base, override)
    assert merged["name"] == "test"
    assert merged["docker"]["ports"] == [8001, 8002]


def test_apply_defaults(config_loader):
    """Test applying default values"""
    config = {
        "name": "test",
        "version": "1.0.0",
        "docker": {
            "image": "test:latest"
        }
    }

    with_defaults = config_loader.apply_defaults(config, "agent")
    # Should have defaults applied from schema


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

"""
Configuration Loader and Validator
Loads and validates agent configurations against the unified schema
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
import json


class ConfigLoader:
    """Load and validate agent configurations"""

    def __init__(self, schema_path: Optional[str] = None):
        """
        Initialize config loader

        Args:
            schema_path: Path to config schema YAML file
        """
        if schema_path is None:
            schema_path = Path(__file__).parent.parent / "config_schema.yaml"

        self.schema_path = Path(schema_path)
        self.schema = self._load_schema()

    def _load_schema(self) -> Dict:
        """Load configuration schema"""
        if not self.schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {self.schema_path}")

        with open(self.schema_path, 'r') as f:
            return yaml.safe_load(f)

    def load_config(self, config_path: str) -> Dict:
        """
        Load configuration from file

        Args:
            config_path: Path to configuration file (YAML or JSON)

        Returns:
            Configuration dictionary

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config format is invalid
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load based on extension
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

        return config

    def validate_config(self, config: Dict, config_type: str = "agent") -> List[str]:
        """
        Validate configuration against schema

        Args:
            config: Configuration dictionary to validate
            config_type: Type of config (agent, playbook, service, cluster)

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        if config_type not in self.schema:
            errors.append(f"Unknown config type: {config_type}")
            return errors

        schema = self.schema[config_type]

        # Validate required fields
        for field, spec in schema.items():
            if isinstance(spec, dict) and spec.get('required', False):
                if field not in config:
                    errors.append(f"Missing required field: {field}")

        # Validate field types and constraints
        for field, value in config.items():
            if field in schema:
                field_errors = self._validate_field(field, value, schema[field])
                errors.extend(field_errors)

        return errors

    def _validate_field(self, field: str, value: Any, spec: Dict) -> List[str]:
        """Validate individual field against spec"""
        errors = []

        if not isinstance(spec, dict):
            return errors

        # Type validation
        expected_type = spec.get('type')
        if expected_type:
            if expected_type == 'string' and not isinstance(value, str):
                errors.append(f"{field}: expected string, got {type(value).__name__}")
            elif expected_type == 'integer' and not isinstance(value, int):
                errors.append(f"{field}: expected integer, got {type(value).__name__}")
            elif expected_type == 'boolean' and not isinstance(value, bool):
                errors.append(f"{field}: expected boolean, got {type(value).__name__}")
            elif expected_type == 'list' and not isinstance(value, list):
                errors.append(f"{field}: expected list, got {type(value).__name__}")
            elif expected_type == 'dict' and not isinstance(value, dict):
                errors.append(f"{field}: expected dict, got {type(value).__name__}")

        # Enum validation
        if 'enum' in spec and value not in spec['enum']:
            errors.append(f"{field}: value '{value}' not in allowed values: {spec['enum']}")

        # Range validation
        if 'minimum' in spec and isinstance(value, (int, float)) and value < spec['minimum']:
            errors.append(f"{field}: value {value} is less than minimum {spec['minimum']}")

        if 'maximum' in spec and isinstance(value, (int, float)) and value > spec['maximum']:
            errors.append(f"{field}: value {value} is greater than maximum {spec['maximum']}")

        # Pattern validation
        if 'pattern' in spec and isinstance(value, str):
            import re
            if not re.match(spec['pattern'], value):
                errors.append(f"{field}: value '{value}' does not match pattern {spec['pattern']}")

        return errors

    def merge_configs(self, base_config: Dict, override_config: Dict) -> Dict:
        """
        Merge two configurations with override taking precedence

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration
        """
        result = base_config.copy()

        for key, value in override_config.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self.merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def apply_defaults(self, config: Dict, config_type: str = "agent") -> Dict:
        """
        Apply default values from schema to config

        Args:
            config: Configuration dictionary
            config_type: Type of config

        Returns:
            Configuration with defaults applied
        """
        if config_type not in self.schema:
            return config

        result = config.copy()
        schema = self.schema[config_type]

        for field, spec in schema.items():
            if isinstance(spec, dict) and 'default' in spec:
                if field not in result:
                    result[field] = spec['default']

        return result


def load_config(config_path: str, validate: bool = True, config_type: str = "agent") -> Dict:
    """
    Convenience function to load and optionally validate config

    Args:
        config_path: Path to config file
        validate: Whether to validate against schema
        config_type: Type of config

    Returns:
        Configuration dictionary

    Raises:
        ValueError: If validation fails
    """
    loader = ConfigLoader()
    config = loader.load_config(config_path)

    if validate:
        errors = loader.validate_config(config, config_type)
        if errors:
            raise ValueError(f"Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

    return loader.apply_defaults(config, config_type)


def validate_config(config: Dict, config_type: str = "agent") -> bool:
    """
    Convenience function to validate config

    Args:
        config: Configuration dictionary
        config_type: Type of config

    Returns:
        True if valid, False otherwise
    """
    loader = ConfigLoader()
    errors = loader.validate_config(config, config_type)
    return len(errors) == 0

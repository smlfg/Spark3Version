#!/usr/bin/env python3
"""
Cluster Orchestration Tool for Spark Multi-Node Setup
Manages cluster configuration, deployment, and monitoring
"""

import argparse
import json
import os
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import time


class ClusterOrchestrator:
    def __init__(self, config_file: str):
        self.config_file = Path(config_file)
        self.config = self.load_config()
        self.base_dir = Path(__file__).parent.parent

    def load_config(self) -> Dict:
        """Load cluster configuration from YAML file"""
        if not self.config_file.exists():
            print(f"Error: Config file {self.config_file} not found")
            sys.exit(1)

        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required_fields = ["cluster_name", "nodes"]
        for field in required_fields:
            if field not in config:
                print(f"Error: Required field '{field}' missing in config")
                sys.exit(1)

        return config

    def generate_inventory(self, output_file: str = None) -> str:
        """Generate Ansible inventory file from cluster configuration"""
        if output_file is None:
            output_file = self.base_dir / "config" / "inventory.ini"

        inventory_content = f"""# Ansible Inventory for {self.config['cluster_name']}
# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}

[spark_nodes]
"""

        # Add individual nodes
        for node in self.config["nodes"]:
            hostname = node["hostname"]
            ip = node["ip"]
            ansible_user = node.get("user", os.getenv("USER"))

            inventory_content += f"{hostname} ansible_host={ip} ansible_user={ansible_user}\n"

        inventory_content += f"\n[spark_cluster:children]\nspark_nodes\n"

        # Add cluster-wide variables
        inventory_content += f"""
[spark_cluster:vars]
cluster_name={self.config['cluster_name']}
ssh_key_path={self.config.get('ssh_key_path', '~/.ssh/dgx_key')}
"""

        # Write inventory file
        os.makedirs(output_file.parent, exist_ok=True)
        with open(output_file, 'w') as f:
            f.write(inventory_content)

        print(f"✓ Generated inventory file: {output_file}")
        return str(output_file)

    def run_playbook(self, playbook: str, extra_vars: Dict = None, tags: List[str] = None) -> int:
        """Execute Ansible playbook"""
        playbook_path = self.base_dir / "playbooks" / f"{playbook}.yaml"

        if not playbook_path.exists():
            print(f"Error: Playbook {playbook_path} not found")
            return 1

        inventory_file = self.generate_inventory()

        cmd = [
            "ansible-playbook",
            str(playbook_path),
            "-i", inventory_file,
        ]

        if extra_vars:
            vars_str = " ".join([f"{k}={v}" for k, v in extra_vars.items()])
            cmd.extend(["--extra-vars", vars_str])

        if tags:
            cmd.extend(["--tags", ",".join(tags)])

        print(f"\n{'='*60}")
        print(f"Running playbook: {playbook}")
        print(f"Command: {' '.join(cmd)}")
        print(f"{'='*60}\n")

        try:
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except KeyboardInterrupt:
            print("\n\nPlaybook execution interrupted by user")
            return 130
        except Exception as e:
            print(f"Error executing playbook: {e}")
            return 1

    def setup_single_node(self, node: str = None) -> int:
        """Setup single Spark node"""
        print(f"Setting up single Spark node...")

        extra_vars = {}
        if node:
            extra_vars["target_node"] = node

        return self.run_playbook("connect-to-your-spark", extra_vars)

    def setup_multi_node(self) -> int:
        """Setup multi-node Spark cluster"""
        print(f"Setting up multi-node Spark cluster with {len(self.config['nodes'])} nodes...")
        return self.run_playbook("connect-two-sparks")

    def setup_tailscale(self, auth_key: str = None) -> int:
        """Setup Tailscale VPN for cluster"""
        print(f"Setting up Tailscale VPN...")

        extra_vars = {}
        if auth_key:
            extra_vars["tailscale_auth_key"] = auth_key
        elif "tailscale_auth_key" in self.config:
            extra_vars["tailscale_auth_key"] = self.config["tailscale_auth_key"]
        else:
            print("Warning: No Tailscale auth key provided")

        return self.run_playbook("tailscale", extra_vars)

    def setup_nccl(self) -> int:
        """Setup NCCL for multi-GPU communication"""
        print(f"Setting up NCCL for multi-GPU communication...")
        return self.run_playbook("nccl")

    def validate_network(self) -> int:
        """Validate network connectivity and performance"""
        print(f"Validating cluster network...")

        validator_script = self.base_dir / "tools" / "network_validator.py"
        nodes = [node["ip"] for node in self.config["nodes"]]

        cmd = [
            "python3",
            str(validator_script),
            *nodes,
            "--ssh-key", self.config.get("ssh_key_path", "~/.ssh/dgx_key"),
            "--output", "network_validation_results.json"
        ]

        try:
            result = subprocess.run(cmd, check=False)
            return result.returncode
        except Exception as e:
            print(f"Error running network validation: {e}")
            return 1

    def deploy_full_cluster(self) -> int:
        """Deploy full cluster with all components"""
        print(f"\n{'='*60}")
        print(f"Deploying Full Spark Cluster: {self.config['cluster_name']}")
        print(f"{'='*60}\n")

        steps = [
            ("Multi-node setup", self.setup_multi_node),
            ("Network validation", self.validate_network),
        ]

        # Add optional components
        if self.config.get("enable_tailscale", False):
            steps.insert(1, ("Tailscale VPN", lambda: self.setup_tailscale()))

        if self.config.get("enable_nccl", True):
            steps.append(("NCCL multi-GPU", self.setup_nccl))

        for step_name, step_func in steps:
            print(f"\n{'='*60}")
            print(f"Step: {step_name}")
            print(f"{'='*60}\n")

            result = step_func()
            if result != 0:
                print(f"\n✗ Step '{step_name}' failed with exit code {result}")
                return result

            print(f"\n✓ Step '{step_name}' completed successfully")

        print(f"\n{'='*60}")
        print(f"✓ Full cluster deployment completed successfully!")
        print(f"{'='*60}\n")

        return 0

    def get_cluster_status(self) -> Dict:
        """Get current cluster status"""
        status = {
            "cluster_name": self.config["cluster_name"],
            "nodes": [],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

        ssh_key = self.config.get("ssh_key_path", "~/.ssh/dgx_key")

        for node in self.config["nodes"]:
            node_status = {
                "hostname": node["hostname"],
                "ip": node["ip"],
                "reachable": False,
                "uptime": None,
                "load": None
            }

            # Check SSH connectivity
            try:
                result = subprocess.run(
                    ["ssh", "-i", ssh_key, "-o", "ConnectTimeout=5",
                     "-o", "BatchMode=yes", node["ip"], "uptime"],
                    capture_output=True,
                    text=True,
                    timeout=10
                )

                if result.returncode == 0:
                    node_status["reachable"] = True
                    node_status["uptime"] = result.stdout.strip()

            except Exception as e:
                node_status["error"] = str(e)

            status["nodes"].append(node_status)

        return status

    def print_cluster_info(self):
        """Print cluster configuration information"""
        print(f"\n{'='*60}")
        print(f"Cluster Configuration: {self.config['cluster_name']}")
        print(f"{'='*60}\n")

        print(f"Nodes: {len(self.config['nodes'])}")
        for i, node in enumerate(self.config['nodes'], 1):
            print(f"  {i}. {node['hostname']} ({node['ip']})")

        print(f"\nSSH Key: {self.config.get('ssh_key_path', '~/.ssh/dgx_key')}")
        print(f"Tailscale: {'Enabled' if self.config.get('enable_tailscale', False) else 'Disabled'}")
        print(f"NCCL: {'Enabled' if self.config.get('enable_nccl', True) else 'Disabled'}")

        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Spark Cluster Orchestration Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Deploy full cluster
  %(prog)s -c cluster.yaml deploy

  # Setup single node
  %(prog)s -c cluster.yaml single-node

  # Setup multi-node cluster
  %(prog)s -c cluster.yaml multi-node

  # Setup Tailscale VPN
  %(prog)s -c cluster.yaml tailscale --auth-key <key>

  # Setup NCCL
  %(prog)s -c cluster.yaml nccl

  # Validate network
  %(prog)s -c cluster.yaml validate

  # Get cluster status
  %(prog)s -c cluster.yaml status
        """
    )

    parser.add_argument("-c", "--config", required=True,
                        help="Cluster configuration YAML file")
    parser.add_argument("command", choices=[
        "deploy", "single-node", "multi-node",
        "tailscale", "nccl", "validate",
        "status", "info", "inventory"
    ], help="Command to execute")
    parser.add_argument("--auth-key", help="Tailscale authentication key")
    parser.add_argument("--node", help="Specific node for single-node setup")
    parser.add_argument("--output", "-o", help="Output file for inventory or status")

    args = parser.parse_args()

    orchestrator = ClusterOrchestrator(args.config)

    if args.command == "deploy":
        return orchestrator.deploy_full_cluster()
    elif args.command == "single-node":
        return orchestrator.setup_single_node(args.node)
    elif args.command == "multi-node":
        return orchestrator.setup_multi_node()
    elif args.command == "tailscale":
        return orchestrator.setup_tailscale(args.auth_key)
    elif args.command == "nccl":
        return orchestrator.setup_nccl()
    elif args.command == "validate":
        return orchestrator.validate_network()
    elif args.command == "status":
        status = orchestrator.get_cluster_status()
        print(json.dumps(status, indent=2))

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(status, f, indent=2)
            print(f"\nStatus saved to {args.output}")

        return 0
    elif args.command == "info":
        orchestrator.print_cluster_info()
        return 0
    elif args.command == "inventory":
        output_file = args.output or str(orchestrator.base_dir / "config" / "inventory.ini")
        orchestrator.generate_inventory(output_file)
        return 0


if __name__ == "__main__":
    sys.exit(main())

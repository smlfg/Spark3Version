#!/usr/bin/env python3
"""
Network Validation Tool for Spark Cluster
Validates network connectivity, bandwidth, and latency between cluster nodes
"""

import argparse
import json
import subprocess
import sys
import time
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import socket
import os


class NetworkValidator:
    def __init__(self, nodes: List[str], ssh_key: str = None):
        self.nodes = nodes
        self.ssh_key = ssh_key or os.path.expanduser("~/.ssh/dgx_key")
        self.results = {}

    def run_ssh_command(self, host: str, command: str, timeout: int = 30) -> Tuple[int, str, str]:
        """Execute command on remote host via SSH"""
        ssh_cmd = [
            "ssh",
            "-i", self.ssh_key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=5",
            "-o", "BatchMode=yes",
            host,
            command
        ]

        try:
            result = subprocess.run(
                ssh_cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timeout"
        except Exception as e:
            return -1, "", str(e)

    def check_ssh_connectivity(self, host: str) -> Dict:
        """Check SSH connectivity to a host"""
        print(f"[SSH] Testing connectivity to {host}...")

        start_time = time.time()
        returncode, stdout, stderr = self.run_ssh_command(host, "hostname && uptime")
        elapsed = time.time() - start_time

        result = {
            "host": host,
            "reachable": returncode == 0,
            "response_time": round(elapsed, 3),
            "hostname": stdout.split('\n')[0] if returncode == 0 else None,
            "error": stderr if returncode != 0 else None
        }

        status = "✓" if result["reachable"] else "✗"
        print(f"[SSH] {status} {host} - {result['response_time']}s")

        return result

    def check_network_latency(self, host: str) -> Dict:
        """Measure network latency using ping"""
        print(f"[PING] Testing latency to {host}...")

        try:
            result = subprocess.run(
                ["ping", "-c", "10", "-i", "0.2", host],
                capture_output=True,
                text=True,
                timeout=10
            )

            if result.returncode == 0:
                # Parse ping output
                lines = result.stdout.split('\n')
                stats_line = [l for l in lines if "min/avg/max" in l]

                if stats_line:
                    stats = stats_line[0].split('=')[1].split('/')[:-1]
                    latency = {
                        "min": float(stats[0]),
                        "avg": float(stats[1]),
                        "max": float(stats[2])
                    }
                else:
                    latency = None

                packet_loss_line = [l for l in lines if "packet loss" in l]
                packet_loss = float(packet_loss_line[0].split('%')[0].split()[-1]) if packet_loss_line else 0

                result_data = {
                    "host": host,
                    "reachable": True,
                    "latency_ms": latency,
                    "packet_loss_percent": packet_loss
                }
            else:
                result_data = {
                    "host": host,
                    "reachable": False,
                    "error": "Ping failed"
                }

            status = "✓" if result_data["reachable"] else "✗"
            latency_str = f"{result_data['latency_ms']['avg']:.2f}ms" if result_data.get('latency_ms') else "N/A"
            print(f"[PING] {status} {host} - {latency_str}")

            return result_data

        except Exception as e:
            return {
                "host": host,
                "reachable": False,
                "error": str(e)
            }

    def check_bandwidth(self, host: str, port: int = 5201) -> Dict:
        """Measure network bandwidth using iperf3"""
        print(f"[BANDWIDTH] Testing bandwidth to {host}...")

        # Check if iperf3 is available
        returncode, _, _ = self.run_ssh_command(host, "which iperf3")
        if returncode != 0:
            return {
                "host": host,
                "error": "iperf3 not installed on remote host"
            }

        # Start iperf3 server on remote host
        server_cmd = f"iperf3 -s -p {port} -D"
        returncode, _, stderr = self.run_ssh_command(host, server_cmd)

        if returncode != 0:
            return {
                "host": host,
                "error": f"Failed to start iperf3 server: {stderr}"
            }

        # Wait for server to start
        time.sleep(1)

        try:
            # Run iperf3 client
            result = subprocess.run(
                ["iperf3", "-c", host, "-p", str(port), "-t", "5", "-J"],
                capture_output=True,
                text=True,
                timeout=15
            )

            # Kill iperf3 server
            self.run_ssh_command(host, f"pkill -9 -f 'iperf3 -s -p {port}'")

            if result.returncode == 0:
                data = json.loads(result.stdout)
                bandwidth_gbps = data['end']['sum_received']['bits_per_second'] / 1e9

                result_data = {
                    "host": host,
                    "bandwidth_gbps": round(bandwidth_gbps, 2),
                    "bandwidth_mbps": round(bandwidth_gbps * 1000, 2),
                    "retransmits": data['end']['sum_sent']['retransmits']
                }

                print(f"[BANDWIDTH] ✓ {host} - {result_data['bandwidth_gbps']:.2f} Gbps")
                return result_data
            else:
                return {
                    "host": host,
                    "error": f"iperf3 test failed: {result.stderr}"
                }

        except Exception as e:
            # Cleanup
            self.run_ssh_command(host, f"pkill -9 -f 'iperf3 -s -p {port}'")
            return {
                "host": host,
                "error": str(e)
            }

    def check_gpu_topology(self, host: str) -> Dict:
        """Check GPU topology on remote host"""
        print(f"[GPU] Checking GPU topology on {host}...")

        returncode, stdout, stderr = self.run_ssh_command(host, "nvidia-smi topo -m")

        if returncode == 0:
            return {
                "host": host,
                "has_gpu": True,
                "topology": stdout
            }
        else:
            return {
                "host": host,
                "has_gpu": False,
                "error": stderr
            }

    def validate_cluster(self, tests: List[str] = None) -> Dict:
        """Run validation tests on all cluster nodes"""
        if tests is None:
            tests = ["ssh", "ping", "bandwidth", "gpu"]

        print(f"\n{'='*60}")
        print(f"Network Validation for {len(self.nodes)} nodes")
        print(f"Tests: {', '.join(tests)}")
        print(f"{'='*60}\n")

        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nodes": self.nodes,
            "tests": {}
        }

        # SSH connectivity test
        if "ssh" in tests:
            print("\n--- SSH Connectivity Test ---")
            with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
                futures = {executor.submit(self.check_ssh_connectivity, node): node for node in self.nodes}
                ssh_results = []
                for future in as_completed(futures):
                    ssh_results.append(future.result())
            results["tests"]["ssh"] = ssh_results

        # Network latency test
        if "ping" in tests:
            print("\n--- Network Latency Test ---")
            with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
                futures = {executor.submit(self.check_network_latency, node): node for node in self.nodes}
                ping_results = []
                for future in as_completed(futures):
                    ping_results.append(future.result())
            results["tests"]["ping"] = ping_results

        # Bandwidth test
        if "bandwidth" in tests:
            print("\n--- Bandwidth Test ---")
            bandwidth_results = []
            for node in self.nodes:
                bandwidth_results.append(self.check_bandwidth(node))
            results["tests"]["bandwidth"] = bandwidth_results

        # GPU topology test
        if "gpu" in tests:
            print("\n--- GPU Topology Test ---")
            with ThreadPoolExecutor(max_workers=len(self.nodes)) as executor:
                futures = {executor.submit(self.check_gpu_topology, node): node for node in self.nodes}
                gpu_results = []
                for future in as_completed(futures):
                    gpu_results.append(future.result())
            results["tests"]["gpu"] = gpu_results

        return results

    def print_summary(self, results: Dict):
        """Print validation summary"""
        print(f"\n{'='*60}")
        print("VALIDATION SUMMARY")
        print(f"{'='*60}\n")

        # SSH Summary
        if "ssh" in results["tests"]:
            ssh_results = results["tests"]["ssh"]
            reachable = sum(1 for r in ssh_results if r["reachable"])
            print(f"SSH Connectivity: {reachable}/{len(ssh_results)} nodes reachable")

        # Ping Summary
        if "ping" in results["tests"]:
            ping_results = results["tests"]["ping"]
            reachable = sum(1 for r in ping_results if r["reachable"])
            if reachable > 0:
                avg_latencies = [r["latency_ms"]["avg"] for r in ping_results if r.get("latency_ms")]
                avg_latency = sum(avg_latencies) / len(avg_latencies) if avg_latencies else 0
                print(f"Network Latency: {reachable}/{len(ping_results)} nodes reachable, avg {avg_latency:.2f}ms")

        # Bandwidth Summary
        if "bandwidth" in results["tests"]:
            bandwidth_results = [r for r in results["tests"]["bandwidth"] if "bandwidth_gbps" in r]
            if bandwidth_results:
                avg_bandwidth = sum(r["bandwidth_gbps"] for r in bandwidth_results) / len(bandwidth_results)
                print(f"Network Bandwidth: avg {avg_bandwidth:.2f} Gbps across {len(bandwidth_results)} nodes")

        # GPU Summary
        if "gpu" in results["tests"]:
            gpu_results = results["tests"]["gpu"]
            gpu_nodes = sum(1 for r in gpu_results if r.get("has_gpu", False))
            print(f"GPU Availability: {gpu_nodes}/{len(gpu_results)} nodes have GPUs")

        print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Spark Cluster Network Validation Tool")
    parser.add_argument("nodes", nargs="+", help="List of cluster node hostnames or IPs")
    parser.add_argument("--ssh-key", default=None, help="Path to SSH private key")
    parser.add_argument("--tests", nargs="+", choices=["ssh", "ping", "bandwidth", "gpu"],
                        default=["ssh", "ping", "bandwidth", "gpu"],
                        help="Tests to run")
    parser.add_argument("--output", "-o", help="Output JSON file for results")

    args = parser.parse_args()

    validator = NetworkValidator(args.nodes, args.ssh_key)
    results = validator.validate_cluster(args.tests)
    validator.print_summary(results)

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()

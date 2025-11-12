#!/usr/bin/env python3
"""
Shared Health Check API for All Agents
Provides standardized health checking capabilities across the cluster
"""

import argparse
import json
import os
import socket
import sys
import time
from typing import Dict, Optional, List
import psutil

try:
    import requests
except ImportError:
    requests = None


class HealthCheck:
    """Health check utility for services and system resources"""

    def __init__(self):
        self.hostname = socket.gethostname()
        self.start_time = time.time()

    def check_service(
        self,
        port: int,
        endpoint: str = "/health",
        protocol: str = "http",
        host: str = "localhost",
        timeout: int = 5
    ) -> bool:
        """
        Check if a service is healthy by making HTTP request

        Args:
            port: Service port number
            endpoint: Health check endpoint path
            protocol: Protocol (http or https)
            host: Host to check (default: localhost)
            timeout: Request timeout in seconds

        Returns:
            True if service is healthy (status code 200), False otherwise
        """
        if requests is None:
            print("Warning: requests library not available, using basic socket check")
            return self.check_port(host, port, timeout)

        url = f"{protocol}://{host}:{port}{endpoint}"

        try:
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException as e:
            print(f"Health check failed for {url}: {e}", file=sys.stderr)
            return False

    def check_port(self, host: str, port: int, timeout: int = 5) -> bool:
        """
        Check if a port is open and accepting connections

        Args:
            host: Host to check
            port: Port number
            timeout: Connection timeout in seconds

        Returns:
            True if port is open, False otherwise
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except socket.error as e:
            print(f"Port check failed for {host}:{port}: {e}", file=sys.stderr)
            return False

    def check_system_resources(self) -> Dict:
        """
        Check system resource usage

        Returns:
            Dictionary with CPU, memory, and disk usage information
        """
        resources = {
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else None
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "used": psutil.virtual_memory().used
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "used": psutil.disk_usage('/').used,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            }
        }

        return resources

    def check_gpu(self) -> Optional[Dict]:
        """
        Check GPU availability and usage (if available)

        Returns:
            Dictionary with GPU information or None if no GPU
        """
        try:
            import pynvml
            pynvml.nvmlInit()

            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []

            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)

                gpus.append({
                    "id": i,
                    "name": name.decode('utf-8') if isinstance(name, bytes) else name,
                    "memory": {
                        "total": memory.total,
                        "used": memory.used,
                        "free": memory.free,
                        "percent": (memory.used / memory.total) * 100
                    },
                    "utilization": {
                        "gpu": utilization.gpu,
                        "memory": utilization.memory
                    },
                    "temperature": temperature
                })

            pynvml.nvmlShutdown()

            return {
                "available": True,
                "count": device_count,
                "devices": gpus
            }

        except (ImportError, Exception) as e:
            return {
                "available": False,
                "error": str(e)
            }

    def check_network_interfaces(self) -> Dict:
        """
        Check network interface status

        Returns:
            Dictionary with network interface information
        """
        interfaces = {}

        for interface, addrs in psutil.net_if_addrs().items():
            if_stats = psutil.net_if_stats().get(interface)

            interfaces[interface] = {
                "addresses": [
                    {
                        "family": addr.family.name,
                        "address": addr.address,
                        "netmask": addr.netmask
                    }
                    for addr in addrs
                ],
                "is_up": if_stats.isup if if_stats else False,
                "speed": if_stats.speed if if_stats else None,
                "mtu": if_stats.mtu if if_stats else None
            }

        return interfaces

    def check_processes(self, process_names: List[str] = None) -> Dict:
        """
        Check if specific processes are running

        Args:
            process_names: List of process names to check

        Returns:
            Dictionary with process status
        """
        if process_names is None:
            return {}

        processes = {}

        for proc in psutil.process_iter(['pid', 'name', 'status', 'cpu_percent', 'memory_percent']):
            try:
                if proc.info['name'] in process_names:
                    processes[proc.info['name']] = {
                        "running": True,
                        "pid": proc.info['pid'],
                        "status": proc.info['status'],
                        "cpu_percent": proc.info['cpu_percent'],
                        "memory_percent": proc.info['memory_percent']
                    }
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        # Mark missing processes
        for name in process_names:
            if name not in processes:
                processes[name] = {
                    "running": False
                }

        return processes

    def self_check(self) -> Dict:
        """
        Perform comprehensive self health check

        Returns:
            Dictionary with complete health status
        """
        uptime = time.time() - self.start_time

        health = {
            "status": "healthy",
            "timestamp": time.time(),
            "hostname": self.hostname,
            "uptime": uptime,
            "system": self.check_system_resources(),
            "network": self.check_network_interfaces()
        }

        # Add GPU info if available
        gpu_info = self.check_gpu()
        if gpu_info and gpu_info.get("available"):
            health["gpu"] = gpu_info

        # Check resource thresholds
        if health["system"]["cpu"]["percent"] > 90:
            health["status"] = "degraded"
            health["warnings"] = health.get("warnings", [])
            health["warnings"].append("High CPU usage")

        if health["system"]["memory"]["percent"] > 90:
            health["status"] = "degraded"
            health["warnings"] = health.get("warnings", [])
            health["warnings"].append("High memory usage")

        if health["system"]["disk"]["percent"] > 90:
            health["status"] = "degraded"
            health["warnings"] = health.get("warnings", [])
            health["warnings"].append("High disk usage")

        return health

    def get_status_code(self) -> int:
        """
        Get HTTP status code based on health

        Returns:
            200 for healthy, 503 for unhealthy
        """
        health = self.self_check()
        return 200 if health["status"] in ["healthy", "degraded"] else 503


def check_service(
    port: int,
    endpoint: str = "/health",
    host: str = "localhost",
    protocol: str = "http",
    timeout: int = 5
) -> bool:
    """
    Convenience function to check service health

    Args:
        port: Service port
        endpoint: Health check endpoint
        host: Host to check
        protocol: Protocol (http/https)
        timeout: Request timeout

    Returns:
        True if healthy, False otherwise
    """
    checker = HealthCheck()
    return checker.check_service(port, endpoint, protocol, host, timeout)


def main():
    parser = argparse.ArgumentParser(description="Health Check Utility")
    parser.add_argument("--port", type=int, help="Port to check")
    parser.add_argument("--endpoint", default="/health", help="Health check endpoint")
    parser.add_argument("--host", default="localhost", help="Host to check")
    parser.add_argument("--protocol", default="http", choices=["http", "https"], help="Protocol")
    parser.add_argument("--timeout", type=int, default=5, help="Request timeout")
    parser.add_argument("--self-check", action="store_true", help="Perform self health check")
    parser.add_argument("--json", action="store_true", help="Output in JSON format")
    parser.add_argument("--check-gpu", action="store_true", help="Check GPU status")
    parser.add_argument("--check-process", nargs="+", help="Check if processes are running")

    args = parser.parse_args()

    checker = HealthCheck()

    if args.self_check:
        health = checker.self_check()
        if args.json:
            print(json.dumps(health, indent=2))
        else:
            print(f"Status: {health['status']}")
            print(f"Hostname: {health['hostname']}")
            print(f"Uptime: {health['uptime']:.2f}s")
            print(f"CPU: {health['system']['cpu']['percent']:.1f}%")
            print(f"Memory: {health['system']['memory']['percent']:.1f}%")
            print(f"Disk: {health['system']['disk']['percent']:.1f}%")

            if "warnings" in health:
                print(f"Warnings: {', '.join(health['warnings'])}")

        sys.exit(0 if health["status"] in ["healthy", "degraded"] else 1)

    elif args.check_gpu:
        gpu_info = checker.check_gpu()
        if args.json:
            print(json.dumps(gpu_info, indent=2))
        else:
            if gpu_info.get("available"):
                print(f"GPUs: {gpu_info['count']}")
                for gpu in gpu_info['devices']:
                    print(f"  GPU {gpu['id']}: {gpu['name']}")
                    print(f"    Memory: {gpu['memory']['percent']:.1f}% used")
                    print(f"    Utilization: {gpu['utilization']['gpu']}%")
                    print(f"    Temperature: {gpu['temperature']}°C")
            else:
                print(f"GPU not available: {gpu_info.get('error', 'Unknown')}")

        sys.exit(0 if gpu_info.get("available") else 1)

    elif args.check_process:
        processes = checker.check_processes(args.check_process)
        if args.json:
            print(json.dumps(processes, indent=2))
        else:
            for name, info in processes.items():
                status = "Running" if info["running"] else "Not running"
                print(f"{name}: {status}")
                if info["running"]:
                    print(f"  PID: {info['pid']}")
                    print(f"  CPU: {info['cpu_percent']}%")
                    print(f"  Memory: {info['memory_percent']:.1f}%")

        all_running = all(p["running"] for p in processes.values())
        sys.exit(0 if all_running else 1)

    elif args.port:
        healthy = checker.check_service(args.port, args.endpoint, args.protocol, args.host, args.timeout)
        if args.json:
            print(json.dumps({"healthy": healthy, "port": args.port, "endpoint": args.endpoint}))
        else:
            status = "healthy" if healthy else "unhealthy"
            print(f"Service at {args.host}:{args.port}{args.endpoint} is {status}")

        sys.exit(0 if healthy else 1)

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

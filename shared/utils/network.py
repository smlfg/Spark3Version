"""
Network Utilities
Common network operations for agents
"""

import socket
import subprocess
from typing import Optional, List, Dict
import time


class NetworkUtils:
    """Network utility functions"""

    @staticmethod
    def get_local_ip() -> str:
        """Get local IP address"""
        try:
            # Connect to external host to determine local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"

    @staticmethod
    def get_hostname() -> str:
        """Get system hostname"""
        return socket.gethostname()

    @staticmethod
    def resolve_hostname(hostname: str) -> Optional[str]:
        """
        Resolve hostname to IP address

        Args:
            hostname: Hostname to resolve

        Returns:
            IP address or None if resolution fails
        """
        try:
            return socket.gethostbyname(hostname)
        except socket.gaierror:
            return None

    @staticmethod
    def check_port_open(host: str, port: int, timeout: int = 5) -> bool:
        """
        Check if a port is open

        Args:
            host: Host to check
            port: Port number
            timeout: Connection timeout

        Returns:
            True if port is open, False otherwise
        """
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)

        try:
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except socket.error:
            return False

    @staticmethod
    def wait_for_port(
        host: str,
        port: int,
        timeout: int = 60,
        interval: int = 1
    ) -> bool:
        """
        Wait for a port to become available

        Args:
            host: Host to check
            port: Port number
            timeout: Maximum wait time in seconds
            interval: Check interval in seconds

        Returns:
            True if port becomes available, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if NetworkUtils.check_port_open(host, port, timeout=1):
                return True
            time.sleep(interval)

        return False

    @staticmethod
    def get_open_port(start_port: int = 8000, end_port: int = 9000) -> Optional[int]:
        """
        Find an available port in range

        Args:
            start_port: Start of port range
            end_port: End of port range

        Returns:
            Available port number or None if no port available
        """
        for port in range(start_port, end_port):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(("", port))
                sock.close()
                return port
            except OSError:
                continue

        return None

    @staticmethod
    def ping(host: str, count: int = 4, timeout: int = 5) -> Dict:
        """
        Ping a host

        Args:
            host: Host to ping
            count: Number of ping packets
            timeout: Timeout in seconds

        Returns:
            Dictionary with ping results
        """
        try:
            result = subprocess.run(
                ["ping", "-c", str(count), "-W", str(timeout), host],
                capture_output=True,
                text=True,
                timeout=timeout * count
            )

            success = result.returncode == 0

            # Parse output for statistics
            if success and "min/avg/max" in result.stdout:
                stats_line = [l for l in result.stdout.split('\n') if "min/avg/max" in l][0]
                stats = stats_line.split('=')[1].split('/')[:-1]

                return {
                    "success": True,
                    "host": host,
                    "packets_sent": count,
                    "latency_ms": {
                        "min": float(stats[0]),
                        "avg": float(stats[1]),
                        "max": float(stats[2])
                    }
                }

            return {
                "success": success,
                "host": host,
                "packets_sent": count
            }

        except Exception as e:
            return {
                "success": False,
                "host": host,
                "error": str(e)
            }

    @staticmethod
    def get_network_interfaces() -> List[Dict]:
        """
        Get list of network interfaces

        Returns:
            List of network interface information
        """
        try:
            import psutil
            interfaces = []

            for name, addrs in psutil.net_if_addrs().items():
                interface_info = {
                    "name": name,
                    "addresses": []
                }

                for addr in addrs:
                    interface_info["addresses"].append({
                        "family": str(addr.family),
                        "address": addr.address,
                        "netmask": addr.netmask
                    })

                interfaces.append(interface_info)

            return interfaces

        except ImportError:
            return []

    @staticmethod
    def test_bandwidth(host: str, port: int = 5201, duration: int = 5) -> Optional[Dict]:
        """
        Test network bandwidth using iperf3

        Args:
            host: Target host
            port: iperf3 port
            duration: Test duration in seconds

        Returns:
            Bandwidth test results or None if iperf3 not available
        """
        try:
            result = subprocess.run(
                ["iperf3", "-c", host, "-p", str(port), "-t", str(duration), "-J"],
                capture_output=True,
                text=True,
                timeout=duration + 10
            )

            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)

                return {
                    "success": True,
                    "host": host,
                    "bandwidth_bps": data['end']['sum_received']['bits_per_second'],
                    "bandwidth_gbps": data['end']['sum_received']['bits_per_second'] / 1e9,
                    "retransmits": data['end']['sum_sent'].get('retransmits', 0)
                }

            return {
                "success": False,
                "error": result.stderr
            }

        except (FileNotFoundError, subprocess.TimeoutExpired, Exception) as e:
            return {
                "success": False,
                "error": str(e)
            }

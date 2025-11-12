"""
GPU Utilities
Common GPU operations and monitoring for agents
"""

from typing import Optional, List, Dict
import subprocess
import os


class GPUUtils:
    """GPU utility functions"""

    @staticmethod
    def is_available() -> bool:
        """Check if GPU is available"""
        try:
            result = subprocess.run(
                ["nvidia-smi"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def get_gpu_count() -> int:
        """Get number of available GPUs"""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=count", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # nvidia-smi returns count for each GPU, so just count lines
                return len(result.stdout.strip().split('\n'))

            return 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return 0

    @staticmethod
    def get_gpu_info(gpu_id: Optional[int] = None) -> List[Dict]:
        """
        Get detailed GPU information

        Args:
            gpu_id: Specific GPU ID or None for all GPUs

        Returns:
            List of GPU information dictionaries
        """
        try:
            query = [
                "index",
                "name",
                "driver_version",
                "memory.total",
                "memory.used",
                "memory.free",
                "utilization.gpu",
                "utilization.memory",
                "temperature.gpu",
                "power.draw",
                "power.limit"
            ]

            cmd = [
                "nvidia-smi",
                f"--query-gpu={','.join(query)}",
                "--format=csv,noheader,nounits"
            ]

            if gpu_id is not None:
                cmd.extend(["-i", str(gpu_id)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return []

            gpus = []
            for line in result.stdout.strip().split('\n'):
                values = [v.strip() for v in line.split(',')]

                gpu_info = {
                    "id": int(values[0]),
                    "name": values[1],
                    "driver_version": values[2],
                    "memory": {
                        "total_mb": float(values[3]),
                        "used_mb": float(values[4]),
                        "free_mb": float(values[5]),
                        "utilization_percent": float(values[6])
                    },
                    "utilization": {
                        "gpu_percent": float(values[6]),
                        "memory_percent": float(values[7])
                    },
                    "temperature_c": float(values[8]),
                    "power": {
                        "draw_w": float(values[9]),
                        "limit_w": float(values[10])
                    }
                }

                gpus.append(gpu_info)

            return gpus

        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return []

    @staticmethod
    def get_cuda_version() -> Optional[str]:
        """Get CUDA version"""
        try:
            result = subprocess.run(
                ["nvcc", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                # Parse version from output
                for line in result.stdout.split('\n'):
                    if "release" in line.lower():
                        version = line.split("release")[1].split(",")[0].strip()
                        return version

            return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def set_visible_devices(gpu_ids: List[int]):
        """
        Set CUDA_VISIBLE_DEVICES environment variable

        Args:
            gpu_ids: List of GPU IDs to make visible
        """
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))

    @staticmethod
    def get_visible_devices() -> Optional[List[int]]:
        """
        Get currently visible GPU devices

        Returns:
            List of visible GPU IDs or None if not set
        """
        devices = os.environ.get("CUDA_VISIBLE_DEVICES")

        if devices is None:
            return None

        if devices == "":
            return []

        try:
            return [int(d.strip()) for d in devices.split(",")]
        except ValueError:
            return None

    @staticmethod
    def check_nccl() -> bool:
        """Check if NCCL is available"""
        try:
            result = subprocess.run(
                ["dpkg", "-l", "libnccl2"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def get_gpu_topology() -> Optional[str]:
        """
        Get GPU topology information

        Returns:
            GPU topology string or None
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "topo", "-m"],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                return result.stdout

            return None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def reset_gpu(gpu_id: int) -> bool:
        """
        Reset GPU (requires root privileges)

        Args:
            gpu_id: GPU ID to reset

        Returns:
            True if successful, False otherwise
        """
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(gpu_id), "-r"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    @staticmethod
    def get_processes(gpu_id: Optional[int] = None) -> List[Dict]:
        """
        Get list of processes using GPU

        Args:
            gpu_id: Specific GPU ID or None for all GPUs

        Returns:
            List of process information
        """
        try:
            cmd = [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits"
            ]

            if gpu_id is not None:
                cmd.extend(["-i", str(gpu_id)])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode != 0:
                return []

            processes = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue

                values = [v.strip() for v in line.split(',')]

                processes.append({
                    "pid": int(values[0]),
                    "name": values[1],
                    "memory_mb": float(values[2])
                })

            return processes

        except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
            return []

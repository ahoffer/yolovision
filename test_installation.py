#!/usr/bin/env python3
"""
Simple Installation Checker for ML/DL Development Environment
Checks dependencies and configurations needed for ML/DL development.
"""

import importlib.metadata
import logging
import os
import subprocess
import sys
from typing import Dict, List, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_command(cmd: List[str]) -> tuple[bool, str]:
    """Simple command runner"""
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        return True, result.stdout.strip()
    except Exception as e:
        return False, str(e)


def check_package(package_name: str) -> Dict[str, str]:
    """Check if a Python package is installed"""
    try:
        version = importlib.metadata.version(package_name)
        return {"status": "installed", "version": version}
    except importlib.metadata.PackageNotFoundError:
        return {"status": "not installed", "version": "none"}
    except Exception as e:
        logger.error(f"Error checking package {package_name}: {e}")
        return {"status": "error", "version": "unknown"}


def check_system() -> Dict[str, str]:
    """Get basic system information"""
    info = {"os": "unknown", "kernel": "unknown", "arch": "unknown", "python": sys.version.split()[0]}

    try:
        # Get OS info
        if os.path.exists('/etc/os-release'):
            with open('/etc/os-release') as f:
                for line in f:
                    if line.startswith('PRETTY_NAME='):
                        info['os'] = line.split('=')[1].strip().strip('"')
                        break

        # Get kernel and architecture
        success, kernel = run_command(['uname', '-r'])
        if success:
            info['kernel'] = kernel

        success, arch = run_command(['uname', '-m'])
        if success:
            info['arch'] = arch

    except Exception as e:
        logger.error(f"Error getting system info: {e}")

    return info


def check_cuda() -> Dict[str, str]:
    """Check CUDA installation"""
    cuda_info = {"available": "no", "version": "not found", "device_name": "no gpu", "device_count": "0",
        "nvidia_smi": "not available"}

    try:
        import torch
        cuda_info["available"] = str(torch.cuda.is_available())

        if torch.cuda.is_available():
            cuda_info["version"] = torch.version.cuda
            cuda_info["device_name"] = torch.cuda.get_device_name(0)
            cuda_info["device_count"] = str(torch.cuda.device_count())

            # Check nvidia-smi
            success, _ = run_command(['nvidia-smi'])
            if success:
                cuda_info["nvidia_smi"] = "available"
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")

    return cuda_info


def check_gstreamer() -> Dict[str, Any]:
    """Check GStreamer installation"""
    gst_info = {"installed": "no", "version": "not found", "python_support": "no", "plugins": []}

    # Check system GStreamer
    success, output = run_command(['gst-launch-1.0', '--version'])
    if success:
        gst_info["installed"] = "yes"
        version_line = [line for line in output.split('\n') if line.startswith('gst-launch-1.0 version')]
        if version_line:
            gst_info["version"] = version_line[0].split()[-1]

        # Check plugins
        success, plugins_output = run_command(['gst-inspect-1.0'])
        if success:
            gst_info["plugins"] = [line.split(':')[0].strip() for line in plugins_output.split('\n') if ':' in line]

    # Check Python GStreamer support
    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst
        Gst.init(None)
        gst_info["python_support"] = "yes"
    except:
        pass

    return gst_info


def check_tensorrt() -> Dict[str, Any]:
    """Check TensorRT installation"""
    trt_info = {"installed": "no", "version": "not found", "test_passed": "no"}

    try:
        import tensorrt as trt
        trt_info["installed"] = "yes"
        trt_info["version"] = trt.__version__

        # Simple test
        logger = trt.Logger()
        builder = trt.Builder(logger)
        if builder:
            trt_info["test_passed"] = "yes"
    except:
        pass

    return trt_info


def check_environment() -> Dict[str, str]:
    """Check relevant environment variables"""
    return {"CUDA_HOME": os.getenv("CUDA_HOME", "not set"),
        "LD_LIBRARY_PATH": os.getenv("LD_LIBRARY_PATH", "not set"), }


def print_results(results: Dict[str, Any]) -> None:
    """Print results in a readable format"""
    print("\n=== ML/DL Environment Check Results ===")

    print("\nSystem Information:")
    print(f"OS: {results['system']['os']}")
    print(f"Kernel: {results['system']['kernel']}")
    print(f"Architecture: {results['system']['arch']}")
    print(f"Python: {results['system']['python']}")

    print("\nPackages:")
    for pkg, info in results['packages'].items():
        print(f"{pkg}: {info['version']}")

    print("\nCUDA:")
    print(f"Available: {results['cuda']['available']}")
    print(f"Version: {results['cuda']['version']}")
    print(f"GPU: {results['cuda']['device_name']}")
    print(f"GPU Count: {results['cuda']['device_count']}")
    print(f"nvidia-smi: {results['cuda']['nvidia_smi']}")

    print("\nGStreamer:")
    print(f"Installed: {results['gstreamer']['installed']}")
    print(f"Version: {results['gstreamer']['version']}")
    print(f"Python Support: {results['gstreamer']['python_support']}")
    if results['gstreamer']['plugins']:
        print("Key plugins:", ', '.join(p for p in ['v4l2', 'videoconvert', 'x264', 'nvenc', 'nvdec'] if
                                        any(p in plugin.lower() for plugin in results['gstreamer']['plugins'])))

    print("\nTensorRT:")
    print(f"Installed: {results['tensorrt']['installed']}")
    print(f"Version: {results['tensorrt']['version']}")
    print(f"Test: {results['tensorrt']['test_passed']}")

    print("\nEnvironment Variables:")
    for var, value in results['env'].items():
        print(f"{var}: {value}")


def main():
    """Main function to run all checks"""
    try:
        results = {"system": check_system(), "packages": {pkg: check_package(pkg) for pkg in
            ["torch", "torchvision", "numpy", "opencv-python", "albumentations", "tensorboard"]}, "cuda": check_cuda(),
            "gstreamer": check_gstreamer(), "tensorrt": check_tensorrt(), "env": check_environment()}

        print_results(results)

    except Exception as e:
        logger.error(f"Error during checks: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

"""
Utilities for monitoring GPU memory and usage.
"""

import logging
import subprocess
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)

def get_gpu_memory_info():
    """Get GPU memory usage directly from nvidia-smi"""
    try:
        cmd = ['nvidia-smi', '-q', '-x']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return None
            
        root = ET.fromstring(result.stdout)
        gpu_info = []
        
        for gpu in root.findall("gpu"):
            memory = gpu.find("fb_memory_usage")
            gpu_dict = {
                'id': gpu.find("minor_number").text,
                'total': memory.find("total").text.replace('MiB', '').strip(),
                'used': memory.find("used").text.replace('MiB', '').strip(),
                'free': memory.find("free").text.replace('MiB', '').strip(),
                'processes': []
            }
            
            processes = gpu.find("processes")
            if processes is not None:
                for process in processes.findall("process_info"):
                    pid = process.find("pid").text
                    used_memory = process.find("used_memory").text
                    gpu_dict['processes'].append({
                        'pid': pid,
                        'used_memory': used_memory.replace('MiB', '').strip()
                    })
                    
            gpu_info.append(gpu_dict)
            
        return gpu_info
    except Exception as e:
        logger.error(f"Error getting GPU info: {str(e)}")
        return None

def log_memory_usage(tag=""):
    """Log both PyTorch and nvidia-smi memory stats"""
    try:
        # nvidia-smi stats
        gpu_info = get_gpu_memory_info()
        if gpu_info:
            for gpu in gpu_info:
                logger.info(f"GPU {gpu['id']} (nvidia-smi) - Used: {gpu['used']}MB, Free: {gpu['free']}MB, Total: {gpu['total']}MB")
                if gpu['processes']:
                    process_info = [f"PID {p['pid']}: {p['used_memory']}MB" for p in gpu['processes']]
                    logger.info(f"Active processes: {', '.join(process_info)}")
    except Exception as e:
        logger.error(f"Error logging memory usage: {str(e)}")
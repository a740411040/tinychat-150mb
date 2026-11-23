"""Record the actual local environment without changing system configuration."""
import json
from pathlib import Path
import platform
import sys

import psutil
import torch


def main():
    memory = psutil.virtual_memory()
    report = {"python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "torch": torch.__version__,
              "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
              "ram_gib": memory.total / 2**30, "available_ram_gib": memory.available / 2**30,
              "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False)}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        report.update(gpu=props.name, gpu_total_mib=props.total_memory/2**20,
                      compute_capability=list(torch.cuda.get_device_capability()),
                      compiled_archs=torch.cuda.get_arch_list(),
                      bf16_supported=torch.cuda.is_bf16_supported(),
                      gpu_free_mib=torch.cuda.mem_get_info()[0]/2**20)
    Path("logs").mkdir(exist_ok=True)
    Path("logs/environment.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["cuda_available"]:
        raise SystemExit("CUDA is unavailable; refusing to benchmark on CPU")


if __name__ == "__main__":
    main()

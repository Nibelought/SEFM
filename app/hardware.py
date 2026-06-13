"""Hardware probe and acceleration planner.

Detects CPU, RAM, GPUs and NPUs, plus whether the *installed* runtime
(llama-cpp-python build, torch) can actually use a GPU, and produces an
AccelerationPlan (threads, GPU offload depth, embedding device).

GPU offload is only enabled when a GPU is present AND the installed wheel was
built with GPU support; the default CPU-only wheel reports
llama_supports_gpu_offload() == False. NPUs are detected and reported only.

Heavy deps (psutil, pynvml, torch, llama_cpp) are imported inside guarded
functions, so the module and planner run with none of them installed.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import platform
import re
import subprocess
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from app.config import Settings

log = logging.getLogger(__name__)

Vendor = Literal["nvidia", "intel", "amd", "unknown"]
GpuKind = Literal["discrete", "integrated", "unknown"]
Device = Literal["cpu", "cuda", "xpu"]

# Max seconds for any external probe (nvidia-smi, PowerShell CIM) before fallback.
_PROBE_TIMEOUT_S = 8.0


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class CpuInfo:
    brand: str
    physical_cores: int
    logical_cores: int
    ram_total_gb: float


@dataclass(frozen=True, slots=True)
class GpuInfo:
    vendor: Vendor
    name: str
    kind: GpuKind = "unknown"
    vram_mb: int | None = None


@dataclass(frozen=True, slots=True)
class NpuInfo:
    name: str


@dataclass(frozen=True, slots=True)
class RuntimeCaps:
    """What the installed runtime can use, not what hardware exists."""

    llama_gpu_offload: bool = False
    torch_cuda: bool = False
    torch_xpu: bool = False


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    cpu: CpuInfo
    gpus: list[GpuInfo] = field(default_factory=list)
    npus: list[NpuInfo] = field(default_factory=list)
    caps: RuntimeCaps = field(default_factory=RuntimeCaps)

    @classmethod
    def unknown(cls) -> HardwareProfile:
        """Used when detection is disabled (SEFM_HW_DETECT=false)."""
        return cls(cpu=CpuInfo(brand="unknown", physical_cores=0, logical_cores=0, ram_total_gb=0.0))


@dataclass(frozen=True, slots=True)
class AccelerationPlan:
    llm_n_threads: int
    llm_n_gpu_layers: int
    embedding_device: Device
    reasons: list[str] = field(default_factory=list)

    def summary(self) -> str:
        gpu = "all" if self.llm_n_gpu_layers < 0 else str(self.llm_n_gpu_layers)
        return (
            f"acceleration: threads={self.llm_n_threads} "
            f"llm_gpu_layers={gpu} embed_device={self.embedding_device}"
        )


# --------------------------------------------------------------------------- #
# Parse / classify helpers
# --------------------------------------------------------------------------- #
_NVIDIA_RE = re.compile(r"nvidia|geforce|rtx|gtx|quadro|tesla|titan", re.I)
_INTEL_RE = re.compile(r"intel|iris|\buhd\b|\bhd graphics\b|\barc\b", re.I)
_AMD_RE = re.compile(r"amd|radeon|\brx\b|vega|firepro", re.I)
_INTEGRATED_RE = re.compile(r"iris|\buhd\b|\bhd graphics\b|vega|radeon graphics", re.I)


def _classify_vendor(name: str) -> Vendor:
    if _NVIDIA_RE.search(name):
        return "nvidia"
    if _INTEL_RE.search(name):
        return "intel"
    if _AMD_RE.search(name):
        return "amd"
    return "unknown"


def _classify_kind(name: str, vendor: Vendor) -> GpuKind:
    if _INTEGRATED_RE.search(name):
        return "integrated"
    if vendor in ("nvidia", "amd"):
        return "discrete"
    if vendor == "intel":
        # Arc is discrete; other Intel graphics are integrated.
        return "discrete" if re.search(r"\barc\b", name, re.I) else "integrated"
    return "unknown"


def _parse_nvidia_smi(csv_text: str) -> list[GpuInfo]:
    """Parse nvidia-smi CSV lines: 'NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB'."""
    gpus: list[GpuInfo] = []
    for line in csv_text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        name = parts[0]
        vram_mb: int | None = None
        if len(parts) > 1:
            m = re.search(r"(\d+)", parts[1])
            if m:
                vram_mb = int(m.group(1))
        gpus.append(GpuInfo(vendor="nvidia", name=name, kind="discrete", vram_mb=vram_mb))
    return gpus


def _parse_video_controllers(rows: list[dict[str, object]]) -> list[GpuInfo]:
    """Parse Win32_VideoController rows (Name, AdapterRAM). AdapterRAM is in bytes
    and wraps for >4 GiB cards, so it's only a hint."""
    gpus: list[GpuInfo] = []
    for row in rows:
        name = str(row.get("Name") or "").strip()
        if not name:
            continue
        vendor = _classify_vendor(name)
        vram_mb: int | None = None
        raw = row.get("AdapterRAM")
        if isinstance(raw, (int, float)) and raw > 0:
            vram_mb = int(raw) // (1024 * 1024)
        gpus.append(GpuInfo(vendor=vendor, name=name, kind=_classify_kind(name, vendor), vram_mb=vram_mb))
    return gpus


def _dedup_gpus(gpus: list[GpuInfo]) -> list[GpuInfo]:
    """Merge GPUs from multiple probes (dedup on name), preferring entries with
    known VRAM. nvidia-smi and Win32_VideoController both list NVIDIA cards."""
    by_name: dict[str, GpuInfo] = {}
    for g in gpus:
        key = g.name.strip().lower()
        existing = by_name.get(key)
        if existing is None or (existing.vram_mb is None and g.vram_mb is not None):
            by_name[key] = g
    return list(by_name.values())


# --------------------------------------------------------------------------- #
# Probes (each isolated: one failure degrades a single field)
# --------------------------------------------------------------------------- #
def _run(cmd: list[str]) -> str:
    """Run a probe command, returning stdout ('' on failure). UTF-8 with
    errors='replace' so non-ASCII device names can't crash the probe."""
    try:
        out = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_PROBE_TIMEOUT_S,
            check=False,
        )
        return out.stdout or ""
    except (OSError, subprocess.SubprocessError) as exc:
        log.debug("probe failed: %s (%s)", cmd[0], exc)
        return ""


def _detect_cpu() -> CpuInfo:
    physical = 0
    logical = os.cpu_count() or 0
    ram_gb = 0.0

    try:
        import psutil  # type: ignore  # optional dep

        physical = psutil.cpu_count(logical=False) or 0
        logical = psutil.cpu_count(logical=True) or logical
        ram_gb = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception as exc:  # fall back to stdlib/ctypes
        log.debug("psutil cpu/ram probe unavailable: %s", exc)

    if physical <= 0:
        physical = logical  # over-counts on SMT but beats reporting zero
    if ram_gb <= 0.0:
        ram_gb = _ram_via_ctypes_gb()

    return CpuInfo(
        brand=_cpu_brand(),
        physical_cores=physical,
        logical_cores=logical,
        ram_total_gb=ram_gb,
    )


def _cpu_brand() -> str:
    if platform.system() == "Windows":
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            try:
                val, _ = winreg.QueryValueEx(key, "ProcessorNameString")
                if val:
                    return str(val).strip()
            finally:
                winreg.CloseKey(key)
        except OSError as exc:
            log.debug("winreg cpu brand probe failed: %s", exc)
    return platform.processor() or platform.machine() or "unknown"


def _ram_via_ctypes_gb() -> float:
    if platform.system() != "Windows":
        return 0.0
    try:
        import ctypes

        class _MemStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        stat = _MemStatus()
        stat.dwLength = ctypes.sizeof(_MemStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
            return float(round(stat.ullTotalPhys / (1024**3), 1))
    except Exception as exc:  # RAM stays unknown
        log.debug("ctypes ram probe failed: %s", exc)
    return 0.0


def _detect_nvidia_gpus() -> list[GpuInfo]:
    # Preferred: pynvml (structured, no parsing).
    try:
        import pynvml  # type: ignore  # optional dep

        pynvml.nvmlInit()
        try:
            gpus: list[GpuInfo] = []
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(h)
                if isinstance(name, bytes):
                    name = name.decode(errors="replace")
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                gpus.append(
                    GpuInfo(
                        vendor="nvidia",
                        name=str(name),
                        kind="discrete",
                        vram_mb=int(mem.total) // (1024 * 1024),
                    )
                )
            return gpus
        finally:
            pynvml.nvmlShutdown()
    except Exception as exc:  # no driver / pynvml not installed
        log.debug("pynvml unavailable: %s", exc)

    # Fallback: nvidia-smi text.
    out = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    return _parse_nvidia_smi(out) if out.strip() else []


def _cim_query(class_name: str, properties: list[str]) -> list[dict[str, object]]:
    """Query a CIM class via PowerShell ConvertTo-Json, returning rows as dicts.
    Windows-only; returns [] elsewhere or on failure."""
    if platform.system() != "Windows":
        return []
    import json

    props = ",".join(properties)
    # Force UTF-8 stdout to match _run's decode regardless of console codepage.
    script = (
        "[Console]::OutputEncoding = [System.Text.Encoding]::UTF8; "
        f"Get-CimInstance {class_name} | "
        f"Select-Object {props} | ConvertTo-Json -Compress"
    )
    out = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", script]).strip()
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError as exc:
        log.debug("CIM json parse failed for %s: %s", class_name, exc)
        return []
    if isinstance(data, dict):
        data = [data]
    return [d for d in data if isinstance(d, dict)]


def _detect_all_gpus() -> list[GpuInfo]:
    gpus = _detect_nvidia_gpus()
    rows = _cim_query("Win32_VideoController", ["Name", "AdapterRAM"])
    gpus.extend(_parse_video_controllers(rows))
    return _dedup_gpus(gpus)


def _detect_npus() -> list[NpuInfo]:
    rows = _cim_query("Win32_PnPEntity", ["Name"])
    npus: list[NpuInfo] = []
    seen: set[str] = set()
    pattern = re.compile(r"\bNPU\b|Neural|AI Boost", re.I)
    for row in rows:
        name = str(row.get("Name") or "").strip()
        if name and pattern.search(name) and name.lower() not in seen:
            seen.add(name.lower())
            npus.append(NpuInfo(name=name))
    return npus


def _detect_caps() -> RuntimeCaps:
    llama_gpu = False
    try:
        import llama_cpp

        fn = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        llama_gpu = bool(fn()) if callable(fn) else False
    except Exception as exc:  # not installed
        log.debug("llama_cpp capability probe unavailable: %s", exc)

    torch_cuda = False
    torch_xpu = False
    try:
        import torch

        torch_cuda = bool(torch.cuda.is_available())
        xpu = getattr(torch, "xpu", None)
        torch_xpu = bool(xpu is not None and xpu.is_available())
    except Exception as exc:  # not installed
        log.debug("torch capability probe unavailable: %s", exc)

    return RuntimeCaps(llama_gpu_offload=llama_gpu, torch_cuda=torch_cuda, torch_xpu=torch_xpu)


# --------------------------------------------------------------------------- #
# Public entry points
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def detect_hardware() -> HardwareProfile:
    """Probe the host once per process and cache the result."""
    log.debug("detecting hardware ...")
    profile = HardwareProfile(
        cpu=_detect_cpu(),
        gpus=_detect_all_gpus(),
        npus=_detect_npus(),
        caps=_detect_caps(),
    )
    log.info(
        "hardware: %s, %d core(s), %.1f GB RAM, %d GPU(s), %d NPU(s)",
        profile.cpu.brand,
        profile.cpu.physical_cores,
        profile.cpu.ram_total_gb,
        len(profile.gpus),
        len(profile.npus),
    )
    return profile


def _usable_gpu(profile: HardwareProfile) -> GpuInfo | None:
    """First real NVIDIA/AMD/Intel adapter (not a basic display), or None."""
    for g in profile.gpus:
        if g.vendor in ("nvidia", "intel", "amd"):
            return g
    return None


def plan_acceleration(profile: HardwareProfile, settings: Settings) -> AccelerationPlan:
    """Turn a hardware profile + settings into backend knobs. Explicit settings
    are honored; only 'auto' values are inferred. GPU offload requires the
    runtime to confirm it can use one."""
    reasons: list[str] = []

    # --- threads ---
    if settings.llm_n_threads > 0:
        threads = settings.llm_n_threads
    else:
        threads = profile.cpu.physical_cores or 4
        reasons.append(f"threads={threads} from {profile.cpu.physical_cores} physical core(s)")

    # --- llm gpu offload ---
    gpu = _usable_gpu(profile)
    if settings.acceleration == "cpu":
        gpu_layers = 0
        reasons.append("LLM on CPU (acceleration=cpu)")
    elif gpu is None:
        gpu_layers = 0
        reasons.append("LLM on CPU (no GPU detected)")
    elif not profile.caps.llama_gpu_offload:
        gpu_layers = 0
        reasons.append(
            f"LLM on CPU: {gpu.name} present but installed llama-cpp-python is CPU-only "
            "- reinstall a CUDA/Vulkan/SYCL wheel to enable offload"
        )
    else:
        gpu_layers = settings.llm_n_gpu_layers
        depth = "all layers" if gpu_layers < 0 else f"{gpu_layers} layers"
        reasons.append(f"LLM offload to {gpu.name} ({depth})")

    # --- embedding device ---
    embedding_device = _resolve_embedding_device(settings.embedding_device, profile.caps, reasons)

    return AccelerationPlan(
        llm_n_threads=threads,
        llm_n_gpu_layers=gpu_layers,
        embedding_device=embedding_device,
        reasons=reasons,
    )


def format_report(profile: HardwareProfile, plan: AccelerationPlan) -> str:
    """Human-readable hardware + acceleration report."""
    lines: list[str] = []
    cpu = profile.cpu
    lines.append("CPU:")
    lines.append(f"  {cpu.brand}")
    lines.append(f"  cores: {cpu.physical_cores} physical / {cpu.logical_cores} logical")
    lines.append(f"  RAM:   {cpu.ram_total_gb:.1f} GB")

    lines.append("GPU(s):")
    if profile.gpus:
        for g in profile.gpus:
            vram = f"{g.vram_mb} MB" if g.vram_mb else "VRAM unknown"
            lines.append(f"  [{g.vendor}/{g.kind}] {g.name} ({vram})")
    else:
        lines.append("  (none detected)")

    lines.append("NPU(s):")
    if profile.npus:
        for n in profile.npus:
            lines.append(f"  {n.name}  (detected only - no acceleration path)")
    else:
        lines.append("  (none detected)")

    caps = profile.caps
    lines.append("Runtime capability (what the installed build can use):")
    lines.append(f"  llama.cpp GPU offload: {'yes' if caps.llama_gpu_offload else 'no'}")
    lines.append(f"  torch CUDA:            {'yes' if caps.torch_cuda else 'no'}")
    lines.append(f"  torch XPU (Intel):     {'yes' if caps.torch_xpu else 'no'}")

    lines.append("Acceleration plan:")
    lines.append(f"  {plan.summary()}")
    for reason in plan.reasons:
        lines.append(f"  - {reason}")
    return "\n".join(lines)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def recommended_install_commands(
    profile: HardwareProfile, acceleration: str, embedding_device: str
) -> list[str]:
    """pip commands to install what the chosen plan still needs. Uses the venv
    interpreter's absolute path; only lists missing capabilities. Lines starting
    with '#' are notes, not commands."""
    py = f'"{sys.executable}"'
    cmds: list[str] = []

    # Detection extras (more accurate cores/VRAM).
    missing_probe: list[str] = []
    if not _module_available("psutil"):
        missing_probe.append("psutil")
    if not _module_available("pynvml"):  # from nvidia-ml-py
        missing_probe.append("nvidia-ml-py")
    if missing_probe:
        cmds.append(f"{py} -m pip install {' '.join(missing_probe)}")

    gpu = _usable_gpu(profile)
    want_gpu = acceleration in ("auto", "gpu")

    # llama.cpp offload needs a GPU-enabled wheel.
    if want_gpu and gpu is not None and not profile.caps.llama_gpu_offload:
        if gpu.vendor == "nvidia":
            cmds.append(
                f"{py} -m pip install --upgrade --force-reinstall llama-cpp-python "
                "--prefer-binary "
                "--extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu124"
            )
        else:
            cmds.append(
                f'# {gpu.vendor.upper()} GPU: no prebuilt wheel - rebuild llama-cpp-python '
                'with Vulkan: set CMAKE_ARGS="-DGGML_VULKAN=on" then reinstall from source'
            )

    # Embedding device needs torch built for the target.
    if embedding_device == "cuda" and not profile.caps.torch_cuda:
        cmds.append(f"{py} -m pip install torch --index-url https://download.pytorch.org/whl/cu124")
    elif embedding_device == "xpu" and not profile.caps.torch_xpu:
        cmds.append(
            f"{py} -m pip install torch intel-extension-for-pytorch "
            "--extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/"
        )

    return cmds


def _resolve_embedding_device(
    requested: str, caps: RuntimeCaps, reasons: list[str]
) -> Device:
    req = requested.lower()
    if req in ("cpu", "cuda", "xpu"):
        return req  # type: ignore[return-value]
    # auto
    if caps.torch_cuda:
        reasons.append("embeddings on CUDA (torch reports CUDA available)")
        return "cuda"
    if caps.torch_xpu:
        reasons.append("embeddings on Intel XPU (torch reports XPU available)")
        return "xpu"
    reasons.append("embeddings on CPU")
    return "cpu"

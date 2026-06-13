"""Hardware planner logic tests. Synthetic profiles and probe strings only; no
real detection (subprocess / pynvml / winreg) is exercised."""

from __future__ import annotations

import pytest

from app.config import Settings
from app.hardware import (
    AccelerationPlan,
    CpuInfo,
    GpuInfo,
    HardwareProfile,
    NpuInfo,
    RuntimeCaps,
    _classify_kind,
    _classify_vendor,
    _dedup_gpus,
    _parse_nvidia_smi,
    _parse_video_controllers,
    format_report,
    plan_acceleration,
    recommended_install_commands,
)


def _profile(
    *,
    physical: int = 8,
    gpus: list[GpuInfo] | None = None,
    npus: list[NpuInfo] | None = None,
    caps: RuntimeCaps | None = None,
) -> HardwareProfile:
    return HardwareProfile(
        cpu=CpuInfo(brand="Test CPU", physical_cores=physical, logical_cores=physical * 2, ram_total_gb=16.0),
        gpus=gpus or [],
        npus=npus or [],
        caps=caps or RuntimeCaps(),
    )


def _settings(**overrides: object) -> Settings:
    return Settings(_env_file=None, **overrides)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Vendor / kind classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "name,vendor",
    [
        ("NVIDIA GeForce RTX 4060 Laptop GPU", "nvidia"),
        ("NVIDIA Quadro P2000", "nvidia"),
        ("Intel(R) Iris(R) Xe Graphics", "intel"),
        ("Intel(R) UHD Graphics 770", "intel"),
        ("Intel(R) Arc(TM) A770 Graphics", "intel"),
        ("AMD Radeon RX 6700 XT", "amd"),
        ("Microsoft Basic Display Adapter", "unknown"),
    ],
)
def test_classify_vendor(name: str, vendor: str) -> None:
    assert _classify_vendor(name) == vendor


@pytest.mark.parametrize(
    "name,vendor,kind",
    [
        ("NVIDIA GeForce RTX 4060", "nvidia", "discrete"),
        ("Intel(R) Iris(R) Xe Graphics", "intel", "integrated"),
        ("Intel(R) Arc(TM) A770 Graphics", "intel", "discrete"),
        ("AMD Radeon RX 6700 XT", "amd", "discrete"),
    ],
)
def test_classify_kind(name: str, vendor: str, kind: str) -> None:
    assert _classify_kind(name, vendor) == kind  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Raw probe parsing
# --------------------------------------------------------------------------- #
def test_parse_nvidia_smi() -> None:
    text = "NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB\nNVIDIA RTX A2000, 6144 MiB\n"
    gpus = _parse_nvidia_smi(text)
    assert [g.name for g in gpus] == ["NVIDIA GeForce RTX 4060 Laptop GPU", "NVIDIA RTX A2000"]
    assert gpus[0].vram_mb == 8188
    assert all(g.vendor == "nvidia" and g.kind == "discrete" for g in gpus)


def test_parse_nvidia_smi_empty() -> None:
    assert _parse_nvidia_smi("") == []


def test_parse_video_controllers() -> None:
    rows = [
        {"Name": "Intel(R) Iris(R) Xe Graphics", "AdapterRAM": 1073741824},
        {"Name": "NVIDIA GeForce RTX 4060 Laptop GPU", "AdapterRAM": 4293918720},
        {"Name": "", "AdapterRAM": 0},  # dropped
    ]
    gpus = _parse_video_controllers(rows)
    assert [g.vendor for g in gpus] == ["intel", "nvidia"]
    assert gpus[0].kind == "integrated"
    assert gpus[0].vram_mb == 1024


def test_dedup_prefers_vram_known() -> None:
    smi = GpuInfo(vendor="nvidia", name="NVIDIA GeForce RTX 4060", kind="discrete", vram_mb=8188)
    cim = GpuInfo(vendor="nvidia", name="NVIDIA GeForce RTX 4060", kind="discrete", vram_mb=None)
    merged = _dedup_gpus([cim, smi])
    assert len(merged) == 1
    assert merged[0].vram_mb == 8188


# --------------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------------- #
def test_threads_auto_uses_physical_cores() -> None:
    plan = plan_acceleration(_profile(physical=12), _settings(llm_n_threads=0))
    assert plan.llm_n_threads == 12


def test_threads_explicit_honored() -> None:
    plan = plan_acceleration(_profile(physical=12), _settings(llm_n_threads=4))
    assert plan.llm_n_threads == 4


def test_gpu_offload_when_capable() -> None:
    nv = GpuInfo(vendor="nvidia", name="RTX 4060", kind="discrete", vram_mb=8188)
    profile = _profile(gpus=[nv], caps=RuntimeCaps(llama_gpu_offload=True))
    plan = plan_acceleration(profile, _settings(acceleration="auto", llm_n_gpu_layers=-1))
    assert plan.llm_n_gpu_layers == -1


def test_gpu_present_but_runtime_cpu_only() -> None:
    nv = GpuInfo(vendor="nvidia", name="RTX 4060", kind="discrete", vram_mb=8188)
    profile = _profile(gpus=[nv], caps=RuntimeCaps(llama_gpu_offload=False))
    plan = plan_acceleration(profile, _settings(acceleration="auto"))
    assert plan.llm_n_gpu_layers == 0
    assert any("CPU-only" in r for r in plan.reasons)


def test_acceleration_cpu_forces_cpu() -> None:
    nv = GpuInfo(vendor="nvidia", name="RTX 4060", kind="discrete", vram_mb=8188)
    profile = _profile(gpus=[nv], caps=RuntimeCaps(llama_gpu_offload=True))
    plan = plan_acceleration(profile, _settings(acceleration="cpu"))
    assert plan.llm_n_gpu_layers == 0


def test_no_gpu_stays_cpu() -> None:
    plan = plan_acceleration(_profile(gpus=[]), _settings(acceleration="auto"))
    assert plan.llm_n_gpu_layers == 0


@pytest.mark.parametrize(
    "caps,expected",
    [
        (RuntimeCaps(torch_cuda=True), "cuda"),
        (RuntimeCaps(torch_xpu=True), "xpu"),
        (RuntimeCaps(), "cpu"),
    ],
)
def test_embedding_device_auto(caps: RuntimeCaps, expected: str) -> None:
    plan = plan_acceleration(_profile(caps=caps), _settings(embedding_device="auto"))
    assert plan.embedding_device == expected


def test_embedding_device_explicit_overrides_caps() -> None:
    plan = plan_acceleration(
        _profile(caps=RuntimeCaps(torch_cuda=True)), _settings(embedding_device="cpu")
    )
    assert plan.embedding_device == "cpu"


def _nvidia_profile(*, offload: bool = False, torch_cuda: bool = False) -> HardwareProfile:
    return _profile(
        gpus=[GpuInfo(vendor="nvidia", name="RTX 4070", kind="discrete", vram_mb=12282)],
        caps=RuntimeCaps(llama_gpu_offload=offload, torch_cuda=torch_cuda),
    )


def test_install_cmds_nvidia_needs_cuda_wheel() -> None:
    cmds = recommended_install_commands(_nvidia_profile(offload=False), "auto", "auto")
    llama = [c for c in cmds if "llama-cpp-python" in c]
    assert llama and "cu124" in llama[0]


def test_install_cmds_none_when_offload_already_supported() -> None:
    cmds = recommended_install_commands(_nvidia_profile(offload=True), "auto", "auto")
    assert not any("llama-cpp-python" in c for c in cmds)


def test_install_cmds_cpu_mode_skips_llama() -> None:
    cmds = recommended_install_commands(_nvidia_profile(offload=False), "cpu", "auto")
    assert not any("llama-cpp-python" in c for c in cmds)


def test_install_cmds_cuda_embedding_needs_torch() -> None:
    cmds = recommended_install_commands(_nvidia_profile(torch_cuda=False), "cpu", "cuda")
    assert any("torch" in c and "download.pytorch.org" in c for c in cmds)


def test_summary_and_report_render() -> None:
    nv = GpuInfo(vendor="nvidia", name="RTX 4060", kind="discrete", vram_mb=8188)
    profile = _profile(gpus=[nv], npus=[NpuInfo(name="Intel(R) AI Boost")])
    plan = plan_acceleration(profile, _settings(llm_n_threads=8))
    assert isinstance(plan, AccelerationPlan)
    assert "threads=8" in plan.summary()
    report = format_report(profile, plan)
    assert "RTX 4060" in report
    assert "AI Boost" in report

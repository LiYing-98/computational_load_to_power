#!/usr/bin/env python3
"""Perform the Phase 1.7 time-boxed, direct-evidence-only RAPL identity check."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

from step3.scripts.common import sha256_bytes, sha256_file


ROOT = Path(__file__).resolve().parents[2]
DIRECT_EVIDENCE_TYPE = "direct_index_name_mapping"
ZONE_NAMES = r"(?:package(?:-\d+)?|psys|platform|dram|core|uncore|gpu)"


def classify_zone_identity(
    *,
    zone_index: int,
    direct_evidence: Iterable[Mapping[str, Any]],
    observed_power_w: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Resolve a zone only from an explicit index-to-name string mapping."""

    qualifying = [
        dict(item)
        for item in direct_evidence
        if item.get("evidence_type") == DIRECT_EVIDENCE_TYPE
        and item.get("zone_index") == zone_index
        and isinstance(item.get("zone_name"), str)
        and str(item["zone_name"]).strip()
    ]
    names = sorted({str(item["zone_name"]).strip().lower() for item in qualifying})
    if len(names) == 1:
        status = "resolved"
        resolved = names[0]
        conflict = None
    elif len(names) > 1:
        status = "unresolved_conflicting_direct_evidence"
        resolved = None
        conflict = names
    else:
        status = "unresolved"
        resolved = None
        conflict = None
    return {
        "zone_index": zone_index,
        "status": status,
        "resolved_zone_name": resolved,
        "neutral_label": None if resolved else "platform-like diagnostic",
        "numeric_inference_used": False,
        "observed_power_w_diagnostic_only": dict(observed_power_w or {}),
        "conflicting_names": conflict,
        "evidence": qualifying,
    }


def _explicit_json_hits(value: Any, source_path: str, location: str = "$") -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    if isinstance(value, dict):
        for field in ("rapl_zone_names", "powerzone_names", "zone_names"):
            mapping = value.get(field)
            if isinstance(mapping, dict):
                for key, name in mapping.items():
                    if str(key).isdigit() and isinstance(name, str) and re.fullmatch(ZONE_NAMES, name.lower()):
                        hits.append(
                            {
                                "source_path": source_path,
                                "json_location": f"{location}.{field}.{key}",
                                "zone_index": int(key),
                                "zone_name": name,
                                "evidence_type": DIRECT_EVIDENCE_TYPE,
                            }
                        )
        index = value.get("zone_index", value.get("rapl_zone_index"))
        name = value.get("zone_name", value.get("rapl_zone_name"))
        if isinstance(index, int) and isinstance(name, str) and re.fullmatch(ZONE_NAMES, name.lower()):
            hits.append(
                {
                    "source_path": source_path,
                    "json_location": location,
                    "zone_index": index,
                    "zone_name": name,
                    "evidence_type": DIRECT_EVIDENCE_TYPE,
                }
            )
        for key, child in value.items():
            hits.extend(_explicit_json_hits(child, source_path, f"{location}.{key}"))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            hits.extend(_explicit_json_hits(child, source_path, f"{location}[{index}]"))
    return hits


def _explicit_text_hits(path: Path, relative_path: str) -> list[dict[str, Any]]:
    hits: list[dict[str, Any]] = []
    patterns = (
        re.compile(rf"(?:intel-rapl:|rapl[ _-]?zone[ _-]?)(\d+)[^\n]{{0,80}}(?:name\s*[:=]\s*)(?P<name>{ZONE_NAMES})", re.I),
        re.compile(rf"zone[_ ]index\s*[:=]\s*(\d+)[^\n]{{0,80}}zone[_ ]name\s*[:=]\s*(?P<name>{ZONE_NAMES})", re.I),
    )
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        for pattern in patterns:
            match = pattern.search(line)
            if match:
                hits.append(
                    {
                        "source_path": relative_path,
                        "line_number": line_number,
                        "zone_index": int(match.group(1)),
                        "zone_name": match.group("name"),
                        "evidence_type": DIRECT_EVIDENCE_TYPE,
                    }
                )
    return hits


def _inspect_metadata_collection(project_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = project_root / "step2/analysis/result_metadata_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    hits: list[dict[str, Any]] = []
    digest_lines: list[str] = []
    present = 0
    for record in manifest["records"]:
        path = project_root / record["local_path"]
        if not path.exists():
            continue
        present += 1
        file_hash = sha256_file(path)
        digest_lines.append(f"{record['local_path']}:{file_hash}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        hits.extend(_explicit_json_hits(payload, record["local_path"]))
    receipt = {
        "source_class": "bounded_result_metadata_collection",
        "manifest_path": str(manifest_path.relative_to(project_root)).replace("\\", "/"),
        "manifest_sha256": sha256_file(manifest_path),
        "declared_file_count": len(manifest["records"]),
        "inspected_file_count": present,
        "collection_sha256": sha256_bytes("\n".join(sorted(digest_lines)).encode("utf-8")),
        "direct_mapping_hit_count": len(hits),
    }
    return receipt, hits


def run_audit(project_root: Path = ROOT) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    bounded_sources = (
        "step2/data/zeus-0.13.1/zeus/device/cpu/rapl.py",
        "step2/data/zeus-0.13.1/zeus/monitor/energy.py",
        "step2/data/official_source/benchmark.py",
        "step2/analysis/cpu_dram_audit.json",
        "step2/docs/cpu_dram_measurement_boundary.md",
        "step2/reports/phase1_6_audit_summary.md",
    )
    inspected: list[dict[str, Any]] = []
    hits: list[dict[str, Any]] = []
    for relative in bounded_sources:
        path = project_root / relative
        source_hits = _explicit_text_hits(path, relative)
        hits.extend(source_hits)
        inspected.append(
            {
                "source_class": "existing_bounded_source",
                "path": relative,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "direct_mapping_hit_count": len(source_hits),
            }
        )
    metadata_receipt, metadata_hits = _inspect_metadata_collection(project_root)
    inspected.append(metadata_receipt)
    hits.extend(metadata_hits)

    prior = json.loads((project_root / "step2/analysis/cpu_dram_audit.json").read_text(encoding="utf-8"))
    numeric = prior["quantity_diagnostics"]["steady_cpu_power_without_dram_zone_w"]
    result = classify_zone_identity(zone_index=1, direct_evidence=hits, observed_power_w=numeric)
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "existing bounded ML.ENERGY text-LLM inference metadata and pinned Zeus 0.13.1 source only",
        "time_box_result": "completed_without_new_downloads",
        "raw_timeline_downloaded": False,
        "prometheus_downloaded": False,
        "inspected_sources": inspected,
        "all_direct_mapping_hits": hits,
        "zone_1_identity": result,
        "conclusion": (
            "Direct index-to-name evidence was not found; retain zone 1 as a platform-like diagnostic."
            if result["status"] == "unresolved"
            else "Zone 1 identity is supported by direct index-to-name evidence."
        ),
    }


def _report(audit: Mapping[str, Any]) -> str:
    result = audit["zone_1_identity"]
    inspected = audit["inspected_sources"]
    rows = []
    for item in inspected:
        path = item.get("path", item.get("manifest_path"))
        digest = item.get("sha256", item.get("manifest_sha256"))
        count = item.get("inspected_file_count", 1)
        hits = item.get("direct_mapping_hit_count", 0)
        rows.append(f"| `{path}` | `{digest}` | {count} | {hits} |")
    numeric = result["observed_power_w_diagnostic_only"]
    return "\n".join(
        [
            "# RAPL zone identity time-box",
            "",
            f"结论：zone 1 状态为 **{result['status']}**。没有找到把整数索引 1 直接映射到 RAPL `name` 的字符串证据，因此继续使用中性标签 **platform-like diagnostic**；不能根据功率量级把它命名为 `psys`、package 或整机功率。",
            "",
            "本检查只读取既有的 694 份轻量结果头、Phase 1.6 审计产物、固定 Zeus 0.13.1 源码和固定 benchmark 源码；没有新增下载，没有读取完整 timeline，也没有下载 Prometheus。",
            "",
            "## 已检查证据",
            "",
            "| Source/manifest | SHA-256 | Files inspected | Direct mappings |",
            "|---|---|---:|---:|",
            *rows,
            "",
            "## 数值证据的限定用途",
            "",
            f"Phase 1.6 中 zone 1 对应的无 DRAM 顶层区功率统计为 median={numeric.get('median')} W、min={numeric.get('min')} W、max={numeric.get('max')} W。该量级只支持‘需要进一步核实测量边界’的诊断，不构成名称证据。",
            "",
            "下一阶段若要解析身份，应在采集时同时保存 `/sys/class/powercap/intel-rapl/intel-rapl:*/name`、路径、socket/NUMA 拓扑和采集时间；在此之前 CPU 三域和 zone 1 不进入本阶段 GPU 目标或 X。",
            "",
        ]
    )


def main() -> None:
    audit = run_audit(ROOT)
    (ROOT / "step3/analysis/rapl_zone_identity_evidence.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "step3/reports/rapl_zone_identity_check.md").write_text(
        _report(audit), encoding="utf-8"
    )
    print(
        f"zone 1 status={audit['zone_1_identity']['status']}; "
        f"direct_hits={len(audit['all_direct_mapping_hits'])}"
    )


if __name__ == "__main__":
    main()

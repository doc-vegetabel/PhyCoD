import csv
from typing import List, Optional

from .model import StudentBeamModel


def _pick_key(fieldnames: List[str], candidates: List[str]) -> Optional[str]:
    normalized = {name.strip().lower(): name for name in fieldnames}
    for cand in candidates:
        key = normalized.get(cand.strip().lower())
        if key is not None:
            return key
    return None


def _as_float(value: str, field_name: str) -> float:
    if value is None or str(value).strip() == "":
        raise ValueError(f"Empty value found in column '{field_name}'.")
    return float(value)


def build_minimal_student_cantilever(
    span_m: float = 61.5,
    n_stations: int = 11,
) -> StudentBeamModel:
    """
    阶段1占位模型：
    在没有正式 blade_master.csv 时，先生成一个可运行的最小学生梁模型。
    """
    if n_stations < 2:
        raise ValueError("n_stations must be >= 2.")

    eta = [i / (n_stations - 1) for i in range(n_stations)]

    mass_per_length_kgpm = [275.0 for _ in eta]
    flapwise_ei_nm2 = [1.0e9 * (1.0 - 0.7 * e) for e in eta]
    edgewise_ei_nm2 = [6.0e8 * (1.0 - 0.5 * e) for e in eta]
    torsional_gj_nm2 = [2.0e8 * (1.0 - 0.4 * e) for e in eta]

    model = StudentBeamModel(
        model_name="student_minimal_cantilever",
        span_m=span_m,
        eta=eta,
        mass_per_length_kgpm=mass_per_length_kgpm,
        flapwise_ei_nm2=flapwise_ei_nm2,
        edgewise_ei_nm2=edgewise_ei_nm2,
        torsional_gj_nm2=torsional_gj_nm2,
        notes="Stage 1 skeleton placeholder model; not calibrated to teacher model yet.",
    )
    model.validate()
    return model


def load_student_model_from_blade_master(
    csv_path: str,
    model_name: str = "student_from_blade_master",
) -> StudentBeamModel:
    """
    从 blade_master.csv 构造学生模型。

    当前优先适配两类表头：
    1) 你之前的简化/自定义表头
    2) 现在根据 BeamDyn/OpenFAST 参考文件生成的正式结构表头

    当前阶段1的工程映射约定：
    - span: 优先使用 r_m / span_m / span
    - eta: 优先使用 station_eta / eta
    - mass_per_length: 使用 mass_per_length_kgpm
    - flapwise_ei_nm2:
        优先显式 flapwise_ei_nm2；
        若没有，则先使用 bending_stiffness_k55_nm2
    - edgewise_ei_nm2:
        优先显式 edgewise_ei_nm2；
        若没有，则使用 bending_stiffness_k44_nm2
    - torsional_gj_nm2:
        优先显式 torsional_gj_nm2；
        若没有，则使用 torsional_stiffness_k66_nm2

    注：
    k44 / k55 到 flapwise / edgewise 的映射，在阶段1先作为工程约定使用，
    后续若结合更严格的 BeamDyn 局部坐标定义，可再校正。
    """
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []

    if not rows:
        raise ValueError(f"No data rows found in CSV: {csv_path}")

    span_key = _pick_key(
        fieldnames,
        [
            "r_m",
            "span_m",
            "span",
            "r",
            "z",
            "radius_m",
            "blade_span_m",
        ],
    )
    eta_key = _pick_key(
        fieldnames,
        [
            "station_eta",
            "eta",
            "span_frac",
            "span_fraction",
            "r_over_r",
            "r_over_R",
        ],
    )
    mass_key = _pick_key(
        fieldnames,
        [
            "mass_per_length_kgpm",
            "mass_per_length",
            "rhoA",
            "m",
            "mass_density",
        ],
    )

    flap_key = _pick_key(
        fieldnames,
        [
            "flapwise_ei_nm2",
            "bending_stiffness_k55_nm2",
            "flap_ei",
            "ei_flap",
            "EI_flap",
            "EI11",
            "k_flap",
        ],
    )
    edge_key = _pick_key(
        fieldnames,
        [
            "edgewise_ei_nm2",
            "bending_stiffness_k44_nm2",
            "edge_ei",
            "ei_edge",
            "EI_edge",
            "EI22",
            "k_edge",
        ],
    )
    torsion_key = _pick_key(
        fieldnames,
        [
            "torsional_gj_nm2",
            "torsional_stiffness_k66_nm2",
            "gj",
            "GJ",
            "torsion_gj",
            "torsional_stiffness",
        ],
    )

    if span_key is None:
        raise ValueError(
            f"Could not find span column in {csv_path}. "
            f"Available columns: {fieldnames}"
        )

    if mass_key is None:
        raise ValueError(
            f"Could not find mass-per-length column in {csv_path}. "
            f"Available columns: {fieldnames}"
        )

    if flap_key is None:
        raise ValueError(
            f"Could not find flapwise EI column in {csv_path}. "
            f"Available columns: {fieldnames}"
        )

    parsed = []
    for row in rows:
        span_value = _as_float(row[span_key], span_key)

        eta_value = None
        if eta_key is not None and str(row.get(eta_key, "")).strip() != "":
            eta_value = _as_float(row[eta_key], eta_key)

        edge_value = None
        if edge_key is not None and str(row.get(edge_key, "")).strip() != "":
            edge_value = _as_float(row[edge_key], edge_key)

        torsion_value = None
        if torsion_key is not None and str(row.get(torsion_key, "")).strip() != "":
            torsion_value = _as_float(row[torsion_key], torsion_key)

        parsed.append(
            {
                "span_m": span_value,
                "eta": eta_value,
                "mass_per_length_kgpm": _as_float(row[mass_key], mass_key),
                "flapwise_ei_nm2": _as_float(row[flap_key], flap_key),
                "edgewise_ei_nm2": edge_value,
                "torsional_gj_nm2": torsion_value,
            }
        )

    parsed.sort(key=lambda item: item["span_m"])

    span_start = parsed[0]["span_m"]
    span_end = parsed[-1]["span_m"]
    span_m = span_end - span_start
    if span_m <= 0.0:
        raise ValueError(
            f"Invalid span range in {csv_path}: start={span_start}, end={span_end}"
        )

    if eta_key is not None and all(item["eta"] is not None for item in parsed):
        eta_raw = [item["eta"] for item in parsed]
        eta0 = eta_raw[0]
        eta1 = eta_raw[-1]
        denom = eta1 - eta0
        if denom <= 0.0:
            raise ValueError("Invalid eta range in CSV.")
        eta = [(e - eta0) / denom for e in eta_raw]
    else:
        eta = [(item["span_m"] - span_start) / span_m for item in parsed]

    edgewise_values = [item["edgewise_ei_nm2"] for item in parsed]
    torsional_values = [item["torsional_gj_nm2"] for item in parsed]

    if all(v is None for v in edgewise_values):
        edgewise_values = None
    else:
        edgewise_values = [0.0 if v is None else v for v in edgewise_values]

    if all(v is None for v in torsional_values):
        torsional_values = None
    else:
        torsional_values = [0.0 if v is None else v for v in torsional_values]

    model = StudentBeamModel(
        model_name=model_name,
        span_m=span_m,
        eta=eta,
        mass_per_length_kgpm=[item["mass_per_length_kgpm"] for item in parsed],
        flapwise_ei_nm2=[item["flapwise_ei_nm2"] for item in parsed],
        edgewise_ei_nm2=edgewise_values,
        torsional_gj_nm2=torsional_values,
        notes=f"Loaded from CSV: {csv_path}",
    )
    model.validate()
    return model
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Union

Number = Union[int, float]


def _to_fortran_path(p: str | Path) -> str:
    """转成更稳妥的路径字符串，统一用正斜杠。"""
    return str(Path(p).resolve()).replace("\\", "/")


def _replace_driver_settings(
    template_inp: Path,
    temp_inp: Path,
    use_time_series_load: bool,
    time_series_load_file: Optional[str | Path] = None,
    t_initial: Optional[Number] = None,
    t_final: Optional[Number] = None,
    dt: Optional[Number] = None,
) -> None:
    """
    从模板 driver inp 生成一个临时 inp，并替换：
      - t_initial
      - t_final
      - dt
      - UseTimeSeriesLoad
      - TimeSeriesLoadFile
    """
    lines = template_inp.read_text(encoding="utf-8", errors="ignore").splitlines()

    found_t0 = False
    found_tf = False
    found_dt = False
    found_use = False
    found_file = False

    new_lines = []
    for line in lines:
        stripped = line.strip()

        if "t_initial" in line:
            if t_initial is None:
                new_lines.append(line)
            else:
                new_lines.append(
                    f'{float(t_initial):<12g} t_initial            - Starting time of simulation'
                )
            found_t0 = True
            continue

        if "t_final" in line:
            if t_final is None:
                new_lines.append(line)
            else:
                new_lines.append(
                    f'{float(t_final):<12g} t_final              - Ending time of simulation'
                )
            found_tf = True
            continue

        if stripped and stripped.split(maxsplit=1)[-1].startswith("dt"):
            if dt is None:
                new_lines.append(line)
            else:
                new_lines.append(
                    f'{float(dt):<12g} dt                   - Time increment size'
                )
            found_dt = True
            continue

        if "UseTimeSeriesLoad" in line:
            val = "True" if use_time_series_load else "False"
            new_line = f'{val:<12} UseTimeSeriesLoad    - Use time-series multi-point load? (True/False)'
            new_lines.append(new_line)
            found_use = True
            continue

        if "TimeSeriesLoadFile" in line:
            if time_series_load_file is None:
                ts_file_str = "./dummy_not_used.dat"
            else:
                ts_file_str = _to_fortran_path(time_series_load_file)

            new_line = f'"{ts_file_str}"   TimeSeriesLoadFile - Time-series multi-point load file'
            new_lines.append(new_line)
            found_file = True
            continue

        new_lines.append(line)

    if not found_t0:
        raise RuntimeError(f"在 {template_inp} 中没有找到 t_initial 这一行。")
    if not found_tf:
        raise RuntimeError(f"在 {template_inp} 中没有找到 t_final 这一行。")
    if not found_dt:
        raise RuntimeError(f"在 {template_inp} 中没有找到 dt 这一行。")
    if not found_use:
        raise RuntimeError(f"在 {template_inp} 中没有找到 UseTimeSeriesLoad 这一行。")
    if not found_file:
        raise RuntimeError(f"在 {template_inp} 中没有找到 TimeSeriesLoadFile 这一行。")

    temp_inp.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def run_beamdyn_case(
    exe_path: str | Path,
    template_inp: str | Path,
    output_dir: str | Path,
    case_name: str,
    use_time_series_load: bool,
    t_initial: Number,
    t_final: Number,
    dt: Number,
    time_series_load_file: Optional[str | Path] = None,
    keep_temp_inp: bool = False,
) -> Dict[str, Path | None]:
    """
    跑一个 BeamDyn case，并把结果保存到 output_dir。
    """
    exe_path = Path(exe_path).resolve()
    template_inp = Path(template_inp).resolve()
    output_dir = Path(output_dir).resolve()

    if not exe_path.exists():
        raise FileNotFoundError(f"找不到 exe: {exe_path}")
    if not template_inp.exists():
        raise FileNotFoundError(f"找不到 driver inp: {template_inp}")

    if float(dt) <= 0:
        raise ValueError("dt 必须大于 0。")
    if float(t_final) < float(t_initial):
        raise ValueError("t_final 必须大于或等于 t_initial。")

    if use_time_series_load:
        if time_series_load_file is None:
            raise ValueError("use_time_series_load=True 时必须提供 time_series_load_file。")
        if not Path(time_series_load_file).resolve().exists():
            raise FileNotFoundError(f"找不到时历载荷文件: {Path(time_series_load_file).resolve()}")

    output_dir.mkdir(parents=True, exist_ok=True)

    driver_dir = template_inp.parent
    temp_inp = driver_dir / f"{template_inp.stem}__{case_name}.inp"

    _replace_driver_settings(
        template_inp=template_inp,
        temp_inp=temp_inp,
        use_time_series_load=use_time_series_load,
        time_series_load_file=time_series_load_file,
        t_initial=t_initial,
        t_final=t_final,
        dt=dt,
    )

    cmd = [str(exe_path), temp_inp.name]
    print(f"\n[RUN] {case_name}")
    print(" ".join(cmd))
    print(f"[CWD] {driver_dir}")

    result = subprocess.run(
        cmd,
        cwd=driver_dir,
        capture_output=True,
        text=True,
    )

    stdout_file = output_dir / f"{case_name}.stdout.txt"
    stderr_file = output_dir / f"{case_name}.stderr.txt"
    stdout_file.write_text(result.stdout or "", encoding="utf-8", errors="ignore")
    stderr_file.write_text(result.stderr or "", encoding="utf-8", errors="ignore")

    if result.returncode != 0:
        raise RuntimeError(
            f"BeamDyn 运行失败，case={case_name}\n"
            f"returncode={result.returncode}\n"
            f"stdout 已保存到: {stdout_file}\n"
            f"stderr 已保存到: {stderr_file}"
        )

    produced_out = driver_dir / f"{temp_inp.stem}.out"
    produced_ech = driver_dir / f"{temp_inp.stem}.BD.ech"

    if not produced_out.exists():
        raise FileNotFoundError(f"运行成功但没找到输出文件: {produced_out}")

    saved_out = output_dir / f"{case_name}.out"
    shutil.copy2(produced_out, saved_out)

    saved_ech = None
    if produced_ech.exists():
        saved_ech = output_dir / f"{case_name}.BD.ech"
        shutil.copy2(produced_ech, saved_ech)

    produced_out.unlink(missing_ok=True)
    produced_ech.unlink(missing_ok=True)
    if not keep_temp_inp:
        temp_inp.unlink(missing_ok=True)

    return {
        "case_name": Path(case_name),
        "out": saved_out,
        "ech": saved_ech,
        "stdout": stdout_file,
        "stderr": stderr_file,
    }


def run_beamdyn_suite(
    exe_path: str | Path,
    template_inp: str | Path,
    output_dir: str | Path,
    cases: List[Dict],
) -> List[Dict[str, Path | None]]:
    """
    连续跑多个 case。
    每个 case 需要包含：
      - case_name
      - use_time_series_load
      - t_initial
      - t_final
      - dt
      - time_series_load_file (可选，ts_off 可不填)
    """
    results = []
    for case in cases:
        res = run_beamdyn_case(
            exe_path=exe_path,
            template_inp=template_inp,
            output_dir=output_dir,
            case_name=case["case_name"],
            use_time_series_load=case["use_time_series_load"],
            t_initial=case["t_initial"],
            t_final=case["t_final"],
            dt=case["dt"],
            time_series_load_file=case.get("time_series_load_file", None),
            keep_temp_inp=case.get("keep_temp_inp", False),
        )
        results.append(res)
    return results


if __name__ == "__main__":
    exe_path = r"D:\openfast\openfast-main\openfast-main\build\modules\beamdyn\Release\beamdyn_driver.exe"

    THIS_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = THIS_DIR.parent

    template_inp = PROJECT_ROOT / "data/raw/reference_cases/beamdyn/nrel5mw/bd_driver_dynamic_nrel_5mw.inp"
    output_dir   = PROJECT_ROOT / "results/teacher/newdyn_test"
    ts_dat       = PROJECT_ROOT / "data/load/train_complex_case.dat"

    cases = [
        {
            "case_name": "ts_off_10s",
            "use_time_series_load": False,
            "t_initial": 0.0,
            "t_final": 10.0,
            "dt": 0.01,
        },
        {
            "case_name": "ts_on_10s",
            "use_time_series_load": True,
            "t_initial": 0.0,
            "t_final": 10.0,
            "dt": 0.01,
            "time_series_load_file": ts_dat,
        }
    ]

    results = run_beamdyn_suite(
        exe_path=exe_path,
        template_inp=template_inp,
        output_dir=output_dir,
        cases=cases,
    )

    print("\n全部完成：")
    for r in results:
        print(f"- {r['case_name']}:")
        print(f"    out    = {r['out']}")
        print(f"    ech    = {r['ech']}")
        print(f"    stdout = {r['stdout']}")
        print(f"    stderr = {r['stderr']}")
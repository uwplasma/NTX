"""Execution and artifact writing for TOML-driven NTX runs."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel

from ._inputfiles_model import RunConfig, _load_surface, load_run_config
from ._inputfiles_output import infer_run_output_format, save_run_output
from ._inputfiles_reporting import (
    _algorithm_table,
    _case_table,
    _geometry_table,
    _output_table,
    _result_table,
    _surface_metadata_table,
    _surface_table,
    _timing_table,
)
from .config import enable_x64
from .solver import (
    prepare_monoenergetic_system,
    solve_prepared,
)


def run_from_input_file(
    path: str | Path,
    *,
    console: Console | None = None,
    output_path: str | Path | None = None,
    plot: bool = False,
    plot_path: str | Path | None = None,
) -> dict[str, Any]:
    """Execute an NTX run from a TOML input file and save a file-backed payload."""

    config = load_run_config(path)
    if output_path is not None:
        config = RunConfig(
            input_path=config.input_path,
            surface=config.surface,
            grid=config.grid,
            case=config.case,
            output=type(config.output)(
                path=Path(output_path).expanduser(),
                include_modes=config.output.include_modes,
            ),
            verbose=config.verbose,
        )
    console = Console() if console is None else console

    if config.verbose:
        console.print(
            Panel.fit(
                f"[bold]NTX[/bold]\nInput file: [cyan]{config.input_path}[/cyan]",
                title="Run",
            )
        )

    t0 = time.perf_counter()
    enable_x64(config.grid.x64)
    surface = _load_surface(config.surface)
    prepared = prepare_monoenergetic_system(surface, config.grid)
    geom = prepared.geometry
    t_prepared = time.perf_counter()
    if config.verbose:
        console.print(_surface_table(surface, config))
        console.print(_surface_metadata_table(surface))
        console.print(_geometry_table(geom))
        console.print(_case_table(config, surface))
        console.print(_algorithm_table(config, geom))
        console.print("[bold green]Solving monoenergetic system...[/bold green]")

    result = solve_prepared(prepared, config.case)
    t_solved = time.perf_counter()
    result_dict = result.as_dict()
    written_path = save_run_output(config.output.path, config, surface, result, geometry=geom)
    t_written = time.perf_counter()
    plot_pdf = None
    if plot:
        from .plotting import plot_run_output

        plot_outputs = plot_run_output(
            written_path,
            output_prefix=None if plot_path is None else Path(plot_path).with_suffix(""),
            formats=("pdf",),
        )
        plot_pdf = str(plot_outputs[0])
    t_done = time.perf_counter()
    timings = {
        "prepare": t_prepared - t0,
        "solve": t_solved - t_prepared,
        "write": t_written - t_solved,
        "total": t_done - t0,
    }
    if plot:
        timings["plot"] = t_done - t_written

    if config.verbose:
        console.print(_result_table(result_dict))
        console.print(_timing_table(timings))
        console.print(_output_table(written_path, config))
        console.print(f"[bold green]Wrote[/bold green] [cyan]{written_path}[/cyan]")
        if plot_pdf is not None:
            console.print(f"[bold green]Wrote plot[/bold green] [cyan]{plot_pdf}[/cyan]")

    payload = {
        "result": result_dict,
        "output_path": str(written_path),
        "output_format": infer_run_output_format(written_path),
        "timing_seconds": timings,
    }
    if plot_pdf is not None:
        payload["plot_pdf"] = plot_pdf
    if written_path.suffix.lower() == ".npz":
        payload["output_npz"] = str(written_path)
    return payload

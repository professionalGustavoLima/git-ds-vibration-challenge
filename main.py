#!/usr/bin/env python3
"""
Tractian DS Challenge — Condition Monitoring

Usage:
    uv run python main.py --part1
    uv run python main.py --part2
    uv run python main.py --part1 --part2
"""

import argparse
import logging
import time
from pathlib import Path
from typing import List

from src import metrics
from src.carpet_detector import CarpetDetector
from src.data_loader import load_part1, load_part2
from src.interface import CarpetRegion
from src.lubrication_model import LubricationModel

OUTPUTS_DIR = Path("outputs")

logger = logging.getLogger("main")


def _progress(step: str, done: int, total: int, start: float, every: int) -> None:
    """Log loop progress every ``every`` items and at the end, with elapsed time and a rough ETA."""
    if done % every and done != total:
        return
    elapsed = time.perf_counter() - start
    eta = elapsed / done * (total - done)
    logger.info("%s: %d/%d (%.0f%%) | elapsed %.1fs | eta %.1fs", step, done, total, 100 * done / total, elapsed, eta)


def _write_report(
    path: Path, title: str, metrics_result: dict, extra: str = ""
) -> None:
    with open(path, "w") as f:
        f.write(f"# {title}\n\n")
        f.write("## Metrics\n\n")
        for k, v in metrics_result.items():
            f.write(f"- **{k}**: {v}\n")
        if extra:
            f.write(f"\n{extra}")


def run_part1(output_dir: Path) -> None:
    part_start = time.perf_counter()
    logger.info("[Part 1] Starting carpet detector")
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    t0 = time.perf_counter()
    waves, labels = load_part1()
    logger.info("[Part 1] Loaded %d signals and %d labelled regions in %.1fs",
                len(waves), sum(len(r) for r in labels.values()), time.perf_counter() - t0)
    detector = CarpetDetector()

    predictions: dict[str, List[CarpetRegion]] = {}
    t0 = time.perf_counter()
    for i, (sample_id, wave) in enumerate(waves.items(), 1):
        predictions[sample_id] = detector.predict(wave)
        _progress("[Part 1] Predicting", i, len(waves), t0, every=25)
    logger.info("[Part 1] Detected %d regions in %d of %d signals",
                sum(len(r) for r in predictions.values()), sum(1 for r in predictions.values() if r), len(predictions))

    metrics_result = metrics.evaluate_part1(predictions, labels)
    logger.info("[Part 1] mean_iou = %s (in-sample: the detector was trained on these signals)",
                metrics_result.get("mean_iou"))

    report_path = output_dir / "report_part1.md"
    _write_report(report_path, "[Part 1] Carpet Detector Report", metrics_result)

    print(f"[Part 1] Metrics: {metrics_result}")
    print(f"[Part 1] Report  → {report_path}")

    t0 = time.perf_counter()
    for i, (sample_id, wave) in enumerate(waves.items(), 1):
        detector.plot_results(
            wave,
            predictions[sample_id],
            plots_dir,
            sample_id=sample_id,
            true_regions=labels[sample_id],
        )
        _progress("[Part 1] Plotting", i, len(waves), t0, every=25)
    logger.info("[Part 1] Plots saved to %s", plots_dir)
    logger.info("[Part 1] Done in %.1fs", time.perf_counter() - part_start)


def run_part2(output_dir: Path) -> None:
    part_start = time.perf_counter()
    logger.info("[Part 2] Starting lubrication model")
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_plots_dir = output_dir / "plots" / "assets"
    test_plots_dir = output_dir / "plots" / "test"
    assets_plots_dir.mkdir(parents=True, exist_ok=True)
    test_plots_dir.mkdir(parents=True, exist_ok=True)

    assets, labels, test_data = load_part2()
    logger.info("[Part 2] Found %d assets (%d labelled) and %d test assets",
                len(assets), sum(1 for a in assets if a in labels), len(test_data))

    predictions: dict[str, bool] = {}
    t0 = time.perf_counter()
    for i, (asset_id, data) in enumerate(assets.items(), 1):
        model = LubricationModel()
        model.fit(data.fit)
        predictions[asset_id] = model.predict(data.predict)

        asset_dir = assets_plots_dir / asset_id
        asset_dir.mkdir(exist_ok=True)
        model.plot_results(data.predict, asset_dir)
        logger.debug("[Part 2] Asset %s -> %s", asset_id, "starved" if predictions[asset_id] else "healthy")
        _progress("[Part 2] Assets", i, len(assets), t0, every=20)
    logger.info("[Part 2] Assets: %d predicted starved, %d healthy",
                sum(predictions.values()), len(predictions) - sum(predictions.values()))

    metrics_result = metrics.evaluate_part2(predictions, labels)

    test_predictions: dict[str, bool] = {}
    t0 = time.perf_counter()
    for i, (asset_id, data) in enumerate(test_data.items(), 1):
        model = LubricationModel()
        model.fit(data.fit)
        test_predictions[asset_id] = model.predict(data.predict)

        asset_dir = test_plots_dir / asset_id
        asset_dir.mkdir(exist_ok=True)
        model.plot_results(data.predict, asset_dir)
        logger.info("[Part 2] Test %d/%d: %s -> %s", i, len(test_data), asset_id,
                    "STARVED LUBRICATION" if test_predictions[asset_id] else "healthy")
    logger.info("[Part 2] Test: %d predicted starved, %d healthy (%.1fs)", sum(test_predictions.values()),
                len(test_predictions) - sum(test_predictions.values()), time.perf_counter() - t0)

    test_summary = metrics.summarize_part2_predictions(test_predictions)
    summary_lines = ["## Test Data Summary", ""] + [f"- **{k}**: {v}" for k, v in test_summary.items()] + [
        "",
        "No labels are available for `test_data/`: `n_starved` and `n_healthy` are predicted counts.",
        "",
    ]

    header = ["## Test Data Predictions", "", "| Asset ID | Prediction |", "|---|---|"]
    body = [
        f"| {aid} | {'Starved Lubrication' if pred else 'Healthy'} |"
        for aid, pred in test_predictions.items()
    ]
    test_table = "\n".join(header + body) + "\n"

    report_path = output_dir / "report_part2.md"
    _write_report(
        report_path,
        "[Part 2] Starved Lubrication Model Report",
        metrics_result,
        extra="\n".join(summary_lines) + "\n" + test_table,
    )

    print(f"[Part 2] Metrics: {metrics_result}")
    print(f"[Part 2] Test summary (predicted): {test_summary}")
    print(f"[Part 2] Report  → {report_path}")
    logger.info("[Part 2] Done in %.1fs", time.perf_counter() - part_start)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tractian DS Challenge — Condition Monitoring"
    )
    parser.add_argument(
        "--part1", action="store_true", help="Run Part 1: Carpet Detector"
    )
    parser.add_argument(
        "--part2", action="store_true", help="Run Part 2: Lubrication Model"
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Also log one line per Part 2 asset"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s %(message)s", datefmt="%H:%M:%S")
    if args.verbose:
        logger.setLevel(logging.DEBUG)  # only this script; keeps third-party debug logs (matplotlib) quiet

    if not args.part1 and not args.part2:
        parser.print_help()
        return

    OUTPUTS_DIR.mkdir(exist_ok=True)
    run_start = time.perf_counter()

    if args.part1:
        run_part1(OUTPUTS_DIR / "part1")

    if args.part2:
        run_part2(OUTPUTS_DIR / "part2")

    logger.info("Finished in %.1fs", time.perf_counter() - run_start)


if __name__ == "__main__":
    main()

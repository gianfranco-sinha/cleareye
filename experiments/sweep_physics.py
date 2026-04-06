"""
Physics-informed hyperparameter sweep for ClearEye.

Jointly optimises ML architecture and physics parameters using Optuna.
Composite objective combines regime classification accuracy, residual
calibration MAE, PDE mass conservation error, and regime-Peclet agreement.

Usage:
    python experiments/sweep_physics.py --n-trials 50 --sweep-epochs 30
    python experiments/sweep_physics.py --n-trials 3 --sweep-epochs 5  # smoke test
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.models import AnomalyAutoencoder, RegimeClassifierNet, ResidualCorrectionNet
from app.regime import TurbidityRegime
from app.sensor_physics import adc_to_voltage, sen0189_voltage_to_ntu
from simulator.conditions import Pulse
from simulator.geometry import PipeGeometry, load_geometry
from simulator.solver import AdvectionDiffusionSolver, SimulationParams
from training.data_sources import SyntheticDataSource
from training.utils import detect_device

logging.basicConfig(
    level=logging.INFO,
    format="%(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("sweep_physics")


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def _relabel_regime(
    adc: int,
    tds: float,
    adc_sol_thresh: int,
    adc_susp_thresh: int,
) -> int:
    """Assign regime label (0=solution, 1=colloid, 2=suspension) using trial thresholds."""
    if adc >= adc_sol_thresh:
        return 0
    if adc <= adc_susp_thresh:
        return 2
    return 1


def prepare_regime_data(
    n_samples: int,
    seed: int,
    adc_sol_thresh: int,
    adc_susp_thresh: int,
    val_frac: float = 0.2,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build regime classifier tensors, relabeled with trial thresholds."""
    df = SyntheticDataSource(n_samples=n_samples, seed=seed).load()

    labels = np.array([
        _relabel_regime(int(row.turbidity_adc), row.tds, adc_sol_thresh, adc_susp_thresh)
        for _, row in df.iterrows()
    ])

    features = df[["turbidity_adc", "tds", "water_temperature"]].values.astype(np.float32)

    # Normalise features to [0, 1] for stable training
    f_min = features.min(axis=0, keepdims=True)
    f_max = features.max(axis=0, keepdims=True)
    features = (features - f_min) / np.maximum(f_max - f_min, 1e-8)

    n_val = max(1, int(len(features) * val_frac))
    X_train = torch.tensor(features[:-n_val], device=device)
    y_train = torch.tensor(labels[:-n_val], dtype=torch.long, device=device)
    X_val = torch.tensor(features[-n_val:], device=device)
    y_val = torch.tensor(labels[-n_val:], dtype=torch.long, device=device)
    return X_train, y_train, X_val, y_val


def prepare_residual_data(
    n_samples: int,
    seed: int,
    temp_coefficient: float,
    val_frac: float = 0.2,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build residual correction tensors.

    Features: [voltage, water_temperature, tds, d_adc_dt (0), hour_sin (0), hour_cos (1)]
    Target: synthetic reference NTU - datasheet NTU (with trial temp_coefficient).
    """
    df = SyntheticDataSource(n_samples=n_samples, seed=seed).load()

    voltages = np.array([adc_to_voltage(int(a)) for a in df["turbidity_adc"]])
    datasheet_ntu = np.array([sen0189_voltage_to_ntu(v) for v in voltages])

    # Temperature compensation with trial coefficient
    temps = df["water_temperature"].values
    correction = 1.0 - temp_coefficient * (temps - 25.0)
    compensated_ntu = datasheet_ntu * correction

    # Synthetic "reference" NTU — add small structured noise as proxy for
    # real ground-truth that the residual net would learn to correct.
    rng = np.random.default_rng(seed + 1)
    reference_ntu = compensated_ntu + rng.normal(0, 5.0, size=len(compensated_ntu))
    reference_ntu = np.maximum(reference_ntu, 0.0)

    residuals = (reference_ntu - compensated_ntu).astype(np.float32)

    # Feature matrix
    tds = df["tds"].values.astype(np.float32)
    features = np.column_stack([
        voltages,
        temps,
        tds,
        np.zeros(len(df)),   # d_adc_dt placeholder
        np.zeros(len(df)),   # hour_sin placeholder
        np.ones(len(df)),    # hour_cos placeholder
    ]).astype(np.float32)

    # Normalise
    f_min = features.min(axis=0, keepdims=True)
    f_max = features.max(axis=0, keepdims=True)
    features = (features - f_min) / np.maximum(f_max - f_min, 1e-8)

    n_val = max(1, int(len(features) * val_frac))
    X_train = torch.tensor(features[:-n_val], device=device)
    y_train = torch.tensor(residuals[:-n_val], device=device)
    X_val = torch.tensor(features[-n_val:], device=device)
    y_val = torch.tensor(residuals[-n_val:], device=device)
    return X_train, y_train, X_val, y_val


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_regime_classifier(
    trial: optuna.Trial,
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    hidden_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
) -> float:
    """Train RegimeClassifierNet; return validation accuracy. Reports to Optuna for pruning."""
    model = RegimeClassifierNet(input_dim=3, hidden_dim=hidden_dim).to(X_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
            val_loss = criterion(logits, y_val).item()
            preds = logits.argmax(dim=-1)
            accuracy = (preds == y_val).float().mean().item()

        trial.report(val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return accuracy


def train_residual_net(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    hidden_dim: int,
    lr: float,
    batch_size: int,
    epochs: int,
) -> float:
    """Train ResidualCorrectionNet; return validation MAE."""
    model = ResidualCorrectionNet(input_dim=6, hidden_dim=hidden_dim).to(X_train.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(X_val)
        mae = (preds - y_val).abs().mean().item()
    return mae


# ---------------------------------------------------------------------------
# Physics consistency evaluation
# ---------------------------------------------------------------------------

def evaluate_physics_consistency(
    nx: int,
    solution_pe: int,
    suspension_pe: int,
    adc_sol_thresh: int,
    adc_susp_thresh: int,
    n_eval: int,
    seed: int,
) -> tuple[float, float]:
    """Run PDE forward solves and check physics consistency.

    Returns:
        (mean_mass_conservation_error, regime_pe_agreement_fraction)
    """
    geometry = load_geometry()
    geo = replace(geometry, suspension_pe=float(suspension_pe), solution_pe=float(solution_pe))
    solver = AdvectionDiffusionSolver(geo)
    rng = np.random.default_rng(seed)

    mass_errors: list[float] = []
    pe_matches = 0
    pe_total = 0
    duration = 30.0

    for _ in range(n_eval):
        d_mol = float(np.exp(rng.uniform(np.log(1e-12), np.log(1e-5))))
        velocity = float(np.exp(rng.uniform(np.log(0.001), np.log(0.5))))
        c0 = float(np.exp(rng.uniform(np.log(1.0), np.log(5000.0))))
        temperature = float(rng.uniform(1.0, 35.0))

        params = SimulationParams(
            d_molecular=d_mol,
            velocity=velocity,
            c0=c0,
            temperature=temperature,
            nx=nx,
            dt=None,
            duration=duration,
            boundary_type="pulse",
            geometry=geo,
        )
        bc = Pulse(c0=c0, t0=duration * 0.15, sigma=duration * 0.05)

        try:
            result = solver.solve(params, bc)
        except Exception:
            continue

        if np.any(np.isnan(result.downstream)):
            continue

        mass_errors.append(abs(result.mass_conservation_error))

        # Regime from Peclet number
        pe = result.peclet_number
        if pe > suspension_pe:
            pe_regime = 2  # suspension
        elif pe < solution_pe:
            pe_regime = 0  # solution
        else:
            pe_regime = 1  # colloid

        # Regime from ADC — map c0 to approximate ADC (inverse of transfer function)
        # c0 is in NTU-like concentration units; map through sensor range heuristic
        approx_adc = int(np.clip(1023 - (c0 / 4000.0) * 1023, 0, 1023))
        adc_regime = _relabel_regime(approx_adc, 300.0, adc_sol_thresh, adc_susp_thresh)

        if pe_regime == adc_regime:
            pe_matches += 1
        pe_total += 1

    mean_mass_error = float(np.mean(mass_errors)) if mass_errors else 1.0
    pe_agreement = pe_matches / pe_total if pe_total > 0 else 0.0
    return mean_mass_error, pe_agreement


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def make_objective(
    sweep_epochs: int,
    n_synthetic: int,
    n_physics_eval: int,
    seed: int,
    device: torch.device | str,
    w_cls: float,
    w_mae: float,
    w_mass: float,
    w_pe: float,
):
    """Return a closure that Optuna calls for each trial."""

    def objective(trial: optuna.Trial) -> float:
        # --- Physics parameters ---
        solution_pe = trial.suggest_int("solution_pe", 1, 100, log=True)
        suspension_pe = trial.suggest_int(
            "suspension_pe", max(solution_pe * 2, 100), 10000, log=True,
        )
        nx = trial.suggest_categorical("nx", [30, 60, 120])
        w_upstream = trial.suggest_float("w_upstream", 0.5, 2.0)
        w_downstream = trial.suggest_float("w_downstream", 0.5, 2.0)
        temp_coefficient = trial.suggest_float("temp_coefficient", 0.005, 0.02)
        adc_solution_threshold = trial.suggest_int("adc_solution_threshold", 700, 900)
        adc_suspension_threshold = trial.suggest_int(
            "adc_suspension_threshold", 300, min(adc_solution_threshold - 50, 500),
        )

        # --- ML parameters ---
        hidden_dim = trial.suggest_categorical("hidden_dim", [16, 32, 64, 128])
        encoding_dim = trial.suggest_categorical("encoding_dim", [2, 4, 8])
        lr = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])

        # --- Phase 1: Regime classifier ---
        X_tr, y_tr, X_v, y_v = prepare_regime_data(
            n_synthetic, seed, adc_solution_threshold, adc_suspension_threshold,
            device=device,
        )
        regime_accuracy = train_regime_classifier(
            trial, X_tr, y_tr, X_v, y_v,
            hidden_dim, lr, batch_size, sweep_epochs,
        )

        # --- Phase 2: Residual correction ---
        Xr_tr, yr_tr, Xr_v, yr_v = prepare_residual_data(
            n_synthetic, seed, temp_coefficient, device=device,
        )
        residual_mae = train_residual_net(
            Xr_tr, yr_tr, Xr_v, yr_v,
            hidden_dim, lr, batch_size, sweep_epochs,
        )

        # --- Phase 3: Physics consistency ---
        mass_error, pe_agreement = evaluate_physics_consistency(
            nx, solution_pe, suspension_pe,
            adc_solution_threshold, adc_suspension_threshold,
            n_physics_eval, seed,
        )

        # Store individual metrics as user attrs
        trial.set_user_attr("regime_accuracy", regime_accuracy)
        trial.set_user_attr("residual_mae", residual_mae)
        trial.set_user_attr("mass_conservation_error", mass_error)
        trial.set_user_attr("pe_agreement", pe_agreement)

        # Composite score (lower is better)
        score = (
            (1.0 - regime_accuracy) * w_cls
            + residual_mae * w_mae
            + mass_error * w_mass
            + (1.0 - pe_agreement) * w_pe
        )
        return score

    return objective


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Physics-informed hyperparameter sweep")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--sweep-epochs", type=int, default=30)
    parser.add_argument("--final-epochs", type=int, default=100)
    parser.add_argument("--n-physics-eval", type=int, default=50)
    parser.add_argument("--n-synthetic", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="trained_models/sweep_physics")
    parser.add_argument("--w-cls", type=float, default=1.0)
    parser.add_argument("--w-mae", type=float, default=1.0)
    parser.add_argument("--w-mass", type=float, default=0.5)
    parser.add_argument("--w-pe", type=float, default=0.5)
    args = parser.parse_args()

    device = detect_device()

    print("=" * 70)
    print("PHYSICS-INFORMED HYPERPARAMETER SWEEP")
    print(f"  Trials:          {args.n_trials}")
    print(f"  Sweep epochs:    {args.sweep_epochs}")
    print(f"  Final epochs:    {args.final_epochs}")
    print(f"  Physics evals:   {args.n_physics_eval}")
    print(f"  Synthetic rows:  {args.n_synthetic}")
    print(f"  Seed:            {args.seed}")
    print(f"  Device:          {device}")
    print(f"  Weights:         cls={args.w_cls} mae={args.w_mae} "
          f"mass={args.w_mass} pe={args.w_pe}")
    print("=" * 70)

    # Create Optuna study
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, n_warmup_steps=10,
    )
    study = optuna.create_study(
        study_name="cleareye_physics_sweep",
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )

    objective = make_objective(
        sweep_epochs=args.sweep_epochs,
        n_synthetic=args.n_synthetic,
        n_physics_eval=args.n_physics_eval,
        seed=args.seed,
        device=device,
        w_cls=args.w_cls,
        w_mae=args.w_mae,
        w_mass=args.w_mass,
        w_pe=args.w_pe,
    )

    start = time.time()
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    elapsed = time.time() - start

    # --- Results table ---
    print(f"\n{'=' * 70}")
    print("SWEEP RESULTS (ranked by composite score)")
    print(f"{'=' * 70}")

    header = (
        f"{'#':<4} {'score':>7} {'acc':>6} {'MAE':>7} {'mass':>7} {'Pe%':>5} "
        f"{'hid':>4} {'lr':>8} {'nx':>4} {'sol_pe':>7} {'sus_pe':>7} {'Status':>8}"
    )
    print(header)
    print("-" * len(header))

    trials_sorted = sorted(study.trials, key=lambda t: t.value if t.value is not None else 1e10)

    for t in trials_sorted:
        if t.state == optuna.trial.TrialState.COMPLETE:
            acc = t.user_attrs.get("regime_accuracy", 0)
            mae = t.user_attrs.get("residual_mae", 0)
            mass = t.user_attrs.get("mass_conservation_error", 0)
            pe_ag = t.user_attrs.get("pe_agreement", 0)
            print(
                f"{t.number:<4} {t.value:>7.4f} {acc:>6.3f} {mae:>7.4f} {mass:>7.5f} "
                f"{pe_ag:>5.2f} {t.params['hidden_dim']:>4} {t.params['learning_rate']:>8.5f} "
                f"{t.params['nx']:>4} {t.params['solution_pe']:>7} {t.params['suspension_pe']:>7} "
                f"{'ok':>8}"
            )
        elif t.state == optuna.trial.TrialState.PRUNED:
            print(f"{t.number:<4} {'---':>7} {'---':>6} {'---':>7} {'---':>7} {'---':>5} "
                  f"{'':>4} {'':>8} {'':>4} {'':>7} {'':>7} {'pruned':>8}")
        else:
            print(f"{t.number:<4} {'---':>7} {'---':>6} {'---':>7} {'---':>7} {'---':>5} "
                  f"{'':>4} {'':>8} {'':>4} {'':>7} {'':>7} {'failed':>8}")

    # --- Best trial ---
    best = study.best_trial
    print(f"\n{'=' * 70}")
    print(f"BEST TRIAL: #{best.number}  score={best.value:.4f}")
    print(f"  Physics: solution_pe={best.params['solution_pe']}, "
          f"suspension_pe={best.params['suspension_pe']}, "
          f"nx={best.params['nx']}")
    print(f"  Physics: temp_coeff={best.params['temp_coefficient']:.4f}, "
          f"w_up={best.params['w_upstream']:.2f}, "
          f"w_down={best.params['w_downstream']:.2f}")
    print(f"  Regime:  adc_sol={best.params['adc_solution_threshold']}, "
          f"adc_susp={best.params['adc_suspension_threshold']}")
    print(f"  ML:      hidden={best.params['hidden_dim']}, "
          f"lr={best.params['learning_rate']:.5f}, "
          f"batch={best.params['batch_size']}")
    print(f"  Metrics: acc={best.user_attrs['regime_accuracy']:.3f}, "
          f"MAE={best.user_attrs['residual_mae']:.4f}, "
          f"mass_err={best.user_attrs['mass_conservation_error']:.5f}, "
          f"Pe_agree={best.user_attrs['pe_agreement']:.2f}")

    # --- Retrain best for more epochs ---
    print(f"\nRetraining best config for {args.final_epochs} epochs...")

    bp = best.params
    X_tr, y_tr, X_v, y_v = prepare_regime_data(
        args.n_synthetic, args.seed,
        bp["adc_solution_threshold"], bp["adc_suspension_threshold"],
        device=device,
    )

    # Use a dummy trial for retraining (no pruning)
    final_study = optuna.create_study(direction="minimize")
    final_trial = optuna.trial.create_trial(
        state=optuna.trial.TrialState.RUNNING,
        values=None,
        params={},
        distributions={},
    )

    # Train regime classifier
    final_cls_model = RegimeClassifierNet(input_dim=3, hidden_dim=bp["hidden_dim"]).to(device)
    cls_optimizer = torch.optim.Adam(final_cls_model.parameters(), lr=bp["learning_rate"])
    cls_criterion = nn.CrossEntropyLoss()
    cls_loader = DataLoader(
        TensorDataset(X_tr, y_tr), batch_size=bp["batch_size"], shuffle=True,
    )

    for epoch in range(args.final_epochs):
        final_cls_model.train()
        for xb, yb in cls_loader:
            cls_optimizer.zero_grad()
            loss = cls_criterion(final_cls_model(xb), yb)
            loss.backward()
            cls_optimizer.step()

    final_cls_model.eval()
    with torch.no_grad():
        final_acc = (final_cls_model(X_v).argmax(dim=-1) == y_v).float().mean().item()

    # Train residual net
    Xr_tr, yr_tr, Xr_v, yr_v = prepare_residual_data(
        args.n_synthetic, args.seed, bp["temp_coefficient"], device=device,
    )
    final_res_model = ResidualCorrectionNet(input_dim=6, hidden_dim=bp["hidden_dim"]).to(device)
    res_optimizer = torch.optim.Adam(final_res_model.parameters(), lr=bp["learning_rate"])
    res_criterion = nn.MSELoss()
    res_loader = DataLoader(
        TensorDataset(Xr_tr, yr_tr), batch_size=bp["batch_size"], shuffle=True,
    )

    for epoch in range(args.final_epochs):
        final_res_model.train()
        for xb, yb in res_loader:
            res_optimizer.zero_grad()
            loss = res_criterion(final_res_model(xb), yb)
            loss.backward()
            res_optimizer.step()

    final_res_model.eval()
    with torch.no_grad():
        final_mae = (final_res_model(Xr_v) - yr_v).abs().mean().item()

    print(f"  Final regime accuracy: {final_acc:.3f}")
    print(f"  Final residual MAE:    {final_mae:.4f}")

    # --- Save results ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save best models
    torch.save(final_cls_model.state_dict(), output_dir / "regime_classifier.pt")
    torch.save(final_res_model.state_dict(), output_dir / "residual_correction.pt")

    # Build trial records
    trial_records = []
    for t in study.trials:
        record = {
            "number": t.number,
            "state": t.state.name,
            "params": t.params,
        }
        if t.value is not None:
            record["score"] = t.value
            record.update(t.user_attrs)
        trial_records.append(record)

    output = {
        "experiment": "physics_informed_sweep",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": {
            "n_trials": args.n_trials,
            "sweep_epochs": args.sweep_epochs,
            "final_epochs": args.final_epochs,
            "n_physics_eval": args.n_physics_eval,
            "n_synthetic": args.n_synthetic,
            "seed": args.seed,
            "weights": {
                "w_cls": args.w_cls,
                "w_mae": args.w_mae,
                "w_mass": args.w_mass,
                "w_pe": args.w_pe,
            },
        },
        "best_trial": {
            "number": best.number,
            "score": best.value,
            "params": best.params,
            "metrics": best.user_attrs,
        },
        "final_retrain": {
            "regime_accuracy": final_acc,
            "residual_mae": final_mae,
        },
        "trials": trial_records,
        "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
        "elapsed_seconds": round(elapsed, 1),
    }

    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    print(f"Models saved to {output_dir}/")
    print(f"Total time: {elapsed / 60:.1f} minutes")
    print(f"Completed: {output['n_completed']}, Pruned: {output['n_pruned']}")


if __name__ == "__main__":
    main()

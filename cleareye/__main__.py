"""ClearEye CLI — train, verify, version commands."""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cleareye",
        description="ClearEye — ML platform for water quality prediction",
    )
    subparsers = parser.add_subparsers(dest="command")

    # version
    subparsers.add_parser("version", help="Print version and exit")

    # verify — smoke-test the inference pipeline
    subparsers.add_parser("verify", help="Verify inference pipeline with a test reading")

    # train — placeholder for training pipeline
    train_parser = subparsers.add_parser("train", help="Run training pipeline")
    train_parser.add_argument(
        "--config", type=str, default="model_config.yaml",
        help="Path to model config YAML",
    )

    args = parser.parse_args()

    if args.command == "version":
        print("cleareye 0.1.0")

    elif args.command == "verify":
        _verify()

    elif args.command == "train":
        print(f"Training pipeline not yet implemented. Config: {args.config}")
        sys.exit(1)

    else:
        parser.print_help()


def _verify() -> None:
    """Smoke-test the inference engine with a synthetic reading."""
    from datetime import datetime

    from app.inference import InferenceEngine, Reading

    reading = Reading(
        timestamp=datetime.now(),
        rig_id="verify-rig",
        turbidity_adc=500,
        tds=300.0,
        water_temperature=18.0,
    )
    engine = InferenceEngine()
    result = engine.predict(reading)
    print("Verification passed:")
    print(f"  Regime:    {result.regime.value}")
    print(f"  NTU:       {result.turbidity_ntu}")
    print(f"  Voltage:   {result.turbidity_voltage}V")
    print(f"  Confidence: {result.confidence}")


if __name__ == "__main__":
    main()

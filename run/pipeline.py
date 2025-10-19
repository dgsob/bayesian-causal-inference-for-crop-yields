# To run:
# python src/pipeline.py
# python src/pipeline.py --skip-prepare-dataset
# python src/pipeline.py --skip-tune-priors
# python src/pipeline.py --skip-prepare-dataset --skip-tune-priors

import argparse
import subprocess
import sys


def run_script(script_name: str):
    """
    Executes a Python script as a subprocess.

    Args:
        script_name: The name of the script to run.

    Raises:
        SystemExit: If the script returns a non-zero exit code or is not found.
    """
    print(f"--- Running {script_name} ---")
    try:
        subprocess.run([sys.executable, script_name], check=True, text=True)
        print(f"--- Finished {script_name} successfully ---\n")
    except FileNotFoundError:
        print(f"Error: {script_name} not found.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"--- Error running {script_name} ---", file=sys.stderr)
        print(e.stdout, file=sys.stdout)
        print(e.stderr, file=sys.stderr)
        sys.exit(1)


def main():
    """Runs the full data analysis pipeline."""
    parser = argparse.ArgumentParser(description="Run the full analysis pipeline.")
    parser.add_argument(
        "--skip-prepare-dataset",
        action="store_true",
        help="Skip the prepare_dataset.py step.",
    )
    parser.add_argument(
        "--skip-tune-priors", action="store_true", help="Skip the tune_priors.py step."
    )
    args = parser.parse_args()

    if not args.skip_prepare_dataset:
        run_script("data_processing/prepare_dataset.py")
    if not args.skip_tune_priors:
        run_script("modeling/tune_priors.py")

    run_script("modeling/estimate_ate.py")
    run_script("results/summary.py")

    print("Pipeline completed successfully!")


if __name__ == "__main__":
    main()
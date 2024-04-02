from training_pipeline import MFPipeline
import argparse
from src.constants import DEFAULT_YAML_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--yaml_path",
        help="Path to the yaml config.",
        type=str,
        default=DEFAULT_YAML_PATH,
    )

    args = parser.parse_args()
    pipeline = MFPipeline(yaml_path=args.yaml_path)
    breakpoint()
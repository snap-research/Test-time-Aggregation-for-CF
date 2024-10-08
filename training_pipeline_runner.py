import argparse

from src.constants import DEFAULT_GENERAL_YAML_PATH, DEFAULT_MODEL_YAML_PATH
from src.utils import set_seed
from training_pipeline import MFPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--general_yaml_path",
        help="Path to the yaml config.",
        type=str,
        default=DEFAULT_GENERAL_YAML_PATH,
    )

    parser.add_argument(
        "--model_yaml_path",
        help="Path to the yaml config.",
        type=str,
        default=DEFAULT_MODEL_YAML_PATH,
    )
    set_seed()

    args = parser.parse_args()
    pipeline = MFPipeline(
        general_yaml_path=args.general_yaml_path, model_yaml_path=args.model_yaml_path
    )
    if pipeline.config["model_type"] == "TAGCF":
        pipeline.test_time_aggregation()
    else:
        pipeline.train()

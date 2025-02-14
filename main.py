from pathlib import Path
import yaml
import argparse

from diffuser_tools.main import main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Diffusers inference pipeline.", 
        description="Generate images with generative image AI models."
    )

    parser.add_argument(
        "--config_dir", 
        help="Folder under `diffuser_tools/` containing the `.yaml` configuration files.", 
        default="./",
    )
    parser.add_argument(
        "--config", help="Name of the train configuration file.", default="main.yaml"
    )
    args = parser.parse_args()

    config_file = args.config
    config_dir = args.config_dir

    run_path = Path(__file__).parent

    config_file_path = run_path / config_dir / config_file
    with open(config_file_path, 'r') as stream:
        configs = yaml.safe_load(stream)

    main(**configs)

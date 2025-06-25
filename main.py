# Copyright 2025 Natsunoyuki AI Laboratory

# diffuser_tools is free software: you can redistribute it and/or modify it 
# under the terms of the GNU General Public License as published by the Free 
# Software Foundation, either version 3 of the License, or (at your option) any 
# later version.

# diffuser_tools is distributed in the hope that it will be useful, but 
# WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or 
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more 
# details.

# You should have received a copy of the GNU General Public License along with 
# diffuser_tools. If not, see <https://www.gnu.org/licenses/>.

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

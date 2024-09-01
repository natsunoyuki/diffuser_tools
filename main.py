import pathlib
import yaml

from diffuser_tools.main import main

if __name__ == "__main__":
    run_path = pathlib.Path(__file__).parent

    config_file_path = run_path / "main.yml"
    with open(config_file_path, 'r') as stream:
        configs = yaml.safe_load(stream)

    main(**configs)

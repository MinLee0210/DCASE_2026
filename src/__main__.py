from argparse import ArgumentParser
from typing import Any, Dict


def parse_args() -> Dict[str, Any]:
    parser = ArgumentParser(description="DCASE 2026 Challenge")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    args = parse_args()
    print("Parsed arguments:", args)

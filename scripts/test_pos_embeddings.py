import os
import yaml
import subprocess
import shutil

BASE_CONFIG_PATH = "config/config_castella_test.yml"
TEMP_CONFIG_PATH = "config/temp_test_config.yml"
EMBEDDINGS = ["sine", "learned", "rope", "conv"]


def main():
    # Load base config
    with open(BASE_CONFIG_PATH, "r") as f:
        base_config = yaml.safe_load(f)

    # Ensure n_epoch is small for testing
    base_config["n_epoch"] = 1
    base_config["device"] = "cpu"  # Keep it CPU for a quick dry-run test

    for emb in EMBEDDINGS:
        print("=" * 60)
        print(f"Testing Positional Embedding: {emb}")
        print("=" * 60)

        # Modify config
        test_config = base_config.copy()
        test_config["position_embedding"] = emb
        test_config["results_dir"] = f"results_test_{emb}"

        # Save temp config
        with open(TEMP_CONFIG_PATH, "w") as f:
            yaml.dump(test_config, f)

        import sys

        # Run training
        cmd = [sys.executable, "-m", "src", "train", "--config", TEMP_CONFIG_PATH]

        try:
            # We don't want to capture output so the user can see it in real time,
            # but we do want to check for errors.
            subprocess.run(cmd, check=True)
            print(f"✅ Successfully completed dry-run for {emb}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed dry-run for {emb}")

    # Cleanup
    if os.path.exists(TEMP_CONFIG_PATH):
        os.remove(TEMP_CONFIG_PATH)


if __name__ == "__main__":
    main()

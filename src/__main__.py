import argparse

from src.core.config import BaseOptions


def main():
    parser = argparse.ArgumentParser(description="DCASE 2026 Challenge")
    parser.add_argument(
        "pipeline",
        type=str,
        choices=["train", "evaluate", "create_submission"],
        help="The pipeline to run.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        required=True,
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume",
        "-r",
        type=str,
        help="Specify model path for fine-tuning (train pipeline). If None, train the model from scratch.",
    )
    parser.add_argument(
        "--model_path",
        "-m",
        type=str,
        help="Model checkpoint path (required for evaluate and create_submission pipelines).",
    )
    parser.add_argument(
        "--split",
        "-s",
        type=str,
        default="val",
        choices=["val", "test"],
        help="Split name for evaluate: val or test",
    )

    args = parser.parse_args()

    option_manager = BaseOptions(args.config)
    option_manager.parse()
    opt = option_manager.option

    if args.pipeline == "train":
        from src.pipelines.train import main as train_main

        train_main(opt, args.resume)

    elif args.pipeline == "evaluate":
        if not args.model_path:
            parser.error("--model_path is required for evaluate pipeline")
        from src.pipelines.evaluate import start_inference

        opt.model_path = args.model_path
        opt.eval_split_name = args.split
        start_inference(opt)

    elif args.pipeline == "create_submission":
        if not args.model_path:
            parser.error("--model_path is required for create_submission pipeline")
        from src.pipelines.create_submission import start_inference

        opt.model_path = args.model_path
        opt.eval_split_name = "private"
        start_inference(opt)


if __name__ == "__main__":
    main()

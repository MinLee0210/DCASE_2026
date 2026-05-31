import json


def truncate_jsonl(input_file, output_file, max_lines=5):
    with open(input_file, "r") as fin, open(output_file, "w") as fout:
        for i, line in enumerate(fin):
            if i >= max_lines:
                break
            fout.write(line)


truncate_jsonl(
    "dataset/preprocessed/castella_train_release.jsonl",
    "dataset/preprocessed/castella_dummy_train.jsonl",
)
truncate_jsonl(
    "dataset/preprocessed/castella_val_release.jsonl",
    "dataset/preprocessed/castella_dummy_val.jsonl",
)

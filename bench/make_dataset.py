# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Baseten

import json

from datasets import load_dataset


NUM_REQUESTS = 500

def main():

    dataset_stream = load_dataset(
        "glaiveai/code_edits_sample",
        split="train",
        streaming=True
    ).take(NUM_REQUESTS)

    requests = [f"""
Apply the following edit to the following code:\n
<edit>\n{item['edit']}\n</edit>\n\n
<code>\n{item['code']}\n</code>\n\n
Result:\n<edited_code>
"""
        for item in dataset_stream]

    with open("dataset.jsonl", "w") as f:
        for i, request in enumerate(requests):
            f.write(json.dumps({"task_id": i, "prompt": request, "output_tokens": 2048}) + "\n")

if __name__ == "__main__":
    main()

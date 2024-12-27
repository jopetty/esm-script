import os
import logging
import pandas as pd
import pathlib
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
import json
import fire
import pyrootutils

PROJECT_ROOT = path = pyrootutils.find_root(
    search_from=__file__, indicator=".project-root"
)


def main(
    output_dir: pathlib.Path | str = PROJECT_ROOT / "output",
    fullseq: bool = False,
    batch_size: int = 32,
    epochs: int = 10,
):
    torch.cuda.empty_cache()

    output_dir = pathlib.Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    log_file = output_dir / "esm_training.log"

    logging.basicConfig(
        filename=log_file,
        filemode="w",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger()

    logger.info(f"Logging to file: {log_file}")
    logger.info(f"Training on full sequences? {fullseq}")
    logger.info(f"Output directory: {output_dir}")

    logger.info("Loading datasets...")
    hla_allele = pd.read_csv(PROJECT_ROOT / "hlaseq_jackson.csv")
    data_all = pd.read_csv(PROJECT_ROOT / "data_jackson.csv")
    logger.info("Datasets loaded successfully.")

    indices_to_subset = [
        7,
        9,
        13,
        24,
        31,
        45,
        59,
        62,
        63,
        65,
        66,
        67,
        69,
        70,
        71,
        73,
        74,
        76,
        77,
        80,
        81,
        84,
        95,
        97,
        99,
        110,
        114,
        116,
        118,
        138,
        143,
        147,
        150,
        152,
        156,
        158,
        159,
        163,
        167,
        171,
    ]

    # Prepare sequences based on the selected type
    logger.info("Preparing sequences...")
    if fullseq:
        logger.info("Processing fullseq...")
        hla_allele_dict = hla_allele.set_index("allele")["seq"].to_dict()
        max_len = 11 + 182
        logger.info("Fullseq processing complete.")
    else:
        logger.info("Processing pseudoseq...")

        hla_allele["hla_pocket"] = hla_allele["seq"].apply(
            lambda x: "".join([x[i] for i in indices_to_subset if i < len(x)])
        )
        hla_allele_dict = hla_allele.set_index("allele")["hla_pocket"].to_dict()
        max_len = 11 + 40
        logger.info("Pseudoseq processing complete.")

    logger.info("Concatenating sequences...")
    data_all["concat_seq"] = data_all.apply(
        lambda row: row["seq"] + hla_allele_dict.get(row["allele"], ""), axis=1
    )
    logger.info("Concatenation complete.")

    data_subset = data_all[["concat_seq", "label", "set"]]
    data_train = data_subset[data_subset.set == "train"]
    data_valid = data_subset[data_subset.set == "valid"]
    data_test = data_subset[data_subset.set == "test"]

    logger.info("Data split into train, validation, and test sets.")

    train_dataset = Dataset.from_pandas(data_train)
    val_dataset = Dataset.from_pandas(data_valid)
    test_dataset = Dataset.from_pandas(data_test)

    logger.info("Datasets converted to Hugging Face Dataset format.")

    print(train_dataset)
    print(val_dataset)
    print(test_dataset)

    # Load tokenizer and model
    model_name = "facebook/esm2_t33_650M_UR50D"
    logger.info(f"Loading model and tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Preprocess the dataset
    logger.info("Preprocessing datasets...")

    def preprocess_function(input_data):
        tokenized = tokenizer(
            input_data["concat_seq"],
            padding="max_length",
            max_length=max_len,
            return_tensors="np",
        )
        tokenized["labels"] = input_data["label"]
        return tokenized

    train_dataset = train_dataset.map(preprocess_function, batched=True)
    val_dataset = val_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    train_dataset = train_dataset.remove_columns(["concat_seq"])
    train_dataset.set_format("torch")

    val_dataset = val_dataset.remove_columns(["concat_seq"])
    val_dataset.set_format("torch")

    test_dataset = test_dataset.remove_columns(["concat_seq"])
    test_dataset.set_format("torch")

    logger.info("Preprocessing complete.")

    # Define training arguments
    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=os.path.join(output_dir, "training_results"),
        evaluation_strategy="epoch",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_dir=os.path.join(output_dir, "logs"),
        learning_rate=2e-5,
        torch_compile=True,
    )
    logger.info("Training arguments set up.")

    model = EsmForSequenceClassification.from_pretrained(
        model_name, num_labels=len(data_train["label"].unique())
    )

    # Define the Trainer
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Train the model
    logger.info("Starting training...")
    trainer.train()
    logger.info("Training complete.")

    # # Save the final model and tokenizer
    # model_dir = os.path.join(output_dir, "finetuned_esm_model")
    # logger.info(f"Saving model and tokenizer to {model_dir}...")
    # model.save_pretrained(model_dir)
    # tokenizer.save_pretrained(model_dir)
    # logger.info("Model and tokenizer saved.")

    # Evaluate the model
    logger.info("Evaluating the model...")
    evaluation_results = trainer.evaluate()

    # Log the evaluation results
    logger.info(f"Evaluation results: {evaluation_results}")

    # Save the evaluation results to a JSON file
    eval_results_file = os.path.join(output_dir, "evaluation_results.json")
    with open(eval_results_file, "w") as f:
        json.dump(evaluation_results, f, indent=4)

    logger.info(f"Evaluation results saved to {eval_results_file}")


if __name__ == "__main__":
    fire.Fire(main)

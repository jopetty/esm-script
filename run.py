import os
import logging
import pandas as pd
import argparse
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    EsmForSequenceClassification,
    TrainingArguments,
    Trainer,
)
import torch
import json

torch.cuda.empty_cache()

# Setup logging
logging.basicConfig(
    filename="esm_training.log",  # Temporary log file (will move to output_dir)
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger()

logger.info("Script started.")

# Argument parser to choose pseudoseq or fullseq and output directory
parser = argparse.ArgumentParser(
    description="Train ESM model with pseudoseq or fullseq for HLA."
)
parser.add_argument(
    "--sequence_type",
    choices=["pseudoseq", "fullseq"],
    required=True,
    help="Choose between 'pseudoseq' or 'fullseq' for input sequence type.",
)
parser.add_argument(
    "--output_dir",
    required=True,
    help="Directory to save outputs (model, tokenizer, logs, etc.).",
)
args = parser.parse_args()

# Ensure output directory exists
output_dir = args.output_dir
os.makedirs(output_dir, exist_ok=True)

# Update logging to use the output directory
log_file = os.path.join(output_dir, "esm_training.log")
for handler in logger.handlers[:]:
    logger.removeHandler(handler)
logging.basicConfig(
    filename=log_file,
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger.info(f"Logging to file: {log_file}")

logger.info(f"Selected sequence type: {args.sequence_type}")
logger.info(f"Output directory: {output_dir}")

# Load data
logger.info("Loading datasets...")
hla_allele = pd.read_csv(
    "~/hlathena_speduptrainorignn/hlathenav2/hlathena/data/ABCG_prot.parsed.clean.updated.ALL.FEATS.txt",
    sep=" ",
)
data_all = pd.read_csv("/home/jeb7273/hlathenav2/JB_addedFiles/test_split.txt", sep=" ")

logger.info("Datasets loaded successfully.")

max_bat = 32
num_train_ep = 10

# Pocket positions
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
if args.sequence_type == "pseudoseq":
    logger.info("Processing pseudoseq...")
    hla_allele["hla_pocket"] = hla_allele["seq"].apply(
        lambda x: "".join([x[i] for i in indices_to_subset if i < len(x)])
    )
    hla_allele_dict = hla_allele.set_index("allele")["hla_pocket"].to_dict()
    max_len = 11 + 40
    logger.info("Pseudoseq processing complete.")
elif args.sequence_type == "fullseq":
    logger.info("Processing fullseq...")
    hla_allele_dict = hla_allele.set_index("allele")["seq"].to_dict()
    max_len = 11 + 182
    logger.info("Fullseq processing complete.")

# Concatenate sequences
logger.info("Concatenating sequences...")
data_all["concat_seq"] = data_all.apply(
    lambda row: row["seq"] + hla_allele_dict.get(row["allele"], ""), axis=1
)
logger.info("Concatenation complete.")

# Split data into train, validation, and test sets
data_train = data_all[data_all.set == "train"].copy()
data_valid = data_all[data_all.set == "valid"].copy()
data_test = data_all[data_all.set == "test"].copy()

data_train_subset = data_train[["concat_seq", "label"]].copy()
data_valid_subset = data_valid[["concat_seq", "label"]].copy()
data_test_subset = data_test[["concat_seq", "label"]].copy()

logger.info("Data split into train, validation, and test sets.")

# Convert the pandas DataFrame to a Hugging Face Dataset
train_dataset = Dataset.from_pandas(data_train_subset)
val_dataset = Dataset.from_pandas(data_valid_subset)
test_dataset = Dataset.from_pandas(data_test_subset)

logger.info("Datasets converted to Hugging Face Dataset format.")

# Load tokenizer and model
model_name = "facebook/esm2_t33_650M_UR50D"
logger.info(f"Loading model and tokenizer: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = EsmForSequenceClassification.from_pretrained(
    model_name, num_labels=len(data_train["label"].unique())
)

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
    num_train_epochs=num_train_ep,
    per_device_train_batch_size=max_bat,
    per_device_eval_batch_size=max_bat,
    logging_dir=os.path.join(output_dir, "logs"),
    learning_rate=2e-5,
)
logger.info("Training arguments set up.")

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

# Save the final model and tokenizer
model_dir = os.path.join(output_dir, "finetuned_esm_model")
logger.info(f"Saving model and tokenizer to {model_dir}...")
model.save_pretrained(model_dir)
tokenizer.save_pretrained(model_dir)
logger.info("Model and tokenizer saved.")

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

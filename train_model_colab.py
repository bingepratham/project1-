"""
Google Colab Training Script - 70% Dataset

This script is optimized for Google Colab with reduced dataset size.

SETUP IN COLAB:
1. Upload this file to Colab
2. Upload datasets/synthetic_training_data.csv to Colab
3. Run this script
4. Download the trained model
5. Copy to your laptop's models/ folder
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    BertTokenizer, 
    BertForMaskedLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
import zipfile
from pathlib import Path

# --- CONFIGURATION ---
# Modified for faster training
DATASET_PATH = "synthetic_training_data.csv"  # Upload to Colab
DATASET_SIZE_PERCENTAGE = 0.70  # Use only 70% of dataset
TRAIN_TEST_SPLIT = 0.1

BASE_MODEL_NAME = "bert-base-uncased"
OUTPUT_MODEL_PATH = "manuscript-bert-final"

# Reduced training parameters for speed
BATCH_SIZE = 32            # Larger batch (Colab has more GPU memory)
LEARNING_RATE = 5e-5
NUM_EPOCHS = 2             # Reduced from 3 to 2 for speed
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
LOGGING_STEPS = 50         # More frequent logging
SAVE_STEPS = 500
EVAL_STEPS = 250

MAX_SEQUENCE_LENGTH = 128
MLM_PROBABILITY = 0.15


def setup_colab_environment():
    """
    Setup Google Colab environment.
    """
    print("="*60)
    print("GOOGLE COLAB SETUP")
    print("="*60 + "\n")
    
    # Check if running in Colab
    try:
        import google.colab
        in_colab = True
        print("✓ Running in Google Colab")
    except:
        in_colab = False
        print("⚠️  Not running in Google Colab")
    
    # Check GPU
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"✓ GPU available: {gpu_name} ({gpu_memory:.1f}GB)")
        device = "cuda"
    else:
        print("⚠️  No GPU detected - training will be slow!")
        print("In Colab: Runtime → Change runtime type → GPU")
        device = "cpu"
    
    print(f"✓ Device: {device.upper()}\n")
    
    return device, in_colab


def load_reduced_dataset():
    """
    Load dataset and use only 70% for faster training.
    """
    print("="*60)
    print("LOADING DATASET (70% SUBSET)")
    print("="*60 + "\n")
    
    # Check if dataset exists
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print("\nYou need to:")
        print("1. Run build_dataset.py on your laptop")
        print("2. Upload synthetic_training_data.csv to Colab")
        print("3. Then run this script")
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")
    
    print(f"Loading dataset from: {DATASET_PATH}")
    dataset = load_dataset("csv", data_files=DATASET_PATH, split="train")
    original_size = len(dataset)
    print(f"  Full dataset: {original_size} samples")
    
    # Take only 70%
    target_size = int(original_size * DATASET_SIZE_PERCENTAGE)
    dataset = dataset.select(range(target_size))
    print(f"  Using 70%: {len(dataset)} samples")
    print(f"  Saved time: ~30% faster training!\n")
    
    # Split into train/validation
    print(f"Splitting: {int((1-TRAIN_TEST_SPLIT)*100)}% train, {int(TRAIN_TEST_SPLIT*100)}% val")
    dataset = dataset.train_test_split(test_size=TRAIN_TEST_SPLIT)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Validation: {len(eval_dataset)} samples\n")
    
    return train_dataset, eval_dataset


def tokenize_dataset(train_dataset, eval_dataset, tokenizer):
    """
    Tokenize datasets for BERT training.
    """
    print("Tokenizing datasets...")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["input"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQUENCE_LENGTH
        )
    
    train_dataset = train_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=train_dataset.column_names
    )
    eval_dataset = eval_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=eval_dataset.column_names
    )
    
    print("  ✓ Tokenization complete\n")
    
    return train_dataset, eval_dataset


def train_model(train_dataset, eval_dataset, tokenizer, device):
    """
    Train BERT model with reduced parameters for speed.
    """
    print("="*60)
    print("INITIALIZING MODEL")
    print("="*60 + "\n")
    
    print(f"Loading base model: {BASE_MODEL_NAME}")
    model = BertForMaskedLM.from_pretrained(BASE_MODEL_NAME)
    print(f"  ✓ Model loaded ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters)\n")
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=MLM_PROBABILITY
    )
    
    # Training arguments optimized for Colab
    training_args = TrainingArguments(
        output_dir=OUTPUT_MODEL_PATH,
        overwrite_output_dir=True,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=1,  # Keep only best checkpoint (save space)
        push_to_hub=False,
        report_to="none",
        no_cuda=(device == "cpu"),
        fp16=(device == "cuda"),  # Mixed precision for faster training on GPU
    )
    
    print("="*60)
    print("TRAINING CONFIGURATION")
    print("="*60)
    print(f"  Dataset size: 70% ({len(train_dataset)} samples)")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {device.upper()}")
    print(f"  Mixed precision (FP16): {device == 'cuda'}")
    print(f"  Estimated time: 15-25 minutes (GPU)")
    print("="*60 + "\n")
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )
    
    print("="*60)
    print("STARTING TRAINING")
    print("="*60)
    print("Progress will be logged every 50 steps.\n")
    
    # Train!
    train_result = trainer.train()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"  Final train loss: {train_result.training_loss:.4f}")
    print(f"  Total time: {train_result.metrics['train_runtime']:.1f}s")
    print(f"  Steps per second: {train_result.metrics['train_steps_per_second']:.2f}")
    print("="*60 + "\n")
    
    return model, trainer


def save_and_package_model(model, tokenizer):
    """
    Save model and create downloadable zip file.
    """
    print("="*60)
    print("SAVING MODEL")
    print("="*60 + "\n")
    
    print(f"Saving model to: {OUTPUT_MODEL_PATH}")
    model.save_pretrained(OUTPUT_MODEL_PATH)
    tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
    print("  ✓ Model saved")
    print("  ✓ Tokenizer saved\n")
    
    # Create zip file for easy download
    zip_filename = "manuscript-bert-final.zip"
    print(f"Creating zip file: {zip_filename}")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(OUTPUT_MODEL_PATH):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(OUTPUT_MODEL_PATH))
                zipf.write(file_path, arcname)
                print(f"  Added: {arcname}")
    
    zip_size = os.path.getsize(zip_filename) / 1024 / 1024
    print(f"\n✓ Zip created: {zip_filename} ({zip_size:.1f} MB)")
    
    print("\n" + "="*60)
    print("MODEL READY FOR DOWNLOAD!")
    print("="*60)
    print(f"Download: {zip_filename}")
    print(f"Size: {zip_size:.1f} MB")
    print("\nNext steps:")
    print("1. Download manuscript-bert-final.zip from Colab")
    print("2. Extract on your laptop")
    print("3. Copy to: manuscript-restoration/models/")
    print("4. Run: python test/gap_test.py")
    print("="*60 + "\n")


def main():
    """
    Main training pipeline for Google Colab.
    """
    print("\n" + "="*60)
    print("MANUSCRIPT BERT TRAINING - COLAB VERSION")
    print("70% Dataset - Optimized for Speed")
    print("="*60 + "\n")
    
    try:
        # Step 1: Setup environment
        device, in_colab = setup_colab_environment()
        
        # Step 2: Load reduced dataset
        train_dataset, eval_dataset = load_reduced_dataset()
        
        # Step 3: Load tokenizer
        print("Loading tokenizer...")
        tokenizer = BertTokenizer.from_pretrained(BASE_MODEL_NAME)
        print(f"  ✓ Tokenizer loaded\n")
        
        # Step 4: Tokenize datasets
        train_dataset, eval_dataset = tokenize_dataset(train_dataset, eval_dataset, tokenizer)
        
        # Step 5: Train model
        model, trainer = train_model(train_dataset, eval_dataset, tokenizer, device)
        
        # Step 6: Save and package
        save_and_package_model(model, tokenizer)
        
        print("🎉 SUCCESS! Training complete!")
        
        if in_colab:
            print("\n📥 TO DOWNLOAD:")
            print("Click the folder icon on the left → manuscript-bert-final.zip → Download")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user")
        print("Partial checkpoints may be saved")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

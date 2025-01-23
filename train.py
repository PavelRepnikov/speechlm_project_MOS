import os
import torch
import torchaudio
from torch.utils.data import DataLoader
from torch import nn, optim
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import wandb
from transformers import BertTokenizer
from torch.nn import HuberLoss
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import soundfile as sf
from torch.optim.lr_scheduler import CosineAnnealingLR


from src.datasets.somos_dataset import SomosDataset
from src.model.speechlm_mos import SpeechLMMosTTS

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# === Настройки ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 1e-4
AUDIO_DIR = "data/somos/audios"
TRAIN_JSON = "data/somos/audios/train.json"
VAL_JSON = "data/somos/audios/test.json"
SAVE_MODEL_PATH = "best_model.pth"
CHECKPOINT_PATH = "speechlmh_base_checkpoint_clean.pt"


# USE_MULTIPLE_GPUS = torch.cuda.device_count() > 1
USE_MULTIPLE_GPUS = False

MAIN_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_best_model(model, checkpoint_path):
    try:
        model.load_state_dict(torch.load(checkpoint_path))
        print(f"Loaded best model from {checkpoint_path}.")
        return model
    except FileNotFoundError:
        print(f"Checkpoint file not found at {checkpoint_path}.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
        return None


def collate_fn(batch):
    texts = [item['text'] for item in batch]
    audios = [item['audio'].squeeze(0) if item['audio'].ndim == 2 else item['audio'] for item in batch]
    # mos = torch.tensor([item['mos'] for item in batch], dtype=torch.float32) # Для регрессии
    mos = torch.tensor([int(item['mos']) - 1 for item in batch], dtype=torch.long) # Для классификации

    audio_lengths = torch.tensor([audio.size(0) for audio in audios], dtype=torch.long)
    padded_audios = pad_sequence(audios, batch_first=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    padded_audios = padded_audios.to(device)
    mos = mos.to(device)
    audio_lengths = audio_lengths.to(device)

    return {
        'text': texts,
        'audio': padded_audios, 
        'audio_lengths': audio_lengths, 
        'mos': mos
    }


def get_subset(dataset, percentage=1):

    subset_size = int(len(dataset) * (percentage / 100))
    indices = list(range(subset_size))
    return torch.utils.data.Subset(dataset, indices)



def train_epoch(loader, model, criterion, optimizer, epoch, device):
    model.train()
    total_loss = 0

    loop = tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch in loop:
        
        texts = batch["text"]
        audios = batch["audio"].to(device)
        targets = batch["mos"].to(device)
        
        # print(f"Targets shape: {targets.shape}")  # Should be [batch_size]
        # print(f"Targets content: {targets}")

        if targets.dim() > 1:
            targets = torch.argmax(targets, dim=1)


        
        outputs = model(audios, texts)

        # print(f"Outputs shape: {outputs.shape}")
        
        loss = criterion(outputs, targets)
    

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    wandb.log({"Train Loss": avg_loss, "Epoch": epoch})
    return avg_loss






def validate_epoch(loader, model, criterion, epoch, device, optimizer):
    model.eval()
    total_loss = 0
    total_targets = []
    total_outputs = []
    correct_examples = 0
    incorrect_examples = 0

    all_preds = []
    all_labels = []

    audio_table = wandb.Table(columns=["Audio", "Real MOS", "Predicted MOS", "Error", "Is Correct"])

    with torch.no_grad():
        loop = tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False)
        for batch_idx, batch in enumerate(loop):
            texts = batch["text"]
            audios = batch["audio"].to(device)
            targets = batch["mos"].float().to(device)

            text_input_ids = texts
            attention_mask = batch['attention_mask'].to(device) if 'attention_mask' in batch else None

            # Прямой проход
            outputs = model(audios, text_input_ids, attention_mask)
            outputs = outputs.float()
            targets = targets.long()

            # Compute loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            predicted_classes = torch.argmax(outputs, dim=1)



            for i, (audio, target, prediction) in enumerate(zip(audios, targets, predicted_classes)):
                error = target.item() - prediction.item()
                is_correct = target.item() == prediction.item()

                if is_correct and correct_examples < 10:
                    correct_examples += 1

                    audio_np = audio.cpu().numpy()
                    audio_table.add_data(
                        wandb.Audio(audio_np, sample_rate=16000),
                        target.item() + 1,
                        prediction.item() + 1,
                        error,
                        is_correct
                    )

                elif not is_correct and incorrect_examples < 10:
                    incorrect_examples += 1

                    audio_np = audio.cpu().numpy()
                    audio_table.add_data(
                        wandb.Audio(audio_np, sample_rate=16000),
                        target.item() + 1,
                        prediction.item() + 1,
                        error,
                        is_correct
                    )

                if correct_examples >= 10 and incorrect_examples >= 10:
                    break

            all_preds.extend(predicted_classes.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

            loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)

    acc = accuracy_score(all_labels, all_preds)
    precision_macro = precision_score(all_labels, all_preds, average='macro')
    recall_macro = recall_score(all_labels, all_preds, average='macro')
    f1_macro = f1_score(all_labels, all_preds, average='macro')

    precision_weighted = precision_score(all_labels, all_preds, average='weighted')
    recall_weighted = recall_score(all_labels, all_preds, average='weighted')
    f1_weighted = f1_score(all_labels, all_preds, average='weighted')

    wandb.log({
        "Val Loss": avg_loss,
        "Val Accuracy": acc,
        "Precision (Macro)": precision_macro,
        "Recall (Macro)": recall_macro,
        "F1 (Macro)": f1_macro,
        "Precision (Weighted)": precision_weighted,
        "Recall (Weighted)": recall_weighted,
        "F1 (Weighted)": f1_weighted,
        "Epoch": epoch,
        "Learning Rate": optimizer.param_groups[0]['lr'],
        "Audio Examples": audio_table  # Log the audio examples table
    })

    print(f"\n\n\n\n\n\n\nValidation Accuracy: {acc:.4f}")
    print(f"Precision (Macro): {precision_macro:.4f} | Recall (Macro): {recall_macro:.4f} | F1 (Macro): {f1_macro:.4f}")
    print(f"Precision (Weighted): {precision_weighted:.4f} | Recall (Weighted): {recall_weighted:.4f} | F1 (Weighted): {f1_weighted:.4f}\n\n\n\n\n\n\n\n\n")

    return avg_loss



def check_class_imbalance(dataset):

    mos_labels = [item['mos'] for item in dataset]
    
    label_counts = Counter(mos_labels)
    
    print("Class distribution:", label_counts)
    return label_counts


def compute_class_weights(class_counts, num_classes):
    total_samples = sum(class_counts.values())

    class_weights = [total_samples / (num_classes * class_counts.get(i, 0)) for i in range(1, num_classes + 1)]

    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)
    
    return class_weights_tensor



def freeze_model_except_upper_layers(model, num_unfrozen_text_layers=2, num_unfrozen_speech_layers=2):

    # Unfreeze the upper `num_unfrozen_text_layers` of the text encoder
    for i in range(num_unfrozen_text_layers):
        for name, param in model.named_parameters():
            if f"text_encoder.encoder.layer.{11 - i}." in name:
                param.requires_grad = True
                print(f"Unfrozen (Text Encoder): {name}")

    # Unfreeze the upper `num_unfrozen_speech_layers` of the speech encoder
    for i in range(num_unfrozen_speech_layers):
        for name, param in model.named_parameters():
            if f"speech_encoder.unit_encoder.layers.{5 - i}." in name: 
                param.requires_grad = True
                print(f"Unfrozen (Speech Encoder): {name}")



def unfreeze_all_weights(model):

    for name, param in model.named_parameters():
        param.requires_grad = True
        print(f"Unfrozen: {name}")




def main():


    train_dataset = SomosDataset(json_path=TRAIN_JSON, audio_dir=AUDIO_DIR)
    class_distribution = check_class_imbalance(train_dataset)

    num_classes = 5  # 5 классов
    class_weights_tensor = compute_class_weights(class_distribution, num_classes)

    print("Class weights:", class_weights_tensor)





    torch.manual_seed(42)
    if DEVICE.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    wandb.init(project="speechlm_mos_att2", config={
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "model_checkpoint": CHECKPOINT_PATH,
    })



    train_dataset = SomosDataset(json_path=TRAIN_JSON, audio_dir=AUDIO_DIR)
    val_dataset = SomosDataset(json_path=VAL_JSON, audio_dir=AUDIO_DIR)

    assert len(train_dataset) >= BATCH_SIZE, "Training dataset is too small!"
    assert len(val_dataset) >= BATCH_SIZE, "Validation dataset is too small!"

    # # # Use 1% of the dataset for both train and validation
    # train_dataset = get_subset(train_dataset, percentage=1)
    # val_dataset = get_subset(val_dataset, percentage=1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        # num_workers=2,
        collate_fn=collate_fn,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        # num_workers=2,
        collate_fn=collate_fn,
        drop_last=True
    )


    # model = SpeechLMMosTTS(checkpoint_path=CHECKPOINT_PATH).to(DEVICE)
    model = SpeechLMMosTTS(checkpoint_path=CHECKPOINT_PATH, class_weights=class_weights_tensor).to(DEVICE)
    # model = load_best_model(model, SAVE_MODEL_PATH)
    

    # freeze_model_except_upper_layers(model, num_unfrozen_text_layers=3, num_unfrozen_speech_layers=2)
    unfreeze_all_weights(model)


    print("\n\n\n\n\n\n")
    for name, param in model.named_parameters():
        print(f"Name: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
    print("\n\n\n\n\n\n")

    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Unfrozen: {name}")

    print("\n\n\n\n\n\n")


    if USE_MULTIPLE_GPUS:
        print(f"Using DataParallel with {torch.cuda.device_count()} GPUs.")
        model = torch.nn.DataParallel(model).to(DEVICE)
    else:
        print("Using a single GPU/CPU.")

    # Define loss criterion and optimizer
    # criterion = nn.MSELoss()
    # criterion = HuberLoss(delta=1.0)
    # criterion = nn.L1Loss()
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor.to(DEVICE))


    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)

    embedding_params = [
        p for n, p in model.named_parameters() 
        if "embedding" in n or "speech_encoder.embeddings" in n or "text_encoder.embeddings" in n
    ]

    other_params = [
        p for n, p in model.named_parameters() 
        if "embedding" not in n and "speech_encoder.embeddings" not in n and "text_encoder.embeddings" not in n
    ]

    optimizer = optim.Adam([
        {"params": embedding_params, "lr": LEARNING_RATE * 10},  # Higher LR for embeddings
        {"params": other_params, "lr": LEARNING_RATE}            # Default LR for other layers
    ])

    # Log learning rates
    print(f"Learning rate for embeddings: {optimizer.param_groups[0]['lr']}")
    print(f"Learning rate for other parameters: {optimizer.param_groups[1]['lr']}")


    # scheduler = ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.5, patience=3, verbose=True, min_lr=1e-6
    # )
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_loss = float("inf")
    for epoch in range(1, EPOCHS + 1):
        # Train for one epoch
        train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, DEVICE)
        
        # Validate for one epoch
        val_loss = validate_epoch(val_loader, model, criterion, epoch, DEVICE, optimizer)
    

        scheduler.step()
        # scheduler.step(val_loss)
        
        torch.cuda.empty_cache()

        print(f"Epoch {epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint_path = SAVE_MODEL_PATH
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved best model to {checkpoint_path}.")


if __name__ == "__main__":
    main()

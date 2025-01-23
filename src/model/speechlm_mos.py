import torch
import torch.nn as nn
from math import sqrt
from src.SpeechLM.SpeechLM import SpeechLM, SpeechLMConfig
from src.model.base_model import BaseModel
from src.model.fusion_blocks import LinearFusion, CrossAttentionFusion, StrictCrossAttention
from transformers import BertTokenizer, BertModel

# This is a commented out old version of model

# class SpeechLMMosTTS(BaseModel):
#     def __init__(
#         self, 
#         checkpoint_path: str = None, 
#         fusion_mode="cross", 
#         text_model_name="bert-base-uncased", 
#         text_max_len=512, 
#         num_classes=5, 
#         dropout=0.3, 
#         class_weights=None
#     ):
#         super().__init__()
#         assert fusion_mode in ["linear", "cross", "strict"], "Wrong fusion mode!"

#         # Speech and text encoders
#         self._setup_speech_encoder(checkpoint_path)
#         self._setup_text_encoder(text_model_name)

#         self.text_tokenizer = BertTokenizer.from_pretrained(text_model_name)
#         self.text_max_len = text_max_len

#         # Extract output dimensions
#         dummy_audio = torch.randn(1, 16000)  # 1-second audio input at 16kHz
#         audio_features = self.speech_encoder.extract_features(dummy_audio)[0]
#         self.speech_output_dim = audio_features.size(-1)

#         # Fusion block selection
#         if fusion_mode == "linear":
#             self.fusion = LinearFusion(
#                 input_dim_text=768, 
#                 input_dim_audio=self.speech_output_dim, 
#                 hidden_dim=512, 
#                 dropout=dropout
#             )
#         elif fusion_mode == "cross":
#             self.fusion = CrossAttentionFusion(
#                 input_dim_text=768, 
#                 input_dim_audio=self.speech_output_dim, 
#                 hidden_dim=256, 
#                 dropout=dropout
#             )
#         else:  # "strict"
#             self.fusion = StrictCrossAttention()

#         # Dropout layer
#         self.dropout = nn.Dropout(dropout)

#         # Classifier with additional layers
#         self.classifier = nn.Sequential(
#             nn.LayerNorm(self.fusion.output_dim),
#             nn.Linear(self.fusion.output_dim, 512),
#             nn.ReLU(),
#             self.dropout,
#             nn.LayerNorm(512),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             self.dropout,
#             nn.Linear(256, num_classes)
#         )

#         # Store class weights for loss computation
#         self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)

#     def forward(self, audio, transcription, attention_mask=None, **batch):
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         # Audio and text processing
#         audio = audio.to(device)
#         text_inputs = self.text_tokenizer(
#             transcription, 
#             padding=True, truncation=True, 
#             return_tensors="pt", max_length=self.text_max_len
#         )
#         text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

#         # Feature extraction
#         audio_features = self.speech_encoder.extract_features(audio)[0]
#         text_features = self._extract_text_features(transcription)

#         # Fusion and classification
#         fused_features = self.fusion(audio_features, text_features, attention_mask=attention_mask, **batch)
#         logits = self.classifier(fused_features)

#         return logits

#     def predict(self, logits):
#         """ Return predicted class labels """
#         pred = torch.argmax(logits, dim=-1)
#         return pred.int()

#     def compute_loss(self, logits, targets):
#         """ Compute classification loss """
#         class_weights = self.class_weights.to(logits.device)
#         criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
#         loss = criterion(logits, targets)
#         return loss

#     def _setup_speech_encoder(self, checkpoint_path):
#         assert checkpoint_path is not None, "Checkpoint path must be provided for the speech encoder."

#         checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
#         cfg = SpeechLMConfig(checkpoint['cfg']['model'])
#         self.speech_encoder = SpeechLM(cfg)
#         self.speech_encoder.load_state_dict(checkpoint['model'])
#         self.speech_encoder.eval()

#         # Freeze parameters
#         for p in self.speech_encoder.parameters():
#             p.requires_grad_(False)

#     def _setup_text_encoder(self, text_model_name):
#         self.text_encoder = BertModel.from_pretrained(text_model_name)
#         self.text_encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

#         # Freeze parameters
#         for p in self.text_encoder.parameters():
#             p.requires_grad_(False)

#     def _extract_text_features(self, transcription):
#         if isinstance(transcription, torch.Tensor):
#             if transcription.dim() == 1:
#                 transcription = transcription.tolist()
#             else:
#                 raise ValueError("Transcription tensor must be 1D.")

#         if isinstance(transcription, list):
#             transcription = [str(t) if isinstance(t, int) else t for t in transcription]

#         if isinstance(transcription, str):
#             transcription = [transcription]

#         if not isinstance(transcription, list) or not all(isinstance(t, str) for t in transcription):
#             raise ValueError("Transcription must be a string or a list of strings.")

#         # Tokenize and encode text
#         inputs = self.text_tokenizer(
#             transcription, 
#             return_tensors="pt", 
#             padding=True, 
#             truncation=True, 
#             max_length=self.text_max_len
#         )

#         # Forward through text encoder
#         device = next(self.text_encoder.parameters()).device
#         inputs = {key: value.to(device) for key, value in inputs.items()}
#         outputs = self.text_encoder(**inputs)
#         return outputs.last_hidden_state




 
# this is a working more complex version of model

class ResidualBlock(nn.Module):
    """Residual block with layer normalization and dropout."""
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        residual = x
        x = nn.ReLU()(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x


class MultiHeadAttentionFusion(nn.Module):
    """Multi-head attention fusion for combining audio and text features."""
    def __init__(self, input_dim_text, input_dim_audio, hidden_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.text_proj = nn.Linear(input_dim_text, hidden_dim)
        self.audio_proj = nn.Linear(input_dim_audio, hidden_dim)
        self.multihead_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim

    def forward(self, audio_features, text_features, attention_mask=None, **batch):
        # Project features to the same dimension
        text_features = self.text_proj(text_features)  # (batch_size, seq_len, hidden_dim)
        audio_features = self.audio_proj(audio_features)  # (batch_size, seq_len, hidden_dim)

        text_features = text_features.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)
        audio_features = audio_features.transpose(0, 1)  # (seq_len, batch_size, hidden_dim)

        # Multi-head attention
        fused_features, _ = self.multihead_attn(
            query=audio_features, 
            key=text_features, 
            value=text_features, 
            key_padding_mask=attention_mask
        )
        fused_features = self.dropout(fused_features)

        # Reshape back to (batch_size, seq_len, hidden_dim)
        fused_features = fused_features.transpose(0, 1)
        return fused_features


class AudioFeatureExtractor(nn.Module):
    """Convolutional layers for extracting hierarchical audio features."""
    def __init__(self, input_dim, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, channels, time)
        x = nn.ReLU()(self.conv1(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = nn.ReLU()(self.conv2(x))
        x = self.dropout(x)
        x = self.pool(x)
        x = x.transpose(1, 2)  # (batch, time, channels)
        return x


class SpeechLMMosTTS(BaseModel):
    def __init__(
        self, 
        checkpoint_path: str = None, 
        fusion_mode="cross", 
        text_model_name="bert-base-uncased", 
        text_max_len=512, 
        num_classes=5, 
        dropout=0.3, 
        class_weights=None
    ):
        super().__init__()
        assert fusion_mode in ["linear", "cross", "strict"], "Wrong fusion mode!"

        # Initialize dropout layer
        self.dropout = nn.Dropout(dropout)

        # Speech and text encoders
        self._setup_speech_encoder(checkpoint_path)
        self._setup_text_encoder(text_model_name)

        self.text_tokenizer = BertTokenizer.from_pretrained(text_model_name)
        self.text_max_len = text_max_len

        # Extract output dimensions
        dummy_audio = torch.randn(1, 16000)  # 1-second audio input at 16kHz
        audio_features = self.speech_encoder.extract_features(dummy_audio)[0]
        self.speech_output_dim = audio_features.size(-1)

        # Audio feature extractor
        self.audio_feature_extractor = AudioFeatureExtractor(
            input_dim=self.speech_output_dim, 
            hidden_dim=256, 
            dropout=dropout
        )

        # Fusion block
        self.fusion = MultiHeadAttentionFusion(
            input_dim_text=768, 
            input_dim_audio=256,  # Output of AudioFeatureExtractor
            hidden_dim=512, 
            num_heads=8, 
            dropout=dropout
        )

        # Classifier with residual connections
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.fusion.output_dim),
            nn.Linear(self.fusion.output_dim, 1024),
            nn.ReLU(),
            self.dropout,  # Now self.dropout is defined
            ResidualBlock(1024, 512, dropout),
            ResidualBlock(1024, 512, dropout),
            nn.Linear(1024, num_classes)
        )

        # Store class weights for loss computation
        self.class_weights = class_weights if class_weights is not None else torch.ones(num_classes)

    def forward(self, audio, transcription, attention_mask=None, **batch):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Audio processing
        audio = audio.to(device)
        audio_features = self.speech_encoder.extract_features(audio)[0]
        audio_features = self.audio_feature_extractor(audio_features)

        # Text processing
        text_features = self._extract_text_features(transcription)

        # Fusion and classification
        fused_features = self.fusion(audio_features, text_features, attention_mask=attention_mask, **batch)

        # Aggregate sequence dimension (e.g., mean pooling)
        fused_features = torch.mean(fused_features, dim=1)  # Shape: [batch_size, hidden_dim]

        logits = self.classifier(fused_features)  # Shape: [batch_size, num_classes]
        return logits

    def predict(self, logits):
        """Return predicted class labels."""
        pred = torch.argmax(logits, dim=-1)
        return pred.int()

    def compute_loss(self, logits, targets):
        """Compute classification loss."""
        class_weights = self.class_weights.to(logits.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        loss = criterion(logits, targets)
        return loss

    def _setup_speech_encoder(self, checkpoint_path):
        """Load and freeze the speech encoder."""
        assert checkpoint_path is not None, "Checkpoint path must be provided for the speech encoder."

        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        cfg = SpeechLMConfig(checkpoint['cfg']['model'])
        self.speech_encoder = SpeechLM(cfg)
        self.speech_encoder.load_state_dict(checkpoint['model'])
        self.speech_encoder.eval()

        # Freeze parameters
        for p in self.speech_encoder.parameters():
            p.requires_grad_(False)

    def _setup_text_encoder(self, text_model_name):
        """Load and freeze the text encoder."""
        self.text_encoder = BertModel.from_pretrained(text_model_name)
        self.text_encoder.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

        # Freeze parameters
        for p in self.text_encoder.parameters():
            p.requires_grad_(False)

    def _extract_text_features(self, transcription):
        """Extract text features using the text encoder."""
        if isinstance(transcription, str):
            transcription = [transcription]
        elif not isinstance(transcription, list) or not all(isinstance(t, str) for t in transcription):
            raise ValueError("Transcription must be a string or a list of strings.")

        # Tokenize and encode text
        inputs = self.text_tokenizer(
            transcription, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=self.text_max_len
        )

        # Forward through text encoder
        device = next(self.text_encoder.parameters()).device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        outputs = self.text_encoder(**inputs)
        return outputs.last_hidden_state
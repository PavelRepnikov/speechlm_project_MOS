# b4 crossentropy

# import torch
# from torch import nn
# from torch.nn import functional as F
# from math import sqrt
# import torch.nn as nn


# class BaseFusion(nn.Module):
#     def __init__(self):
#         super().__init__()

#     def _init_weights(self, mean=0.0, std=0.01):
#         for layer in self.modules():
#             if isinstance(layer, nn.Linear):
#                 nn.init.xavier_uniform_(layer.weight)
#                 if layer.bias is not None:
#                     layer.bias.data.fill_(0)
#             elif layer.__class__.__name__.find("Conv") != -1:
#                 layer.weight.data.normal_(mean=mean, std=std)

# # 1.0
# class LinearFusion(BaseFusion):
#     def __init__(self, input_dim=768, hidden_dim=128, dropout=0.1):
#         super().__init__()
#         self.projection = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )
#         self._init_weights()

#     def forward(self, x, y, **batch):
#         x_pred = self.projection(x).mean(dim=[1, 2])
#         y_pred = self.projection(y).mean(dim=[1, 2])
#         return 2 * torch.tanh(y_pred - x_pred) + 3
#         # return 4 * torch.sigmoid(y_pred - x_pred) + 1
#         # return 1.0 - torch.sigmoid(y_pred - x_pred)


# class CrossAttentionFusion(BaseFusion):
#     def __init__(self, input_dim_text=768, input_dim_audio=768, hidden_dim=128, dropout=0.1):
#         super().__init__()
#         # Проверим размерность данных для текстовых и аудио фичей
#         self.q_proj = nn.Linear(input_dim_text, hidden_dim)  # Для текста
#         self.k_proj = nn.Linear(input_dim_audio, hidden_dim)  # Для аудио
#         self.v_proj = nn.Linear(input_dim_audio, hidden_dim)  # Для аудио

#         self.projection = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, 1)
#         )
#         self._init_weights()

#     def forward(self, x, y, return_attention=False, **batch):
#         # print("Размерность текста:", x.shape)  # [batch_size, seq_len, input_dim_text]
#         # print("Размерность аудио:", y.shape)   # [batch_size, seq_len, input_dim_audio]
        
#         # Преобразуем данные в скрытые представления
#         query = self.q_proj(x)  # Текст (query)
#         key = self.k_proj(y)  # Аудио (key)
#         value = self.v_proj(y)  # Аудио (value)

#         # Механизм внимания
#         attention = F.softmax(torch.matmul(query, key.transpose(-1, -2)) / sqrt(key.shape[-1]), dim=-1)
#         context = torch.matmul(attention, value)  # Контекст из внимания

#         # Преобразование контекста через линейный слой
#         out = self.projection(context.mean(1).squeeze(1))  # Усреднение по последовательности

#         if return_attention:
#             return out, attention
#         return out




# class StrictCrossAttention(CrossAttentionFusion):
#     def __init__(self, input_dim=768, hidden_dim=128, dropout=0.1):
#         super().__init__(input_dim, hidden_dim, dropout)

#     def forward(self, x, y, l_value, return_attention=False, **batch):
#         """
#         Добавлен строгий механизм обработки данных:
#         - Маска создается на основе l_value, где значения > 0.5 считаются "чистыми".
#         - Разделение данных на чистые и шумные.
#         """
#         # Подготовка данных: x и y имеют форму [batch_size, time_steps, features]
#         mask = l_value > 0.5  # Применение маски на основе l_value
#         clean = torch.cat([x[mask], y[mask]], dim=-1)  # Чистые данные (где l_value > 0.5)
#         aug = torch.cat([y[~mask], x[~mask]], dim=-1)  # Шумные данные (где l_value <= 0.5)

#         # Применение cross-attention для чистых и шумных данных
#         out_clean = super().forward(clean, clean)  # Прогоняем чистые данные через CrossAttention
#         out_aug = super().forward(aug, aug)  # Прогоняем шумные данные через CrossAttention

#         if return_attention:
#             return out_clean, out_aug
#         return out_clean, out_aug


import torch
from torch import nn
from torch.nn import functional as F
from math import sqrt


class BaseFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def _init_weights(self, mean=0.0, std=0.01):
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    layer.bias.data.fill_(0)
            elif layer.__class__.__name__.find("Conv") != -1:
                layer.weight.data.normal_(mean=mean, std=std)


# 1.0
class LinearFusion(BaseFusion):
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.output_dim = 1  # Добавлен выходной размер
        self._init_weights()

    def forward(self, x, y, **batch):
        x_pred = self.projection(x).mean(dim=[1, 2])
        y_pred = self.projection(y).mean(dim=[1, 2])
        return 2 * torch.tanh(y_pred - x_pred) + 3


class CrossAttentionFusion(BaseFusion):
    def __init__(self, input_dim_text=768, input_dim_audio=768, hidden_dim=128, dropout=0.1):
        super().__init__()

        # Проекции для механизма внимания
        self.q_proj = nn.Linear(input_dim_text, hidden_dim)  # Для текста
        self.k_proj = nn.Linear(input_dim_audio, hidden_dim)  # Для аудио
        self.v_proj = nn.Linear(input_dim_audio, hidden_dim)  # Для аудио

        # Линейная проекция выхода
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        self.output_dim = 1  # Добавлен выходной размер
        self._init_weights()

    def forward(self, x, y, return_attention=False, **batch):
        query = self.q_proj(x)
        key = self.k_proj(y)
        value = self.v_proj(y)

        # Внимание
        attention = F.softmax(torch.matmul(query, key.transpose(-1, -2)) / sqrt(key.shape[-1]), dim=-1)
        context = torch.matmul(attention, value)

        out = self.projection(context.mean(1).squeeze(1))
        if return_attention:
            return out, attention
        return out


class StrictCrossAttention(CrossAttentionFusion):
    def __init__(self, input_dim=768, hidden_dim=128, dropout=0.1):
        super().__init__(input_dim, input_dim, hidden_dim, dropout)

    def forward(self, x, y, l_value, return_attention=False, **batch):
        mask = l_value > 0.5
        clean = torch.cat([x[mask], y[mask]], dim=-1)
        aug = torch.cat([y[~mask], x[~mask]], dim=-1)

        out_clean = super().forward(clean, clean)
        out_aug = super().forward(aug, aug)

        if return_attention:
            return out_clean, out_aug
        return out_clean, out_aug

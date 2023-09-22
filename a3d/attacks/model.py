import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class CustomKWSModel(nn.Module):
    def __init__(
        self,
        encoder:str="facebook/wav2vec2-base-960h",
        n_classes:int=12
    ):
        super().__init__()
        self.feature_extraction = AutoModel.from_pretrained(encoder)
        self.feature_extraction.requires_grad_(False)
        self.fc = nn.Linear(768, n_classes)

    def forward(self, x):
        # Extract wav2vec2 features and do an average 
        features = self.feature_extraction(x)
        features = features.last_hidden_state.transpose(1,2)

        # Global average pooling (on time dimension)
        features = F.avg_pool1d(features, kernel_size=features.shape[-1]).squeeze(-1)
        output = self.fc(features)

        return F.log_softmax(output, dim=-1)
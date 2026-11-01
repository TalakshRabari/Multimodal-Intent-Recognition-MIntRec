import torch
import torch.nn as nn
import torch.nn.functional as F


class PrototypeMemory(nn.Module):
    """
    Prototype Memory Module

    Learns one prototype for every intent class.
    """

    def __init__(self,
                 hidden_size,
                 num_classes):

        super().__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.prototypes = nn.Parameter(
            torch.randn(num_classes, hidden_size)
        )

        # Learn how much retrieved memory to use for each feature dimension.
        self.gate = nn.Linear(hidden_size * 2, 1)

        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, features, top_k=3):

    # Preserve original features
             original_features = features

    # Normalize features
             features = F.normalize(features, dim=1)

             prototypes = F.normalize(
                    self.prototypes,
                    dim=1
                  )

    # Cosine similarity
             similarity = torch.matmul(
                features,
                  prototypes.t()
                   )

    # Top-K retrieval
             top_values, top_indices = torch.topk(
                    similarity,
                    k=top_k,
                    dim=1
                   )

    # Attention over Top-K
             attention = F.softmax(
                        top_values,
                        dim=1
                      )

             selected_prototypes = self.prototypes[top_indices]

             memory_feature = torch.sum(
                     attention.unsqueeze(-1) * selected_prototypes,
                     dim=1
                   )

             gate = torch.sigmoid(
                     self.gate(torch.cat((original_features, memory_feature), dim=1))
                   )

             enhanced_feature = original_features + gate * memory_feature
 
             return (enhanced_feature, attention, top_indices, selected_prototypes)

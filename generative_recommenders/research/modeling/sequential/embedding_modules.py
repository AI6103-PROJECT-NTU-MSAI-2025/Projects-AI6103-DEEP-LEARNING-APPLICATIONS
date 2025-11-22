# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import abc

import torch

from generative_recommenders.research.modeling.initialization import truncated_normal


class EmbeddingModule(torch.nn.Module):
    @abc.abstractmethod
    def debug_str(self) -> str:
        pass

    @abc.abstractmethod
    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        pass

    @property
    @abc.abstractmethod
    def item_embedding_dim(self) -> int:
        pass


class LocalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.reset_params()

    def debug_str(self) -> str:
        return f"local_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class CategoricalEmbeddingModule(EmbeddingModule):
    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        item_id_to_category_id: torch.Tensor,
    ) -> None:
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )
        self.register_buffer("_item_id_to_category_id", item_id_to_category_id)
        self.reset_params()

    def debug_str(self) -> str:
        return f"cat_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        for name, params in self.named_parameters():
            if "_item_emb" in name:
                print(
                    f"Initialize {name} as truncated normal: {params.data.size()} params"
                )
                truncated_normal(params, mean=0.0, std=0.02)
            else:
                print(f"Skipping initializing params {name} - not configured")

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        item_ids = self._item_id_to_category_id[(item_ids - 1).clamp(min=0)] + 1
        return self._item_emb(item_ids)

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim


class ItemEmbeddingWithText(EmbeddingModule):

    def __init__(
        self,
        num_items: int,
        item_embedding_dim: int,
        text_embedding_dim: int,
        text_embeddings: torch.Tensor,
        fusion_mode: str = "sum",
        attention_dim: int = None, # dimension for D_att
        use_sigmoid_alpha: bool = True
    ) -> None:
        """
        Embedding module that combines learned item embeddings with precomputed textual embeddings.

        Args:
            num_items (int): Number of unique items.
            item_embedding_dim (int): Dimension of item ID embeddings.
            text_embedding_dim (int): Dimension of textual embeddings.
            text_embeddings (torch.Tensor): Precomputed item textual embeddings.
        """
        super().__init__()

        self._item_embedding_dim: int = item_embedding_dim
        self._item_emb: torch.nn.Embedding = torch.nn.Embedding(
            num_items + 1, item_embedding_dim, padding_idx=0
        )

        self.register_buffer("_text_embeddings", text_embeddings)

        self._text_projection: torch.nn.Linear = torch.nn.Linear(
            text_embedding_dim, item_embedding_dim
        )
  
        self._fusion_mode: str = fusion_mode
        #MLP+RelU
        if self._fusion_mode == "concat_mlp":
            self._concat_mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * item_embedding_dim, item_embedding_dim),
            torch.nn.ReLU(),
            )
        #Gated
        elif self._fusion_mode == "gated":
            self._gate_layer = torch.nn.Linear(2 * item_embedding_dim, item_embedding_dim)
        #RES MLP
        elif self._fusion_mode == "resnet_mlp":
            # R
            self._residual_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * item_embedding_dim, item_embedding_dim),
                torch.nn.ReLU(),
            )
        elif self._fusion_mode == "bi_attention":
            if attention_dim is None: 
                 raise ValueError("attention_dim must be provided for 'bi_attention' mode.")
            
            self.d_att = attention_dim
            # Q, K, V 
            # input：item_embedding_dim (D_emb)
            # output：attention_dim (D_att)
            self._cross_q_layer = torch.nn.Linear(item_embedding_dim, self.d_att)
            self._cross_k_layer = torch.nn.Linear(item_embedding_dim, self.d_att)
            self._cross_v_layer = torch.nn.Linear(item_embedding_dim, self.d_att)
            # final fusion layer (2 * D_att) -> item_embedding_dim (D_emb)
            self._final_fusion_mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * self.d_att, item_embedding_dim),
                torch.nn.ReLU(),
            )
        elif self._fusion_mode == "weighted_sum":
            self.weighted_sum_alpha = torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
            self._use_sigmoid_alpha = use_sigmoid_alpha
            # print("*"*10)
            # print(f"initializing weighted_sum with weighted_sum_alpha:{self.weighted_sum_alpha}, _use_sigmoid_alpha:{self._use_sigmoid_alpha}")
            # print("*"*10)
            
        self.reset_params()

    def debug_str(self) -> str:
        return f"text_emb_d{self._item_embedding_dim}"

    def reset_params(self) -> None:
        FUSION_STD = 0.005
        for name, params in self.named_parameters():
            if "_item_emb" in name or "_text_projection" in name:
                print(f"Initialize {name} as truncated normal: {params.data.size()} params")
                truncated_normal(params.data, mean=0.0, std=0.02)
            # (concat_mlp)
            if "_concat_mlp" in name or "_residual_mlp" in name: # 包含新残差层
                if 'weight' in name:
                    truncated_normal(params.data, mean=0.0, std=FUSION_STD) 
                elif 'bias' in name:
                    torch.nn.init.constant_(params.data, 0.0)
            if "_gate_layer" in name:
                if 'weight' in name:
                    truncated_normal(params.data, mean=0.0, std=FUSION_STD)
                elif 'bias' in name:
                    # 关键修复：设为 1.0，使 Sigmoid(1.0) ~ 0.73，初始时偏向 ID 嵌入
                    torch.nn.init.constant_(params.data, 1)
                    
            if "_cross_q_layer" in name or "_cross_k_layer" in name or "_cross_v_layer" in name or "_final_fusion_mlp" in name:
                if 'weight' in name:
                    # Q, K, V  Final MLP init W 
                    truncated_normal(params.data, mean=0.0, std=FUSION_STD) 
                elif 'bias' in name:
                    # Q, K, V  Final MLP init bais
                    torch.nn.init.constant_(params.data, 0.0)

            if "weighted_sum_alpha" in name:
                print(f"Initialize {name} as scalar parameter")

            

    def get_item_embeddings(self, item_ids: torch.Tensor) -> torch.Tensor:
        """
        Retrieve combined embeddings for item IDs.
        Args:
            item_ids (torch.Tensor): Tensor of item IDs (batch_size, sequence_length).
        Returns:
            torch.Tensor: Combined item embeddings.
        """
        device = item_ids.device  # Ensure all tensors are on the same device

        # Move embeddings to the correct device
        item_embeds = self._item_emb(item_ids.to(device))  # Learned embeddings
        text_embeds = self._text_embeddings[item_ids.to(device)]  # Precomputed textual embeddings
        projected_text_embeds = self._text_projection(text_embeds.to(device))  # Projected textual embeddings

        # let's try some more fusion
        fused_embeddings: torch.Tensor
        if self._fusion_mode == "sum":
            # Summation Fusion
            fused_embeddings = item_embeds + projected_text_embeds

        elif self._fusion_mode == "weighted_sum":
            if self._use_sigmoid_alpha:
                w = torch.sigmoid(self.weighted_sum_alpha)
                # print(f"embedding: using sigmoid weighted sum, w:{w}")
                return w * item_embeds + (1.0 - w) * projected_text_embeds
                
            return self.weighted_sum_alpha * item_embeds + (1.0 - self.weighted_sum_alpha) * projected_text_embeds
            
        elif self._fusion_mode == "concat_mlp":
            # Concatenation and Projection, mlp fusion
            combined = torch.cat([item_embeds, projected_text_embeds], dim=-1)
            fused_embeddings = self._concat_mlp(combined)
        elif self._fusion_mode == "gated":
            # Gated Fusion
            combined = torch.cat([item_embeds, projected_text_embeds], dim=-1)
            # alpha: alpha = sigmoid(Linear(concat))
            alpha = torch.sigmoid(self._gate_layer(combined))
            # gated : alpha * ID_Embed + (1 - alpha) * Text_Embed
            fused_embeddings = alpha * item_embeds + (1 - alpha) * projected_text_embeds
        # New: ResNet Residual MLP Fusion
        elif self._fusion_mode == "resnet_mlp":
            # 1. 拼接输入
            combined = torch.cat([item_embeds, projected_text_embeds], dim=-1)
            # 2. 计算残差 R
            residual = self._residual_mlp(combined) 
            # 3. 残差连接：E_Fused = E_ID + R
            fused_embeddings = item_embeds + residual

        elif self._fusion_mode == "bi_attention":

            # Shapes: (B, L, D_emb) -> (B, L, D_att)
            Q_id = self._cross_q_layer(item_embeds)
            K_text = self._cross_k_layer(projected_text_embeds)
            V_text = self._cross_v_layer(projected_text_embeds)
            
            Q_text = self._cross_q_layer(projected_text_embeds)
            K_id = self._cross_k_layer(item_embeds)
            V_id = self._cross_v_layer(item_embeds)
            
            # Store original B, L, D_att for reshaping
            B, L, D_att = Q_id.shape
            
            # 2. Flatten B and L dimensions for item-wise fusion
            # Shapes: (B, L, D_att) -> (B*L, D_att)
            BL = B * L
            Q_id_flat = Q_id.reshape(BL, D_att)
            K_text_flat = K_text.reshape(BL, D_att)
            V_text_flat = V_text.reshape(BL, D_att)
            
            Q_text_flat = Q_text.reshape(BL, D_att)
            K_id_flat = K_id.reshape(BL, D_att)
            V_id_flat = V_id.reshape(BL, D_att)
            
            # --- direction 1: E_ID -> E_Text (ID enhanced) ---
            # Score (Dot Product): (B*L, D_att) * (B*L, D_att) -> (B*L)
            dot_product_score_id_to_text = torch.sum(Q_id_flat * K_text_flat, dim=-1)  
            # Unsqueeze to add Sequence and Query dimensions for Attention ops: (B*L) -> (B*L, 1, 1)
            scores_id_to_text = dot_product_score_id_to_text.unsqueeze(1).unsqueeze(1) 
            # Softmax: (B*L, 1, 1)
            alpha_id_to_text = torch.softmax(scores_id_to_text / (self.d_att ** 0.5), dim=-1)
            # V unsqueeze: (B*L, D_att) -> (B*L, 1, D_att)
            V_text_flat_e = V_text_flat.unsqueeze(1)
            # Attention-Value Weighting: (B*L, 1, 1) @ (B*L, 1, D_att) -> (B*L, 1, D_att)
            # Squeeze: (B*L, 1, D_att) -> (B*L, D_att)
            E_ID_enhanced_flat = torch.matmul(alpha_id_to_text, V_text_flat_e).squeeze(1) 
            
            # --- direction 2: E_Text -> E_ID (Text enhanced) ---
            dot_product_score_text_to_id = torch.sum(Q_text_flat * K_id_flat, dim=-1)
            scores_text_to_id = dot_product_score_text_to_id.unsqueeze(1).unsqueeze(1)
            alpha_text_to_id = torch.softmax(scores_text_to_id / (self.d_att ** 0.5), dim=-1)
            V_id_flat_e = V_id_flat.unsqueeze(1)
            E_Text_enhanced_flat = torch.matmul(alpha_text_to_id, V_id_flat_e).squeeze(1) 
     
            # Concatenation: (B*L, D_att) + (B*L, D_att) -> (B*L, 2*D_att)
            final_combined_flat = torch.cat([E_ID_enhanced_flat, E_Text_enhanced_flat], dim=-1) 
            
            # MLP Projection: (B*L, 2*D_att) -> (B*L, D_emb)
            fused_embeddings_flat = self._final_fusion_mlp(final_combined_flat)

            # Reshape: (B*L, D_emb) -> (B, L, D_emb)
            fused_embeddings = fused_embeddings_flat.reshape(B, L, -1)
            #print('emb:', fused_embeddings.shape)

        elif self._fusion_mode == "no_fusion":
            fused_embeddings = item_embeds
            
        else:
            raise ValueError(
                f"Unknown fusion mode: {self._fusion_mode}. Supported modes are 'sum', 'concat_mlp', 'gated'.")
            
        return fused_embeddings

    @property
    def item_embedding_dim(self) -> int:
        return self._item_embedding_dim

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_


class FeatureExtraction(nn.Module):
    """
    Feature Extraction (FE)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Kernel size of Conv2d. Default: 3
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FeatureExtraction, self).__init__()

        # FE
        self.fe = nn.Sequential(
            # Conv + LReLU + IN
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x):  # B 3 H W
        # FE
        x = self.fe(x)  # B C H W
        return x


class MultiScaleFeatureExtraction(nn.Module):
    """
    MultiScale Feature Extraction (MFE)
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
    """

    def __init__(self, in_channels, out_channels):
        super(MultiScaleFeatureExtraction, self).__init__()

        # MultiScale (3*3 + 5*5 + 7*7)
        self.scale_3 = FeatureExtraction(in_channels, out_channels, kernel_size=3)
        self.scale_5 = FeatureExtraction(in_channels, out_channels, kernel_size=5)
        self.scale_7 = FeatureExtraction(in_channels, out_channels, kernel_size=7)

    def forward(self, x):  # B 3 H W
        # MultiScale (3*3 + 5*5 + 7*7)
        x1 = self.scale_3(x)  # B C H W
        x2 = self.scale_5(x)  # B C H W
        x3 = self.scale_7(x)  # B C H W
        # Concat
        x = torch.cat([x1, x2, x3], dim=1)  # B 3C H W
        return x


class ChannelAttention(nn.Module):
    """
    Channel Attention (CA)
    Args:
        in_channels (int): Number of input channels.
        ratio (int): Ratio of MLP. Default: 8
    """

    def __init__(self, in_channels, ratio=8):
        super(ChannelAttention, self).__init__()

        # Avg Pool & Max pool
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # MLP
        self.mlp = nn.Sequential(
            # Conv + LReLU + Conv (Replace Linear + LReLU + Linear)
            nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
        )
        # Sigmoid -> Weight(C*1*1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # B C H W
        # CA -> Weight
        w = self.mlp(self.avg_pool(x)) + self.mlp(self.max_pool(x))
        w = self.sigmoid(w)  # B C 1 1
        # Weight * X
        x = w * x  # B C H W
        return x


class SpatialAttention(nn.Module):
    """
    Spatial Attention (SA)
    Args:
        kernel_size (int): Kernel size of Conv2d. Default: 7
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        # Conv (B 2 H W -> B 1 H W)
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        # Sigmoid -> Weight(1*H*W)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):  # B C H W
        # SA -> Weight
        avg_out = torch.mean(x, dim=1, keepdim=True)  # B 1 H W
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # B 1 H W
        w = self.conv(torch.cat([avg_out, max_out], dim=1))  # B 2 H W -> B 1 H W
        w = self.sigmoid(w)  # B 1 H W
        # Weight * X
        x = w * x  # B C H W
        return x


class ChannelSpatialAttentionModule(nn.Module):
    """
    Channel Spatial Attention Module (CSAM)
    Args:
        in_channels (int): Number of input channels.
    """

    def __init__(self, in_channels):
        super(ChannelSpatialAttentionModule, self).__init__()

        # Fusion
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        # CA & SA
        self.ca = ChannelAttention(in_channels)
        self.sa = SpatialAttention()
        # Fusion
        self.conv2 = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)

    def forward(self, x):  # B C H W
        # Fusion
        conv1_out = self.conv1(x)  # B C H W
        # CA SA in Parallel
        cs_out = self.ca(conv1_out) + self.sa(conv1_out)  # B C H W
        # Fusion
        conv2_out = self.conv2(cs_out)  # B C H W
        # Residual
        out = conv2_out + x  # B C H W
        return out


class ChannelSpatialAttentionModuleGroup(nn.Module):
    """
    ChannelSpatialAttentionModuleGroup (CSAM Group)
    Args:
        in_channels (int): Number of input channels.
        nums (int): Number of CSAMs.
    """

    def __init__(self, in_channels, nums):
        super(ChannelSpatialAttentionModuleGroup, self).__init__()

        # CSAM Group = CSAM * nums
        modules = [ChannelSpatialAttentionModule(in_channels=in_channels) for _ in range(nums)]
        self.group = nn.Sequential(*modules)

    def forward(self, x):  # B C H W
        # ChannelSpatialAttentionModuleGroup
        out = self.group(x)  # B C H W
        return out


class DownSample(nn.Module):
    """
    DownSample (DS, Down Channel And Size)
    Args:
        in_channels (int): Number of input channels.
    """
    def __init__(self, in_channels):
        super(DownSample, self).__init__()

        # DS
        self.ds = nn.Sequential(
            # Part1: Conv + LReLU + MaxPool (B C H W -> B C//4 H//2 W//2)
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1, bias=False),  # B C//4 H W
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # B C//4 H//2 W//2
            # Part2: Conv + LReLU + MaxPool (B C//4 H//2 W//2 -> B 1 H//4 W//4)
            nn.Conv2d(in_channels // 4, 1, 3, padding=1, bias=False),  # B 1 H//2 W//2
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # B 1 H//4 W//4
        )

    def forward(self, x):  # B C H W
        x = self.ds(x)  # B 1 H//4 W//4
        return x


class CSAMBranch(nn.Module):
    """
    CSAM Branch of ATUIQP
    Args:
        input_size (tuple(int)): Size of input image. Default: (224, 224)
        in_channels (int): Number of input channels. Default: 3
        embedding_channels (int): Number of embedding channels. Default: 24
        nums (int): Number of CSAMs. Default: 5
        map_channel (int): Number of mapping channels. Default: None
    """
    def __init__(self, input_size=(224, 224), in_channels=3, embedding_channels=24, nums=5, map_channel=None):
        super(CSAMBranch, self).__init__()

        assert embedding_channels % 3 == 0, 'embedding_channels % 3 must be 0'
        # MultiScaleFeatureExtraction (MFE)
        self.mfe = MultiScaleFeatureExtraction(in_channels, embedding_channels//3)
        # ChannelSpatialAttentionModuleGroup (CSAMGroup)
        self.csamgroup = ChannelSpatialAttentionModuleGroup(in_channels=embedding_channels, nums=nums)
        # DownSample (DS)
        self.ds = DownSample(in_channels=embedding_channels)
        # MapLinear (H//4 * W//4 -> map_channel)
        self.maplinear = nn.Linear(input_size[0]//4 * input_size[1]//4, map_channel, bias=False)

    def forward(self, x):  # B 3 H W
        # MultiScaleFeatureExtraction (MFE)
        x = self.mfe(x)  # B C H W
        # ChannelSpatialAttentionModuleGroup (CSAMGroup)
        x = self.csamgroup(x)  # B C H W
        # DownSample (DS)
        x = self.ds(x)  # B 1 H//4 W//4
        # MapLinear
        x = torch.flatten(x, 1)  # B H//4 * W//4
        out = self.maplinear(x)  # B Map Channel
        return out


################################################################################################################


def _init_weights(m):
    """
    Initialization Weights
    """
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif isinstance(m, (nn.LayerNorm, nn.InstanceNorm2d, nn.BatchNorm2d)):
        if m.weight is not None:
            nn.init.constant_(m.weight, 1.0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class DropPath(nn.Module):
    """
    Drop Path (To Address Overfitting)
    Args:
        drop_prob (float): Probability of dropout. Default: 0.0
    """
    def __init__(self,
                 drop_prob=0.0):
        super(DropPath, self).__init__()

        self.drop_prob = drop_prob

    def drop_path(self, x, drop_prob: float = 0.0, training: bool = False):

        if drop_prob == 0.0 or not training:
            return x
        keep_prob = 1.0 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # B (1,)*(3-1) -> B 1 1
        # 1.0 - drop_prob + [0.0, 1,0) -> [1.0 - drop_prob, 2.0 - drop_prob)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        # 1.0 - drop_prob -> 0,  2.0 - drop_prob -> 1
        random_tensor.floor_()
        # Keep Mathematical Expectations Consistent (without drop_path)
        # (1-p)*a+p*0 = (1-p)a, need div 1-p
        output = x.div(keep_prob) * random_tensor
        return output

    def forward(self, x):  # B N C
        # Drop Path
        x = self.drop_path(x, self.drop_prob)  # B N C
        return x


class FeedForwardNet(nn.Module):
    """
    Feed Forward Net (FFN)
    Args:
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features. Default: None
        out_features (int): Number of output features. Default: None
        act_function (nn.Module): Activation function. Default: nn.GELU
        dropout_ratio (float): FFN dropout ratio. Default: 0.0
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_function=nn.GELU,
                 dropout_ratio=0.0):
        super(FeedForwardNet, self).__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_function()
        self.drop = nn.Dropout(p=dropout_ratio)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PatchEmbedding(nn.Module):
    """
    Patch Embedding
    Args:
        in_channels (int): Number of input channels.
        patch_size (int): Size of path.
        embedding_dimensions (int): Number of embedding dimensions.
        norm_layer (nn.Module): Normalization layer.
    """
    def __init__(self,
                 in_channels,
                 patch_size,
                 embedding_dimensions,
                 norm_layer=None):
        super(PatchEmbedding, self).__init__()

        # Projection + Normalization
        self.proj = nn.Conv2d(in_channels, embedding_dimensions, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embedding_dimensions) if norm_layer else nn.Identity()

    def forward(self, x):  # B 3 H W
        # Projection + Flatten + Normalization
        x = self.proj(x)  # B C H//P W//P
        x = x.flatten(2).transpose(1, 2)  # B C H//P*W//P (B C N) -> B N C
        x = self.norm(x)  # B N C, if norm_layer=None then f(x)=x
        return x


class MultiHeadSelfAttention(nn.Module):
    """
    Multi Head Self Attention (MHSA)
    Args:
        embedding_dimensions (int): Number of embedding dimensions.
        heads (int): Number of self attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        attention_drop_ratio (float): Attention dropout ratio. Default: 0.0
        projection_drop_ratio (float): Projection dropout ratio. Default: 0.0
    """
    def __init__(self,
                 embedding_dimensions,
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop_ratio=0.0,
                 projection_drop_ratio=0.0):
        super(MultiHeadSelfAttention, self).__init__()

        # Heads + Head Dimension (C_head) + Scale
        self.heads = heads
        head_dimension = embedding_dimensions // heads
        self.scale = qk_scale or head_dimension ** -0.5
        # QKV Linear
        self.qkv = nn.Linear(embedding_dimensions, embedding_dimensions * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(p=attention_drop_ratio)
        # Output Projection
        self.proj = nn.Linear(embedding_dimensions, embedding_dimensions)
        self.proj_drop = nn.Dropout(p=projection_drop_ratio)

    # query: B 1 C(Decoder input) or B N C(Encoder input), key: None or B N C, value: None or B N C
    def forward(self, query=None, key=None, value=None):
        # Calculate Q, K, V
        if query is not None and key is None and value is None:
            B, N, C = query.shape
            # B N C -> B N 3C -> B N 3 heads C_head -> 3 B heads N C_head
            qkv = self.qkv(query).reshape(B, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
            # Q: B heads N C_head,  K: B heads N C_head,  V: B heads N C_head
            q, k, v = qkv[0], qkv[1], qkv[2]
        elif query is not None and key is not None and value is not None:
            B, N, C = key.shape
            # B 1(N) C -> B 1(N) heads C_head -> B heads 1(N) C_head
            # Q: B heads 1 C_head, K: B heads N C_head,  V: B heads N C_head
            q = query.reshape(B, query.shape[1], self.heads, C // self.heads).permute(0, 2, 1, 3)
            k = key.reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
            v = value.reshape(B, N, self.heads, C // self.heads).permute(0, 2, 1, 3)
        # Calculate Q and K Attention
        # B heads N C_head @ B heads C_head N -> B heads N N
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        # Attention @ V
        # B heads N N @ B heads N C_head -> B heads N C_head -> B N heads C_head -> B N C
        x = (attn @ v).transpose(1, 2).reshape(B, query.shape[1], C)
        # Output Projection
        x = self.proj(x)
        x = self.proj_drop(x)  # B query.shape[1] C(Decoder output) or B N C(Encoder output)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer
    Args:
        embedding_dimensions (int): Number of embedding dimensions.
        heads (int): Number of self attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        attention_drop_ratio (float): Attention dropout ratio. Default: 0.0
        projection_drop_ratio (float): Projection dropout ratio. Default: 0.0
        drop_path_ratio (float): Dropout path ratio. Default: 0.0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim. Default: 4.0
        act_function (nn.Module): Activation function. Default: nn.GELU
        dropout_ratio (float): Dropout ratio. Default: 0.0
    """
    def __init__(self,
                 embedding_dimensions,
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop_ratio=0.0,
                 projection_drop_ratio=0.0,
                 drop_path_ratio=0.0,
                 norm_layer=nn.LayerNorm,
                 ffn_ratio=4.0,
                 act_function=nn.GELU,
                 dropout_ratio=0.0):
        super(TransformerEncoderLayer, self).__init__()

        # LN + MHSA
        self.norm1 = norm_layer(embedding_dimensions)
        self.attn = MultiHeadSelfAttention(embedding_dimensions, heads, qkv_bias, qk_scale,
                                           attention_drop_ratio, projection_drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        # LN + FFN
        self.norm2 = norm_layer(embedding_dimensions)
        ffn_hidden_dimension = int(embedding_dimensions * ffn_ratio)
        self.ffn = FeedForwardNet(in_features=embedding_dimensions, hidden_features=ffn_hidden_dimension,
                                  act_function=act_function, dropout_ratio=dropout_ratio)

    def forward(self, x):  # B N C
        # LN + MHSA + Residual
        x = x + self.drop_path(self.attn(self.norm1(x)))  # B N C
        # LN + FFN + Residual
        x = x + self.drop_path(self.ffn(self.norm2(x)))  # B N C
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    Args:
        depths (int): Number of transformer encoder layers.
        embedding_dimensions (int): Number of embedding dimensions.
        heads (int): Number of self attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        attention_drop_ratio (float): Attention dropout ratio. Default: 0.0
        projection_drop_ratio (float): Projection dropout ratio. Default: 0.0
        drop_path_ratio (float): Dropout path ratio. Default: 0.0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim. Default: 4.0
        act_function (nn.Module): Activation function. Default: nn.GELU
        dropout_ratio (float): Dropout ratio. Default: 0.0
    """
    def __init__(self,
                 depths,
                 embedding_dimensions,
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop_ratio=0.0,
                 projection_drop_ratio=0.0,
                 drop_path_ratio=0.0,
                 norm_layer=nn.LayerNorm,
                 ffn_ratio=4.0,
                 act_function=nn.GELU,
                 dropout_ratio=0.0):
        super(TransformerEncoder, self).__init__()

        # drop_path_ratio, stochastic depth decay rule
        drop_path_array = [x.item() for x in torch.linspace(0, drop_path_ratio, depths)]
        # Transformer Encoder
        # Note: nn.Sequential() only support single input and single output, using nn.Sequential
        self.encoder_layers = nn.Sequential(*[
                TransformerEncoderLayer(embedding_dimensions, heads, qkv_bias, qk_scale, attention_drop_ratio,
                                        projection_drop_ratio, drop_path_array[i], norm_layer, ffn_ratio, act_function,
                                        dropout_ratio)
                for i in range(depths)
            ])
        self.norm = norm_layer(embedding_dimensions)

    def forward(self, x):  # B N+1 C
        # Transformer Encoder
        x = self.encoder_layers(x)  # B N+1 C
        x = self.norm(x)
        token = x[:, 0]  # B C
        token = token.reshape(token.shape[0], 1, -1)  # B 1 C
        qkv = x[:, 1:, :]  # B N C
        return token, qkv


class TransformerDecoderLayer(nn.Module):
    """
    Transformer Decoder Layer
    Args:
        embedding_dimensions (int): Number of embedding dimensions.
        heads (int): Number of self attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        attention_drop_ratio (float): Attention dropout ratio. Default: 0.0
        projection_drop_ratio (float): Projection dropout ratio. Default: 0.0
        drop_path_ratio (float): Dropout path ratio. Default: 0.0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim. Default: 4.0
        act_function (nn.Module): Activation function. Default: nn.GELU
        dropout_ratio (float): Dropout ratio. Default: 0.0
    """
    def __init__(self,
                 embedding_dimensions,
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop_ratio=0.0,
                 projection_drop_ratio=0.0,
                 drop_path_ratio=0.0,
                 norm_layer=nn.LayerNorm,
                 ffn_ratio=4.0,
                 act_function=nn.GELU,
                 dropout_ratio=0.0):
        super(TransformerDecoderLayer, self).__init__()

        # LN + MHSA
        self.norm1 = norm_layer(embedding_dimensions)
        self.attn1 = MultiHeadSelfAttention(embedding_dimensions, heads, qkv_bias, qk_scale,
                                           attention_drop_ratio, projection_drop_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0.0 else nn.Identity()
        # LN + MHCA
        self.norm2 = norm_layer(embedding_dimensions)
        self.attn2 = MultiHeadSelfAttention(embedding_dimensions, heads, qkv_bias, qk_scale,
                                            attention_drop_ratio, projection_drop_ratio)
        # LN + FFN
        self.norm3 = norm_layer(embedding_dimensions)
        ffn_hidden_dimension = int(embedding_dimensions * ffn_ratio)
        self.ffn = FeedForwardNet(in_features=embedding_dimensions, hidden_features=ffn_hidden_dimension,
                                  act_function=act_function, dropout_ratio=dropout_ratio)

    # query: B 1 C, key: None or B N C, value: None or B N C
    def forward(self, query=None, key=None, value=None):
        x = query
        # LN + MHSA + Residual
        x = x + self.drop_path(self.attn1(self.norm1(x)))  # B 1 C
        # LN + MHCA + Residual
        x = x + self.drop_path(self.attn2(self.norm2(x), key, value))  # B 1 C
        # LN + FFN + Residual
        x = x + self.drop_path(self.ffn(self.norm3(x)))  # B 1 C
        query = x
        return query, key, value


class TransformerDecoder(nn.Module):
    """
    Transformer Decoder
    Args:
        depths (int): Number of transformer decoder layers.
        embedding_dimensions (int): Number of embedding dimensions.
        heads (int): Number of self attention heads.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        attention_drop_ratio (float): Attention dropout ratio. Default: 0.0
        projection_drop_ratio (float): Projection dropout ratio. Default: 0.0
        drop_path_ratio (float): Dropout path ratio. Default: 0.0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim. Default: 4.0
        act_function (nn.Module): Activation function. Default: nn.GELU
        dropout_ratio (float): Dropout ratio. Default: 0.0
    """
    def __init__(self,
                 depths,
                 embedding_dimensions,
                 heads,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop_ratio=0.0,
                 projection_drop_ratio=0.0,
                 drop_path_ratio=0.0,
                 norm_layer=nn.LayerNorm,
                 ffn_ratio=4.0,
                 act_function=nn.GELU,
                 dropout_ratio=0.0):
        super(TransformerDecoder, self).__init__()

        # drop_path_ratio, stochastic depth decay rule
        drop_path_array = [x.item() for x in torch.linspace(0, drop_path_ratio, depths)]
        # Transformer Decoder
        # Note: nn.Sequential() only support single input and single output, using nn.ModuleList
        self.decoder_layers = nn.ModuleList([
                TransformerDecoderLayer(embedding_dimensions, heads, qkv_bias, qk_scale, attention_drop_ratio,
                                        projection_drop_ratio, drop_path_array[i], norm_layer, ffn_ratio, act_function,
                                        dropout_ratio)
                for i in range(depths)
            ])
        self.norm = norm_layer(embedding_dimensions)

    # query: B 1 C, key: None or B N C, value: None or B N C
    def forward(self, query=None, key=None, value=None):
        # Transformer Decoder
        for _, layer in enumerate(self.decoder_layers):
            query, key, value = layer(query, key, value)  # B 1 C
        x = query
        x = self.norm(x)  # B 1 C
        x = torch.flatten(x, 1)  # B C
        return x


class TransformerBranch(nn.Module):
    """
    Transformer Branch
    Args:
        input_size (tuple(int)): Size of input image. Default: (224, 224)
        in_channels (int): Number of input channels. Default: 3
        patch_size (int): Size of path. Default: 16
        embedding_dimensions (int): Number of embedding dimensions. Default: 384
        encoder_depths (int): Number of transformer encoder layers. Default: 12
        decoder_depths (int): Number of transformer decoder layers. Default: 1
        encoder_heads (int): Number of self attention heads in transformer encoder layer. Default: 6
        decoder_heads (int): Number of self attention heads in transformer decoder layer. Default: 6
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None): Override default qk scale of head_dim ** -0.5 if set. Default: None
        attention_drop_ratio (float): Attention dropout ratio. Default: 0.0
        projection_drop_ratio (float): Projection dropout ratio. Default: 0.0
        drop_path_ratio (float): Dropout path ratio. Default: 0.0
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm
        ffn_ratio (float): Ratio of ffn hidden dim to embedding dim. Default: 4.0
        act_function (nn.Module): Activation function. Default: nn.GELU
        dropout_ratio (float): Dropout ratio. Default: 0.0
    """
    def __init__(self,
                 input_size=(224, 224),
                 in_channels=3,
                 patch_size=16,
                 embedding_dimensions=384,
                 encoder_depths=12,
                 decoder_depths=1,
                 encoder_heads=6,
                 decoder_heads=6,
                 qkv_bias=False,
                 qk_scale=None,
                 attention_drop_ratio=0.0,
                 projection_drop_ratio=0.0,
                 drop_path_ratio=0.0,
                 norm_layer=nn.LayerNorm,
                 ffn_ratio=4.0,
                 act_function=nn.GELU,
                 dropout_ratio=0.0):
        super(TransformerBranch, self).__init__()

        # Patch Embedding & Position Embedding
        # Patch Embedding
        self.patch_embedding = PatchEmbedding(in_channels, patch_size, embedding_dimensions)  # B N C
        patches = input_size[0] // patch_size * input_size[1] // patch_size
        # Add Token
        self.token = nn.Parameter(torch.randn(1, 1, embedding_dimensions))  # 1 1 C, torch.randn
        # Position Embedding
        self.position_embedding = nn.Parameter(torch.zeros(1, patches+1, embedding_dimensions))  # 1 N+1 C, torch.zeros
        self.position_drop = nn.Dropout(p=dropout_ratio)

        # Transformer Encoder
        self.encoder = TransformerEncoder(encoder_depths, embedding_dimensions, encoder_heads,
                                          qkv_bias, qk_scale, attention_drop_ratio, projection_drop_ratio,
                                          drop_path_ratio, norm_layer, ffn_ratio, act_function, dropout_ratio)
        # Transformer Decoder
        self.decoder = TransformerDecoder(decoder_depths, embedding_dimensions, decoder_heads,
                                          qkv_bias, qk_scale, attention_drop_ratio, projection_drop_ratio,
                                          drop_path_ratio, norm_layer, ffn_ratio, act_function, dropout_ratio)
        # Token & Position Embedding init
        trunc_normal_(self.token, std=0.02)
        trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):  # B 3 H W
        # Patch Embedding & Position Embedding
        x = self.patch_embedding(x)  # B N C
        token = self.token.expand(x.shape[0], -1, -1)  # 1 1 C -> B 1 C
        x = torch.cat((token, x), dim=1)  # B N+1 C
        x = self.position_drop(x + self.position_embedding)  # keep B N+1 C
        # Transformer Encoder
        token, qkv = self.encoder(x)  # token: B C, qkv: B N C
        # Transformer Decoder
        x = self.decoder(token, qkv, qkv)  # B 1 C
        return x


class MultiLayerPerceptron(nn.Module):
    """
    Multi Layer Perceptron (MLP)
    Args:
        in_features (int): Number of input feature.
    """
    def __init__(self, in_features):
        super(MultiLayerPerceptron, self).__init__()

        # MLP
        self.mlp = nn.Sequential(
            # Linear + LReLu + Linear + LReLu + Linear
            nn.Linear(in_features, in_features // 4, bias=False),  # 2C -> 2C // 4
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features // 4, in_features // 16, bias=False),  # 2C // 4 -> 2C // 16
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features // 16, 1, bias=False),  # 2C // 16 -> 1
        )

    def forward(self, x):  # B 2C
        # MLP
        x = self.mlp(x)  # B 1
        return x


class ATUIQP(nn.Module):
    """
    ATUIQP (Attention and Transformer driven Underwater Image Quality Predictor)
    Args:
        input_size (tuple(int)): Size of input image. Default: (224, 224)
        in_channels (int): Number of input channels. Default: 3
        patch_size (int): Size of path. Default: 16
        embedding_channels (int): Number of embedding channels. Default: 24
        csam_number (int): Number of CSAMs. Default: 5
        embedding_dimensions (int): Number of embedding dimensions. Default: 384
        encoder_depths (int): Number of transformer encoder layers. Default: 12
        decoder_depths (int): Number of transformer decoder layers. Default: 1
        encoder_heads (int): Number of self attention heads in transformer encoder layer. Default: 6
        decoder_heads (int): Number of self attention heads in transformer decoder layer. Default: 6
        drop_path_ratio (float): Dropout path ratio. Default: 0.0
        dropout_ratio (float): Dropout ratio. Default: 0.0
    """
    def __init__(self,
                 input_size=(224, 224),
                 in_channels=3,
                 patch_size=16,
                 embedding_channels=24,
                 csam_number=5,
                 embedding_dimensions=384,
                 encoder_depths=12,
                 decoder_depths=1,
                 encoder_heads=6,
                 decoder_heads=6,
                 drop_path_ratio=0.0,  # pytorch doc default 0.1, not using it
                 dropout_ratio=0.0):  # pytorch doc default 0.1, not using it
        super(ATUIQP, self).__init__()

        # CSAMBranch (CSAM)
        self.csam_branch = CSAMBranch(input_size, in_channels, embedding_channels, nums=csam_number, map_channel=embedding_dimensions)
        # TransformerBranch (Transformer)
        self.transformer_branch = TransformerBranch(input_size, in_channels, patch_size, embedding_dimensions,
                                                    encoder_depths, decoder_depths, encoder_heads, decoder_heads,
                                                    drop_path_ratio=drop_path_ratio, dropout_ratio=dropout_ratio)
        # MultiLayerPerceptron (MLP)
        self.mlp = MultiLayerPerceptron(in_features=embedding_dimensions * 2)

        # Weight init
        # self.apply(_init_weights)

    def forward(self, x):  # B 3 H W
        # CSAMBranch (CSAM)
        csam_branch = self.csam_branch(x)  # B C
        # TransformerBranch (Transformer)
        transformer_branch = self.transformer_branch(x)  # B C
        # Concat & Flatten
        x = torch.cat([csam_branch, transformer_branch], dim=1)  # B 2C
        # MultiLayerPerceptron (MLP)
        x = self.mlp(x)  # B 1
        return x


if __name__ == '__main__':
    input = torch.rand([2, 3, 224, 224])
    net = ATUIQP()
    output = net(input)
    print(f"output: {output}")
    print(f"output.shape: {output.shape}")
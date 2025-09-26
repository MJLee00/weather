import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import VARIABLES, CNN_FEATURE_CHANNELS, LATENT_DIM, NUM_BASE_MODELS

# --- Helper: depthwise separable conv (efficient) ---
class SepConv2d(nn.Module):
    """Depthwise separable conv: depthwise conv then 1x1 pointwise conv.
       Reduces FLOPs/params vs standard conv while keeping expressivity.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        # depthwise
        self.dw = nn.Conv2d(in_ch, in_ch, kernel_size=kernel_size, stride=stride,
                            padding=padding, groups=in_ch, bias=bias)
        # pointwise
        self.pw = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
    
    def forward(self, x):
        x = self.dw(x)
        x = self.pw(x)
        x = self.bn(x)
        return x

class FeatureExtractor(nn.Module):
    """
    Efficient feature extractor for very large spatial inputs.
    Input expected: (B, in_channels, H, W), e.g. (B, len(VARIABLES), 721, 1440).
    - Uses a patchify-like stem (large stride) to aggressively reduce H/W early.
    - Then several separable conv stages with strides to further downsample.
    - Ends with AdaptiveAvgPool2d((1,1)) -> flattened feature vector of size out_channels.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels  # Store expected input channels
        # 1) Patchify stem: big kernel + stride to reduce spatial dimension quickly.
        #    e.g. kernel_size=7, stride=4 reduces 1440 -> 360 on first layer.
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 2) Efficient downsampling blocks (separable convs).
        #    Each block may reduce spatial dims (stride=2) or keep (stride=1).
        self.stage1 = nn.Sequential(
            SepConv2d(64, 128, kernel_size=3, stride=2, padding=1),   # downsample
            nn.ReLU(inplace=True)
        )
        self.stage2 = nn.Sequential(
            SepConv2d(128, 256, kernel_size=3, stride=2, padding=1),  # downsample
            nn.ReLU(inplace=True)
        )
        # final feature projection: reduce to out_channels
        self.project = nn.Sequential(
            SepConv2d(256, out_channels, kernel_size=3, stride=2, padding=1),  # more downsample
            nn.ReLU(inplace=True)
        )

        # global spatial pooling to (1,1) to produce fixed-size vector
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        """
        x: Input tensor, expected shape (B, C=len(VARIABLES), H=721, W=1440).
           Handles inputs with extra dimensions by squeezing, e.g., (B, 1, 1, C, H, W).
        returns: flattened vector (B, out_channels)
        """
        # Check input dimensions and squeeze extra dimensions if present
        if x.dim() > 4:
            # Squeeze extra dimensions (e.g., [1, 1, 1, len(VARIABLES), 721, 1440] -> [1, len(VARIABLES), 721, 1440])
            x = x.squeeze()  # Remove dimensions of size 1
            # Ensure we have at least 4D after squeezing
            while x.dim() < 4:
                x = x.unsqueeze(0)  # Add batch dimension if needed

        # Validate input shape
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B, C, H, W), got shape {x.shape}")
        if x.size(1) != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input channels, got {x.size(1)} channels")

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.project(x)
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        return x  # shape (B, out_channels)

def preprocess_input(x):
    """
    Preprocess the input tensor from (batch, hour, day, level, lati, longi) to (batch, level, lati, longi),
    assuming hour and day are singleton dimensions (e.g., 1) that can be squeezed.
    Validates that the level dimension matches len(VARIABLES).
    """
    if x.dim() != 6:
        raise ValueError(f"Expected 6D input (batch, hour, day, level, lati, longi), got shape {x.shape}")
    
    # Validate that the level dimension matches len(VARIABLES)
    if x.size(3) != len(VARIABLES):
        raise ValueError(f"Expected level dimension to match len(VARIABLES)={len(VARIABLES)}, got {x.size(3)}")

    # Squeeze hour (dim=1) and day (dim=2), assuming they are 1
    x = x.squeeze(1).squeeze(1)  # Result: (batch, level, lati, longi)
    
    return x

# ---- Actor / Critic using FeatureExtractor ----
class Actor(nn.Module):
    """Actor network: outputs softmax weights across base models."""
    def __init__(self, in_channels=len(VARIABLES)):  # Use len(VARIABLES) as input channels
        super(Actor, self).__init__()
        # Ensure CNN_FEATURE_CHANNELS matches 'out_channels' used in FeatureExtractor
        self.feature_extractor = FeatureExtractor(in_channels, CNN_FEATURE_CHANNELS)

        # 'feature_dim' is exactly CNN_FEATURE_CHANNELS after the extractor
        feature_dim = CNN_FEATURE_CHANNELS

        # state representation concatenates weather feature vector and model_mses (NUM_BASE_MODELS)
        state_dim = feature_dim + NUM_BASE_MODELS

        # LayerNorm expects normalized_shape equal to the last dimension of the vector
        self.ln1 = nn.LayerNorm(state_dim)
        self.fc1 = nn.Linear(state_dim, LATENT_DIM)
        self.ln2 = nn.LayerNorm(LATENT_DIM)
        self.fc2 = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.ln3 = nn.LayerNorm(LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM, NUM_BASE_MODELS)

    def forward(self, weather_data, model_mses):
        """
        weather_data: (B, hour, day, level, H=721, W=1440)
        model_mses: (B, NUM_BASE_MODELS)
        """
        weather_data = preprocess_input(weather_data)  # Preprocess to (B, len(VARIABLES), H, W)
        weather_features = self.feature_extractor(weather_data)  # (B, CNN_FEATURE_CHANNELS)
        # Concatenate along feature dimension
        state_representation = torch.cat([weather_features, model_mses], dim=1)  # (B, state_dim)

        x = F.relu(self.fc1(self.ln1(state_representation)))
        x = F.relu(self.fc2(self.ln2(x)))
        action = F.softmax(self.fc3(self.ln3(x)), dim=1)
        return action


class Critic(nn.Module):
    """Critic: estimates Q-value for state-action pair."""
    def __init__(self, in_channels=len(VARIABLES)):  # Use len(VARIABLES) as input channels
        super(Critic, self).__init__()
        self.feature_extractor = FeatureExtractor(in_channels, CNN_FEATURE_CHANNELS)

        feature_dim = CNN_FEATURE_CHANNELS
        # state + action dim
        state_action_dim = feature_dim + NUM_BASE_MODELS + NUM_BASE_MODELS

        self.ln1 = nn.LayerNorm(state_action_dim)
        self.fc1 = nn.Linear(state_action_dim, LATENT_DIM)
        self.ln2 = nn.LayerNorm(LATENT_DIM)
        self.fc2 = nn.Linear(LATENT_DIM, LATENT_DIM)
        self.ln3 = nn.LayerNorm(LATENT_DIM)
        self.fc3 = nn.Linear(LATENT_DIM, 1)

    def forward(self, weather_data, model_mses, action):
        """
        weather_data: (B, hour, day, level, H=721, W=1440)
        model_mses: (B, NUM_BASE_MODELS)
        action: (B, NUM_BASE_MODELS)
        """
        weather_data = preprocess_input(weather_data)  # Preprocess to (B, len(VARIABLES), H, W)
        weather_features = self.feature_extractor(weather_data)  # (B, feature_dim)
        state_action = torch.cat([weather_features, model_mses, action], dim=1)
        x = F.relu(self.fc1(self.ln1(state_action)))
        x = F.relu(self.fc2(self.ln2(x)))
        q_value = self.fc3(self.ln3(x))
        return q_value
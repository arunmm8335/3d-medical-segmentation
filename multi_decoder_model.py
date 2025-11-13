import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock3D(nn.Module):
    """3D Convolutional block with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class ResidualBlock3D(nn.Module):
    """3D Residual block"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.relu(out)
        return out

class Encoder3D(nn.Module):
    """Shared 3D Encoder with Residual connections"""
    def __init__(self, in_channels=1):
        super().__init__()
        # 1. 16 -> 8
        self.enc1 = ConvBlock3D(in_channels, 8) 
        self.pool1 = nn.MaxPool3d(2)
        self.res1 = ResidualBlock3D(8)          
        
        # 2. 16, 32 -> 8, 16
        self.enc2 = ConvBlock3D(8, 16)          
        self.pool2 = nn.MaxPool3d(2)
        self.res2 = ResidualBlock3D(16)          
        
        # 3. 32, 64 -> 16, 32
        self.enc3 = ConvBlock3D(16, 32)          
        self.pool3 = nn.MaxPool3d(2)
        self.res3 = ResidualBlock3D(32)          
        
        # 4. 64, 128 -> 32, 64
        self.enc4 = ConvBlock3D(32, 64)          
        self.pool4 = nn.MaxPool3d(2)
        self.res4 = ResidualBlock3D(64)          
        
        # 5. 128, 256 -> 64, 128
        self.bottleneck = ConvBlock3D(64, 128)  
    
    def forward(self, x):
        # Encoder path with skip connections
        e1 = self.res1(self.enc1(x))
        p1 = self.pool1(e1)
        
        e2 = self.res2(self.enc2(p1))
        p2 = self.pool2(e2)
        
        e3 = self.res3(self.enc3(p2))
        p3 = self.pool3(e3)
        
        e4 = self.res4(self.enc4(p3))
        p4 = self.pool4(e4)
        
        bottleneck = self.bottleneck(p4)
        
        return bottleneck, [e1, e2, e3, e4]

class Decoder3D(nn.Module):
    """Individual decoder for each organ"""
    def __init__(self, name="organ"):
        super().__init__()
        self.name = name
        
        # Decoder path
        # 1. 256, 128 -> 128, 64
        self.up4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2) 
        self.dec4 = ConvBlock3D(128, 64) # (64 from up4 + 64 from e4)                               
        
        # 2. 128, 64 -> 64, 32
        self.up3 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)  
        self.dec3 = ConvBlock3D(64, 32) # (32 from up3 + 32 from e3)
        
        # 3. 64, 32 -> 32, 16
        self.up2 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)   
        self.dec2 = ConvBlock3D(32, 16) # (16 from up2 + 16 from e2)
        
        # 4. 32, 16 -> 16, 8
        self.up1 = nn.ConvTranspose3d(16, 8, kernel_size=2, stride=2)   
        self.dec1 = ConvBlock3D(16, 8) # (8 from up1 + 8 from e1)
    
    def forward(self, bottleneck, skip_connections):
        e1, e2, e3, e4 = skip_connections
        
        # Decoder path with skip connections
        d4 = self.up4(bottleneck)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        return d1

class CrossDecoderAttention(nn.Module):
    """Multi-head attention between decoders"""
    def __init__(self, channels=32, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.channels = channels
        
        # Multi-head attention components
        self.query = nn.Conv3d(channels, channels, kernel_size=1)
        self.key = nn.Conv3d(channels, channels, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv3d(channels, channels, kernel_size=1)
        
    def forward(self, decoders_outputs):
        """
        decoders_outputs: list of [B, C, D, H, W] tensors
        """
        B, C, D, H, W = decoders_outputs[0].shape
        
        # Stack decoder outputs
        stacked = torch.stack(decoders_outputs, dim=1)  # [B, num_decoders, C, D, H, W]
        num_decoders = stacked.shape[1]
        
        # Reshape for attention
        stacked = stacked.view(B * num_decoders, C, D, H, W)
        
        # Compute Q, K, V
        Q = self.query(stacked)  # [B*num_decoders, C, D, H, W]
        K = self.key(stacked)
        V = self.value(stacked)
        
        # Reshape for multi-head attention
        Q = Q.view(B, num_decoders, self.num_heads, C // self.num_heads, D * H * W)
        K = K.view(B, num_decoders, self.num_heads, C // self.num_heads, D * H * W)
        V = V.view(B, num_decoders, self.num_heads, C // self.num_heads, D * H * W)
        
        # Attention scores
        attention_scores = torch.einsum('bnhcl,bmhcl->bnmh', Q, K) / (C // self.num_heads) ** 0.5
        attention_weights = F.softmax(attention_scores, dim=2)
        
        # Apply attention
        attended = torch.einsum('bnmh,bmhcl->bnhcl', attention_weights, V)
        attended = attended.contiguous().view(B * num_decoders, C, D, H, W)
        
        # Output projection
        output = self.out_proj(attended)
        output = output.view(B, num_decoders, C, D, H, W)
        
        # Split back to individual decoders
        outputs = [output[:, i] for i in range(num_decoders)]
        
        return outputs

class MultiDecoderUNet3D(nn.Module):
    """
    Multi-Decoder 3D U-Net for Prostate, Bladder, and Rectum segmentation
    Based on the provided architecture diagram
    """
    def __init__(self, in_channels=1, num_classes_per_organ=1):
        super().__init__()
        
        # Shared encoder
        self.encoder = Encoder3D(in_channels)
        
        # Individual decoders for each organ
        self.prostate_decoder = Decoder3D(name="prostate")
        self.bladder_decoder = Decoder3D(name="bladder")
        self.rectum_decoder = Decoder3D(name="rectum")
        
        # Cross-decoder attention
        self.cross_attention = CrossDecoderAttention(channels=8, num_heads=4) # 16 -> 8
        
        # Final segmentation heads (Sigmoid for binary)
        self.prostate_head = nn.Conv3d(8, num_classes_per_organ, kernel_size=1) # 16 -> 8
        self.bladder_head = nn.Conv3d(8, num_classes_per_organ, kernel_size=1)  # 16 -> 8
        self.rectum_head = nn.Conv3d(8, num_classes_per_organ, kernel_size=1)   # 16 -> 8
    
    def forward(self, x):
        # Shared encoder
        bottleneck, skip_connections = self.encoder(x)
        
        # Individual decoders
        prostate_features = self.prostate_decoder(bottleneck, skip_connections)
        bladder_features = self.bladder_decoder(bottleneck, skip_connections)
        rectum_features = self.rectum_decoder(bottleneck, skip_connections)
        
        # Cross-decoder attention
        decoder_outputs = [prostate_features, bladder_features, rectum_features]
        attended_outputs = self.cross_attention(decoder_outputs)
        
        # Final segmentation masks
        prostate_mask = torch.sigmoid(self.prostate_head(attended_outputs[0]))
        bladder_mask = torch.sigmoid(self.bladder_head(attended_outputs[1]))
        rectum_mask = torch.sigmoid(self.rectum_head(attended_outputs[2]))
        
        # Stack masks along channel dimension
        output = torch.cat([prostate_mask, bladder_mask, rectum_mask], dim=1)
        
        return output
    
    def get_parameter_count(self):
        """Count total parameters"""
        return sum(p.numel() for p in self.parameters())

# Test the model
if __name__ == "__main__":
    model = MultiDecoderUNet3D(in_channels=1, num_classes_per_organ=1)
    print(f"Total parameters: {model.get_parameter_count():,}")
    
    # Test forward pass
    x = torch.randn(1, 1, 64, 64, 64)  # [B, C, D, H, W]
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")  # Should be [1, 3, 64, 64, 64]

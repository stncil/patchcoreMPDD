import FrEIA.framework as Ff
import FrEIA.modules as Fm
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from patchcore.fastflow import constants as const


class PatchMaker:
    """Patch extraction and reconstruction class similar to PatchCore."""
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride if stride is not None else patchsize

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            features: [torch.Tensor, bs x c x w x h]
        Returns:
            patches: [torch.Tensor, bs * w//stride * h//stride, c, patchsize, patchsize]
            spatial_info: [tuple] (num_patches_h, num_patches_w) if return_spatial_info=True
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        
        # Calculate number of patches in each dimension
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        
        # Reshape to patch format
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, patch_scores, batchsize):
        """Reshape patch scores back to batch format."""
        return patch_scores.reshape(batchsize, -1, *patch_scores.shape[1:])

    def score_aggregation(self, patch_scores, method="max"):
        """Aggregate patch scores to image-level scores."""
        was_numpy = False
        if isinstance(patch_scores, np.ndarray):
            was_numpy = True
            patch_scores = torch.from_numpy(patch_scores)
        
        if method == "max":
            while patch_scores.ndim > 1:
                patch_scores = torch.max(patch_scores, dim=-1).values
        elif method == "mean":
            while patch_scores.ndim > 1:
                patch_scores = torch.mean(patch_scores, dim=-1)
        
        if was_numpy:
            return patch_scores.numpy()
        return patch_scores

    def reconstruct_anomaly_map(self, patch_scores, patch_shapes, target_size):
        """Reconstruct spatial anomaly map from patch scores."""
        batchsize = patch_scores.shape[0]
        num_patches_h, num_patches_w = patch_shapes
        
        # Reshape patch scores to spatial grid
        spatial_scores = patch_scores.reshape(batchsize, num_patches_h, num_patches_w)
        
        # Interpolate to target size
        spatial_scores = spatial_scores.unsqueeze(1)  # Add channel dimension
        anomaly_map = F.interpolate(
            spatial_scores, size=target_size, mode='bilinear', align_corners=False
        )
        
        return anomaly_map.squeeze(1)  # Remove channel dimension


def subnet_conv_func(kernel_size, hidden_ratio):
    def subnet_conv(in_channels, out_channels):
        hidden_channels = int(in_channels * hidden_ratio)
        return nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size, padding="same"),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size, padding="same"),
        )

    return subnet_conv


def nf_fast_flow(input_chw, conv3x3_only, hidden_ratio, flow_steps, clamp=2.0):
    nodes = Ff.SequenceINN(*input_chw)
    for i in range(flow_steps):
        if i % 2 == 1 and not conv3x3_only:
            kernel_size = 1
        else:
            kernel_size = 3
        nodes.append(
            Fm.AllInOneBlock,
            subnet_constructor=subnet_conv_func(kernel_size, hidden_ratio),
            affine_clamping=clamp,
            permute_soft=False,
        )
    return nodes


class FastFlow(nn.Module):
    def __init__(
        self,
        backbone_name,
        flow_steps,
        input_size,
        conv3x3_only=False,
        hidden_ratio=1.0,
        enable_patch_processing=False,
        patchsize=3,
        patchstride=1,
    ):
        super(FastFlow, self).__init__()
        assert (
            backbone_name in const.SUPPORTED_BACKBONES
        ), "backbone_name must be one of {}".format(const.SUPPORTED_BACKBONES)

        # Patch processing setup
        self.enable_patch_processing = enable_patch_processing
        if enable_patch_processing:
            self.patch_maker = PatchMaker(patchsize, stride=patchstride)

        if backbone_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
            self.feature_extractor = timm.create_model(backbone_name, pretrained=True)
            channels = [768]
            scales = [16]
        else:
            self.feature_extractor = timm.create_model(
                backbone_name,
                pretrained=True,
                features_only=True,
                out_indices=[1, 2, 3],
            )
            channels = self.feature_extractor.feature_info.channels()
            scales = self.feature_extractor.feature_info.reduction()

            # Use channel-only LayerNorm for arbitrary spatial size
            self.norms = nn.ModuleList()
            for in_channels, scale in zip(channels, scales):
                self.norms.append(
                    nn.LayerNorm([in_channels], elementwise_affine=True)
                )

        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Unfreeze last block for ResNet-like backbones
        if hasattr(self.feature_extractor, 'layer4'):
            for param in self.feature_extractor.layer4.parameters():
                param.requires_grad = True

        self.nf_flows = nn.ModuleList()
        for in_channels, scale in zip(channels, scales):
            if enable_patch_processing:
                # For patch processing, flows operate on patch dimensions
                flow_input_size = patchsize
            else:
                # For dense processing, flows operate on feature map dimensions
                flow_input_size = int(input_size / scale)
            
            self.nf_flows.append(
                nf_fast_flow(
                    [in_channels, flow_input_size, flow_input_size],
                    conv3x3_only=conv3x3_only,
                    hidden_ratio=hidden_ratio,
                    flow_steps=flow_steps,
                )
            )
        self.input_size = input_size

    def _extract_features(self, x):
        """Extract features from backbone."""
        self.feature_extractor.eval()
        if isinstance(
            self.feature_extractor, timm.models.vision_transformer.VisionTransformer
        ):
            x = self.feature_extractor.patch_embed(x)
            cls_token = self.feature_extractor.cls_token.expand(x.shape[0], -1, -1)
            if self.feature_extractor.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat(
                    (
                        cls_token,
                        self.feature_extractor.dist_token.expand(x.shape[0], -1, -1),
                        x,
                    ),
                    dim=1,
                )
            x = self.feature_extractor.pos_drop(x + self.feature_extractor.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = self.feature_extractor.blocks[i](x)
            x = self.feature_extractor.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        elif isinstance(self.feature_extractor, timm.models.cait.Cait):
            x = self.feature_extractor.patch_embed(x)
            x = x + self.feature_extractor.pos_embed
            x = self.feature_extractor.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = self.feature_extractor.blocks[i](x)
            N, _, C = x.shape
            x = self.feature_extractor.norm(x)
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, self.input_size // 16, self.input_size // 16)
            features = [x]
        else:
            features = self.feature_extractor(x)
            # Channel-only LayerNorm for arbitrary spatial size
            for i, feature in enumerate(features):
                # feature: [B, C, H, W] -> [B, H, W, C]
                feature = feature.permute(0, 2, 3, 1)
                feature = self.norms[i](feature)
                feature = feature.permute(0, 3, 1, 2)
                features[i] = feature

        return features

    def _process_patches(self, features):
        """Process features through patch-based workflow."""
        patch_features_list = []
        patch_shapes_list = []
        
        # Extract patches from each feature scale
        for feature in features:
            patches, patch_shapes = self.patch_maker.patchify(feature, return_spatial_info=True)
            patch_features_list.append(patches)
            patch_shapes_list.append(patch_shapes)
        
        # Process patches through normalizing flows
        loss = 0
        patch_outputs = []
        
        for i, patches in enumerate(patch_features_list):
            # Reshape patches for flow processing: [B*N_patches, C, H, W]
            B, N_patches, C, H, W = patches.shape
            patches_reshaped = patches.reshape(B * N_patches, C, H, W)
            
            # Process through normalizing flow
            output, log_jac_dets = self.nf_flows[i](patches_reshaped)
            
            # Compute loss for training
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            
            # Reshape back to patch format: [B, N_patches, ...]
            output = output.reshape(B, N_patches, *output.shape[1:])
            patch_outputs.append(output)
        
        return loss, patch_outputs, patch_shapes_list

    def _process_dense(self, features):
        """Process features through dense (non-patch) workflow."""
        loss = 0
        outputs = []
        
        for i, feature in enumerate(features):
            output, log_jac_dets = self.nf_flows[i](feature)
            loss += torch.mean(
                0.5 * torch.sum(output**2, dim=(1, 2, 3)) - log_jac_dets
            )
            outputs.append(output)
        
        return loss, outputs

    def forward(self, batch):
        # Accept either a dict (from dataset) or a tensor
        if isinstance(batch, dict):
            x = batch["image"]
        else:
            x = batch
        
        # Extract features from backbone
        features = self._extract_features(x)
        
        if self.enable_patch_processing:
            loss, patch_outputs, patch_shapes_list = self._process_patches(features)
            ret = {"loss": loss}
            
            if not self.training:
                # Generate anomaly maps from patch outputs
                anomaly_maps_list = []
                
                for i, (patch_output, patch_shapes) in enumerate(zip(patch_outputs, patch_shapes_list)):
                    # Compute patch-level anomaly scores
                    patch_log_prob = -torch.mean(patch_output**2, dim=(2, 3, 4)) * 0.5
                    patch_scores = -patch_log_prob  # Convert to anomaly scores (higher = more anomalous)
                    
                    # Reconstruct spatial anomaly map
                    anomaly_map = self.patch_maker.reconstruct_anomaly_map(
                        patch_scores, patch_shapes, (self.input_size, self.input_size)
                    )
                    anomaly_maps_list.append(anomaly_map.unsqueeze(1))  # Add channel dim
                
                # Combine multi-scale anomaly maps
                if len(anomaly_maps_list) > 1:
                    # Weight different scales differently
                    weights = torch.softmax(torch.tensor([2.0, 1.5, 1.0], device=x.device), dim=0)
                    weights = weights[:len(anomaly_maps_list)]
                    
                    # Weighted combination
                    combined_map = torch.zeros_like(anomaly_maps_list[0])
                    for i, (amap, weight) in enumerate(zip(anomaly_maps_list, weights)):
                        combined_map += amap * weight
                    
                    ret["anomaly_map"] = combined_map.squeeze(1)  # Remove channel dim
                else:
                    ret["anomaly_map"] = anomaly_maps_list[0].squeeze(1)
            
            return ret
        
        else:
            # Dense processing (original FastFlow behavior)
            loss, outputs = self._process_dense(features)
            ret = {"loss": loss}

            if not self.training:
                anomaly_map_list = []
                for i, output in enumerate(outputs):
                    # Improved anomaly score calculation
                    log_prob = -torch.mean(output**2, dim=1, keepdim=True) * 0.5
                    prob = torch.exp(log_prob)
                    
                    # Apply Gaussian smoothing for better localization
                    prob_smooth = F.avg_pool2d(prob, kernel_size=3, stride=1, padding=1)
                    
                    # Weight different scales differently (higher resolution = more weight)
                    scale_weight = 2.0 ** i  # Give more weight to higher resolution features
                    
                    a_map = F.interpolate(
                        (-log_prob) * scale_weight,
                        size=[self.input_size, self.input_size],
                        mode="bilinear",
                        align_corners=False,
                    )
                    anomaly_map_list.append(a_map)
                
                # Weighted combination instead of simple mean
                anomaly_map_list = torch.stack(anomaly_map_list, dim=-1)
                weights = torch.softmax(torch.tensor([2.0, 1.5, 1.0], device=anomaly_map_list.device), dim=0)
                if len(anomaly_map_list.shape) > 4:
                    weights = weights[:anomaly_map_list.shape[-1]]
                    anomaly_map = torch.sum(anomaly_map_list * weights.view(1, 1, 1, 1, -1), dim=-1)
                else:
                    anomaly_map = torch.mean(anomaly_map_list, dim=-1)
                
                ret["anomaly_map"] = anomaly_map
            return ret

    def training_step(self, batch):
        x = batch["image"] if isinstance(batch, dict) else batch
        out = self.forward(x)
        return out["loss"]

    def predict(self, batch):
        x = batch["image"] if isinstance(batch, dict) else batch
        self.eval()
        with torch.no_grad():
            out = self.forward(x)
        return out["anomaly_map"]

    def fit(self, training_data):
        """Training method similar to PatchCore's fit method."""
        self.train()
        # Training loop would be implemented here
        # This is a placeholder to match PatchCore's interface
        pass

    def predict_dataloader(self, dataloader):
        """Predict on entire dataloader similar to PatchCore."""
        self.eval()
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, dict):
                    labels_gt.extend(batch["is_anomaly"].numpy().tolist())
                    masks_gt.extend(batch["mask"].numpy().tolist())
                    images = batch["image"]
                else:
                    images = batch
                
                anomaly_maps = self.predict(images)
                
                # Compute image-level scores
                if self.enable_patch_processing:
                    # For patch processing, use max aggregation similar to PatchCore
                    image_scores = torch.max(anomaly_maps.view(anomaly_maps.shape[0], -1), dim=1)[0]
                else:
                    # For dense processing, use mean aggregation
                    image_scores = torch.mean(anomaly_maps.view(anomaly_maps.shape[0], -1), dim=1)
                
                for score, mask in zip(image_scores, anomaly_maps):
                    scores.append(score.cpu().numpy())
                    masks.append(mask.cpu().numpy())
        
        return scores, masks, labels_gt, masks_gt
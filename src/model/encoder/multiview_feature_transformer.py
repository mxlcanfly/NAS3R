import math

import torch
import torch.nn as nn
from einops import rearrange


class PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, x):
        b, _, h, w = x.size()
        mask = torch.ones((b, h, w), device=x.device)
        y_embed = mask.cumsum(1, dtype=torch.float32)
        x_embed = mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


def split_feature(feature, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0
        feature = feature.view(
            b,
            num_splits,
            h // num_splits,
            num_splits,
            w // num_splits,
            c,
        )
        return feature.permute(0, 1, 3, 2, 4, 5).reshape(
            b * num_splits * num_splits,
            h // num_splits,
            w // num_splits,
            c,
        )

    b, c, h, w = feature.size()
    assert h % num_splits == 0 and w % num_splits == 0
    feature = feature.view(
        b,
        c,
        num_splits,
        h // num_splits,
        num_splits,
        w // num_splits,
    )
    return feature.permute(0, 2, 4, 1, 3, 5).reshape(
        b * num_splits * num_splits,
        c,
        h // num_splits,
        w // num_splits,
    )


def merge_splits(splits, num_splits=2, channel_last=False):
    if channel_last:
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits
        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        return splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b,
            num_splits * h,
            num_splits * w,
            c,
        )

    b, c, h, w = splits.size()
    new_b = b // num_splits // num_splits
    splits = splits.view(new_b, num_splits, num_splits, c, h, w)
    return splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
        new_b,
        c,
        num_splits * h,
        num_splits * w,
    )


def feature_add_position_list(features_list, attn_splits, feature_channels):
    pos_enc = PositionEmbeddingSine(num_pos_feats=feature_channels // 2)
    if attn_splits > 1:
        features_splits = [
            split_feature(x, num_splits=attn_splits) for x in features_list
        ]
        position = pos_enc(features_splits[0])
        features_splits = [x + position for x in features_splits]
        return [merge_splits(x, num_splits=attn_splits) for x in features_splits]

    position = pos_enc(features_list[0])
    return [x + position for x in features_list]


def generate_shift_window_attn_mask(
    input_resolution,
    window_size_h,
    window_size_w,
    shift_size_h,
    shift_size_w,
    device,
):
    h, w = input_resolution
    img_mask = torch.zeros((1, h, w, 1), device=device)
    h_slices = (
        slice(0, -window_size_h),
        slice(-window_size_h, -shift_size_h),
        slice(-shift_size_h, None),
    )
    w_slices = (
        slice(0, -window_size_w),
        slice(-window_size_w, -shift_size_w),
        slice(-shift_size_w, None),
    )
    cnt = 0
    for h_slice in h_slices:
        for w_slice in w_slices:
            img_mask[:, h_slice, w_slice, :] = cnt
            cnt += 1

    mask_windows = split_feature(
        img_mask,
        num_splits=input_resolution[-1] // window_size_w,
        channel_last=True,
    )
    mask_windows = mask_windows.view(-1, window_size_h * window_size_w)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    return attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(
        attn_mask == 0,
        0.0,
    )


def split_window_attention(q, k, v, num_splits, with_shift, h, w, attn_mask):
    b, _, c = q.size()
    b_new = b * num_splits * num_splits
    window_size_h = h // num_splits
    window_size_w = w // num_splits
    scale_factor = c**0.5

    if k.dim() == 4:
        m = k.size(1)
        q = q.view(b, h, w, c)
        k = k.view(b, m, h, w, c)
        v = v.view(b, m, h, w, c)
        if with_shift:
            shift_size_h = window_size_h // 2
            shift_size_w = window_size_w // 2
            q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
            k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))
            v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(2, 3))

        q = split_feature(q, num_splits=num_splits, channel_last=True)
        k = split_feature(
            k.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )
        v = split_feature(
            v.permute(0, 2, 3, 4, 1).reshape(b, h, w, -1),
            num_splits=num_splits,
            channel_last=True,
        )
        k = k.view(b_new, h // num_splits, w // num_splits, c, m)
        k = k.permute(0, 3, 1, 2, 4).reshape(b_new, c, -1)
        v = v.view(b_new, h // num_splits, w // num_splits, c, m)
        v = v.permute(0, 1, 2, 4, 3).reshape(b_new, -1, c)

        scores = torch.matmul(q.view(b_new, -1, c), k) / scale_factor
        if with_shift:
            scores = scores + attn_mask.repeat(b, 1, m)
        out = torch.matmul(torch.softmax(scores, dim=-1), v)
        out = merge_splits(
            out.view(b_new, h // num_splits, w // num_splits, c),
            num_splits=num_splits,
            channel_last=True,
        )
        if with_shift:
            out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))
        return out.view(b, -1, c)

    q = q.view(b, h, w, c)
    k = k.view(b, h, w, c)
    v = v.view(b, h, w, c)
    if with_shift:
        shift_size_h = window_size_h // 2
        shift_size_w = window_size_w // 2
        q = torch.roll(q, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        k = torch.roll(k, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))
        v = torch.roll(v, shifts=(-shift_size_h, -shift_size_w), dims=(1, 2))

    q = split_feature(q, num_splits=num_splits, channel_last=True)
    k = split_feature(k, num_splits=num_splits, channel_last=True)
    v = split_feature(v, num_splits=num_splits, channel_last=True)
    scores = torch.matmul(
        q.view(b_new, -1, c),
        k.view(b_new, -1, c).permute(0, 2, 1),
    ) / scale_factor
    if with_shift:
        scores = scores + attn_mask.repeat(b, 1, 1)
    out = torch.matmul(torch.softmax(scores, dim=-1), v.view(b_new, -1, c))
    out = merge_splits(
        out.view(b_new, h // num_splits, w // num_splits, c),
        num_splits=num_splits,
        channel_last=True,
    )
    if with_shift:
        out = torch.roll(out, shifts=(shift_size_h, shift_size_w), dims=(1, 2))
    return out.view(b, -1, c)


class TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model=128,
        attention_type="swin",
        no_ffn=False,
        ffn_dim_expansion=4,
        with_shift=False,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.no_ffn = no_ffn
        self.with_shift = with_shift
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.merge = nn.Linear(d_model, d_model, bias=False)
        self.norm1 = nn.LayerNorm(d_model)
        if not self.no_ffn:
            in_channels = d_model * 2
            self.mlp = nn.Sequential(
                nn.Linear(in_channels, in_channels * ffn_dim_expansion, bias=False),
                nn.GELU(),
                nn.Linear(in_channels * ffn_dim_expansion, d_model, bias=False),
            )
            self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self,
        source,
        target,
        height,
        width,
        shifted_window_attn_mask,
        attn_num_splits,
    ):
        query = self.q_proj(source)
        key = self.k_proj(target)
        value = self.v_proj(target)
        if self.attention_type == "swin" and attn_num_splits > 1:
            message = split_window_attention(
                query,
                key,
                value,
                num_splits=attn_num_splits,
                with_shift=self.with_shift,
                h=height,
                w=width,
                attn_mask=shifted_window_attn_mask,
            )
        else:
            message = torch.matmul(
                torch.softmax(
                    torch.matmul(query, key.permute(0, 2, 1)) / (query.size(2) ** 0.5),
                    dim=2,
                ),
                value,
            )
        message = self.norm1(self.merge(message))
        if not self.no_ffn:
            message = self.norm2(self.mlp(torch.cat([source, message], dim=-1)))
        return source + message


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model=128,
        attention_type="swin",
        ffn_dim_expansion=4,
        with_shift=False,
    ):
        super().__init__()
        self.self_attn = TransformerLayer(
            d_model=d_model,
            attention_type=attention_type,
            no_ffn=True,
            ffn_dim_expansion=ffn_dim_expansion,
            with_shift=with_shift,
        )
        self.cross_attn_ffn = TransformerLayer(
            d_model=d_model,
            attention_type=attention_type,
            ffn_dim_expansion=ffn_dim_expansion,
            with_shift=with_shift,
        )

    def forward(self, source, target, height, width, shifted_window_attn_mask, attn_num_splits):
        source = self.self_attn(
            source,
            source,
            height,
            width,
            shifted_window_attn_mask,
            attn_num_splits,
        )
        return self.cross_attn_ffn(
            source,
            target,
            height,
            width,
            shifted_window_attn_mask,
            attn_num_splits,
        )


def batch_features(features):
    q = []
    kv = []
    for i in range(len(features)):
        x = features.copy()
        q.append(x.pop(i))
        kv.append(torch.stack(x, dim=1))
    return torch.cat(q, dim=0), torch.cat(kv, dim=0)


class MultiViewFeatureTransformer(nn.Module):
    def __init__(
        self,
        num_layers=6,
        d_model=128,
        attention_type="swin",
        ffn_dim_expansion=4,
    ):
        super().__init__()
        self.attention_type = attention_type
        self.d_model = d_model
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    attention_type=attention_type,
                    ffn_dim_expansion=ffn_dim_expansion,
                    with_shift=(attention_type == "swin" and i % 2 == 1),
                )
                for i in range(num_layers)
            ]
        )
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, multi_view_features, attn_num_splits=2):
        b, c, h, w = multi_view_features[0].shape
        assert self.d_model == c
        num_views = len(multi_view_features)
        if self.attention_type == "swin" and attn_num_splits > 1:
            shifted_window_attn_mask = generate_shift_window_attn_mask(
                input_resolution=(h, w),
                window_size_h=h // attn_num_splits,
                window_size_w=w // attn_num_splits,
                shift_size_h=(h // attn_num_splits) // 2,
                shift_size_w=(w // attn_num_splits) // 2,
                device=multi_view_features[0].device,
            )
        else:
            shifted_window_attn_mask = None

        concat0, concat1 = batch_features(multi_view_features)
        concat0 = concat0.reshape(num_views * b, c, -1).permute(0, 2, 1)
        concat1 = concat1.reshape(num_views * b, num_views - 1, c, -1).permute(
            0,
            1,
            3,
            2,
        )
        for i, layer in enumerate(self.layers):
            concat0 = layer(
                concat0,
                concat1,
                h,
                w,
                shifted_window_attn_mask,
                attn_num_splits,
            )
            if i < len(self.layers) - 1:
                features = list(concat0.chunk(chunks=num_views, dim=0))
                concat0, concat1 = batch_features(features)

        features = concat0.chunk(chunks=num_views, dim=0)
        return [
            f.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            for f in features
        ]

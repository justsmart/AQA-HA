import torch
import torch.nn as nn
import torch.nn.functional as F


def _safe_dims_from_base(d_base, divisors):
    # 基于 d_base 的金字塔切片，至少为 1
    dims = [max(1, int(round(d_base / k))) for k in divisors]
    return dims

class Layer(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layer(x)

class SharedStem(nn.Module):
    """
    共享前端: x[B,D_v] -> h[B,D_v]
    Linear -> GELU -> Dropout -> Linear -> LayerNorm -> GELU
    """
    def __init__(self, d_in, d_hidden=None, dropout=0.1):
        super().__init__()
        assert 0.0 <= float(dropout) <= 1.0
        if d_hidden is None:
            d_hidden = max(d_in, int(2.0 * d_in))
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_in),
            nn.LayerNorm(d_in),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class ChannelHead(nn.Module):
    """
    单通道变分头：h[B,D_v] -> mu/logvar [B, d_k]
    """
    def __init__(self, d_in, d_k, dropout=0.1):
        super().__init__()
        assert 0.0 <= float(dropout) <= 1.0
        self.mlp = nn.Sequential(
            nn.Linear(d_in, d_k),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.mu = nn.Linear(d_k, d_k)
        self.logvar = nn.Linear(d_k, d_k)

    def forward(self, x):
        z = self.mlp(x)
        return self.mu(z), self.logvar(z)


class MultiChannelVAEEncoder(nn.Module):
    """
    多通道 VAE 编码器（修正版）

    约定：
      - 先于本模块外部计算好 d_emb_new，再作为参数传入；
      - channel_scheme:
          - "pyramid": 基于 d_emb 而非 D_v 进行通道划分
                        例：n_channels=4 => d_ks = [d_emb, d_emb/2, d_emb/4, d_emb/8]（取整>=1）
                        拼接后维度 sum(d_ks) = d_emb_new = d_emb * (1 + 1/2 + 1/4 + 1/8) = 1.875 d_emb
          - "equal":   将 d_emb 按通道均分，拼接后仍为 d_emb_new = d_emb（若你想 equal 也扩展，可把 d_emb_new 设为 n_channels*d_emb）
      - 本实现统一将各视图拼接结果线性投影到 d_emb_new，以保证 [B,V,d_emb_new] 一致。
    """
    def __init__(
        self,
        d_v,                 # 各视图输入维度
        d_emb,                  # 基础维度（用于通道切片的基准）
        d_emb_new,              # 最终输出统一嵌入维度（由外部先确定）
        dropout=0.1,
        hidden_ratio=2.0,
        n_channels=4,
        channel_scheme="pyramid"  # "pyramid" | "equal"
    ):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d_v, hidden_states[0])] +
                                    [Layer(hidden_states[_], hidden_states[_ + 1]) for _ in range(n_layer - 1)])

        self.shared_mu = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        self.shared_logvar = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        
        self.private_mu = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        self.private_logvar = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        
        self.dropout = nn.Dropout(dropout)

        # 每个视图一个 SharedStem
        self.stems = nn.ModuleList([SharedStem(d_in=D, dropout=dropout) for D in d_list])

        # 基于 d_emb 定义各通道目标维度 d_ks
        if channel_scheme == "pyramid":
            # 金字塔基于 d_emb
            base_divs = (1, 2, 4, 8)[:self.n_channels]
            d_ks = _safe_dims_from_base(self.d_emb, base_divs)
            concat_dim = sum(d_ks)
        elif channel_scheme == "equal":
            q, r = divmod(self.d_emb, self.n_channels)
            d_ks = [max(1, q + (1 if i < r else 0)) for i in range(self.n_channels)]
            concat_dim = sum(d_ks)
            d_ks = [d_emb] * n_channels
            concat_dim = n_channels * d_emb
        else:
            raise ValueError(f"Unknown channel_scheme: {channel_scheme}")
        self.shared_mu = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        self.shared_logvar = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        # 对每个视图，建立 n_channels 个头部
        self.heads = nn.ModuleList()
        for D in d_list:
            ch = nn.ModuleList([
                ChannelHead(d_in=D, d_k=d_k, dropout=dropout) for d_k in d_ks
            ])
            self.heads.append(ch)

        # 视图级统一线性投影到 d_emb_new（若 concat_dim 已等于 d_emb_new，也保持线性层，便于自由调整）
        # self.view_proj_mu = nn.ModuleList([nn.Linear(concat_dim, self.d_emb_new) for _ in range(self.n_view)])
        # self.view_proj_lv = nn.ModuleList([nn.Linear(concat_dim, self.d_emb_new) for _ in range(self.n_view)])

    @property
    def out_dim(self):
        return self.d_emb_new

    def forward(self, data_v):
        """
        data_v: list of length V, each tensor [B, D_v]
        returns:
          view_mu:     [B, V, d_emb_new]
          view_logvar: [B, V, d_emb_new]
        """
        head_view_mu, head_view_lv = [[] for i in range(self.n_channels)], [[] for i in range(self.n_channels)]
        view_head_mu, view_head_lv = [[] for i in range(self.n_view)], [[] for i in range(self.n_view)]
        
            
        for head_i in range(self.n_channels):
            ch_list = []
            for v in range(self.n_view):
                h = self.stems[v](data_v[v])
                mu_k, lv_k = self.heads[v][head_i](h)   # [B, d_k]
                head_view_mu[head_i].append(mu_k)
                head_view_lv[head_i].append(lv_k)
                view_head_mu[v].append(mu_k)
                view_head_lv[v].append(lv_k)

        head_view_mu_ = [torch.stack(head_view_mu[i], dim=1) for i in range(self.n_channels)] # list [tensor [B,V,d_k]]
        head_view_lv_ = [torch.stack(head_view_lv[i], dim=1) for i in range(self.n_channels)]  # list [tensor [B,V,d_k]]
        channel_cat_mu = torch.stack([torch.cat(view_head_mu[i], dim=-1) for i in range(self.n_view)],dim=1) # tensor [B,V,d_emb_new]
        channel_cat_lv = torch.stack([torch.cat(view_head_lv[i], dim=-1) for i in range(self.n_view)],dim=1) # tensor [B,V,d_emb_new]
        # view_logvar = torch.cat(lv_list, dim=1)
        return head_view_mu_, head_view_lv_ , channel_cat_mu, channel_cat_lv




class Layer(nn.Module):
    def __init__(self, d_in, d_out, dropout=0.2):
        super().__init__()
        assert 0.0 <= float(dropout) <= 1.0
        self.layer = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.LayerNorm(d_out),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layer(x)

class Encoder_oneview(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1, n_channels=4,
        channel_scheme="pyramid"  # "pyramid" | "equal"
        ):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d_v, hidden_states[0])] +
                                    [Layer(hidden_states[_], hidden_states[_ + 1]) for _ in range(n_layer - 1)])
        self.n_channels = n_channels
        self.d_emb = d_emb

        # 基于 d_emb 定义各通道目标维度 d_ks
        if channel_scheme == "pyramid":
            # 金字塔基于 d_emb
            base_divs = (1, 2, 4, 8)[:self.n_channels]
            d_ks = _safe_dims_from_base(self.d_emb, base_divs)
            concat_dim = sum(d_ks)
        elif channel_scheme == "equal":
            q, r = divmod(self.d_emb, self.n_channels)
            d_ks = [max(1, q + (1 if i < r else 0)) for i in range(self.n_channels)]
            concat_dim = sum(d_ks)
            d_ks = [d_emb] * n_channels
            concat_dim = n_channels * d_emb
        else:
            raise ValueError(f"Unknown channel_scheme: {channel_scheme}")
        self.concat_dim = concat_dim
        self.d_ks = d_ks
        self.dropout = nn.Dropout(dropout)
        self.ch_head_mu = nn.ModuleList([nn.Sequential(nn.Linear(hidden_states[-1], d_k), nn.Dropout(dropout)) for d_k in d_ks])
        self.ch_head_logvar = nn.ModuleList([nn.Sequential(nn.Linear(hidden_states[-1], d_k), nn.Dropout(dropout)) for d_k in d_ks])
        # self.shared_mu = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        # self.shared_logvar = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        
        # self.private_mu = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        # self.private_logvar = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        # shared_mu = self.shared_mu(x)
        # shared_logvar = self.shared_logvar(x)
        # private_mu = self.private_mu(x)
        # private_logvar = self.private_logvar(x)
        ch_mu_list = []
        for head in self.ch_head_mu:
            ch_mu_list.append(head(x))
        ch_lv_list = []
        for head in self.ch_head_mu:
            ch_lv_list.append(head(x))
        return ch_mu_list, ch_lv_list

class DecoderPerView(nn.Module):
    """
    简单逐视图解码器：z_v [B, d_emb_new] -> x_hat_v [B, D_v]
    """
    def __init__(self, d_out, d_emb_new, n_layer=2, hidden_ratio=2.0, dropout=0.1):
        super().__init__()
        assert 0.0 <= float(dropout) <= 1.0
        hs = []
        in_dim = d_emb_new
        for _ in range(max(0, n_layer - 1)):
            h = int(max(d_emb_new, hidden_ratio * d_emb_new))
            hs.append(nn.Sequential(
                nn.Linear(in_dim, h),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ))
            in_dim = h
        self.mlp = nn.Sequential(*hs)
        self.out = nn.Linear(in_dim, d_out)

    def forward(self, z):
        h = self.mlp(z) if len(self.mlp) > 0 else z
        return self.out(h)
class Decoder(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1):
        super().__init__()
        self.first = nn.Linear(d_emb, hidden_states[-1])
        self.mid = nn.ModuleList(
            [Layer(hidden_states[n_layer - 1 - _], hidden_states[n_layer - 2 - _]) for _ in range(n_layer - 1)])
        self.last = nn.Linear(hidden_states[0], d_v)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.first(x)
        for layer in self.mid:
            x = layer(x)
        x = self.dropout(x)
        x = self.last(x)
        return x

class Classifier(nn.Module):
    def __init__(self, d_emb, n_cls, dropout=0.2):
        super().__init__()
        assert 0.0 <= float(dropout) <= 1.0
        self.layer = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, n_cls),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layer(x)


class QualityAssessmentNetwork(nn.Module):
    def __init__(self, d_emb, hidden_dim=128):
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.Linear(d_emb + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        self.uncertainty_net = nn.Sequential(
            nn.Linear(d_emb , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        # print(self.quality_net)
        # print(self.uncertainty_net)
    def forward(self, mu, logvar, mask_v):
        # mu/logvar: [B,V,d_emb_new], mask_v: [B,V]
        # uncertainty = torch.mean(logvar, dim=-1, keepdim=True)  # [B,V,1]
        uncertainty = self.uncertainty_net(logvar)
        features = torch.cat([mu, uncertainty], dim=-1)         # [B,V,d_emb_new+1]
        quality_scores = self.quality_net(features).squeeze(-1) # [B,V]
        quality_scores = quality_scores * mask_v
        return quality_scores


class MixtureOfExperts(nn.Module):
    def __init__(self, num_experts, d_emb, use_quality_net=True):
        super().__init__()
        self.num_experts = num_experts
        self.use_quality_net = use_quality_net
        if use_quality_net:
            self.quality_net = QualityAssessmentNetwork(d_emb)
        else:
            self.weights = nn.Parameter(torch.softmax(torch.zeros([1, num_experts, 1]), dim=1))

    def forward(self, mu, logvar, mask_v, eps=1e-8):
        if self.use_quality_net:
            quality_scores = self.quality_net(mu, logvar, mask_v)     # [B,V]
            alpha = quality_scores.unsqueeze(-1)                       # [B,V,1]
            alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + eps)
        else:
            alpha = torch.softmax(
                torch.ones_like(mu[..., :1]).masked_fill(mask_v.unsqueeze(-1) == 0, -1e9),
                dim=1
            )
        # print("alpha:",alpha.min(),alpha.max())
        var = torch.exp(logvar) + eps
        weighted_mu = torch.sum(alpha * mu, dim=1)                 # [B,d_emb_new]
        weighted_var = torch.sum(alpha * alpha * var, dim=1)       # [B,d_emb_new]
        weighted_logvar = torch.log(weighted_var + eps)            # [B,d_emb_new]
        return weighted_mu, weighted_logvar, alpha * self.num_experts


class MVAE(nn.Module):
    """
    修正版 MVAE：
      - 先根据 (d_emb, n_channels, scheme) 确定 d_emb_new；
      - 将 d_emb_new 传入 MultiChannelVAEEncoder 与 Decoder；
      - 其余接口保持兼容。
    """
    def __init__(
        self,
        d_list,
        d_emb,
        n_cls,
        use_quality_net=True,
        dropout=0.1,
        n_enc_layer=2,
        n_dec_layer=2,
        with_recon=True,
        n_channels=4,
        channel_scheme="pyramid"   # "pyramid" | "equal"
    ):
        super().__init__()
        self.n_view = len(d_list)
        enc_hidden_states = []
        dec_hidden_states = []
        for _ in range(self.n_view):
            temp_hidden_states = []
            temp_hidden_states_ = []

            for i in range(n_enc_layer):
                hd = round(d_emb * 2)
                hd = int(hd)
                temp_hidden_states.append(hd)
            for i in range(n_dec_layer):
                hd = round(d_emb * 2)
                hd = int(hd)
                temp_hidden_states_.append(hd)

            enc_hidden_states.append(temp_hidden_states)
            dec_hidden_states.append(temp_hidden_states_)
        assert 0.0 <= float(dropout) <= 1.0
        
        self.with_recon = with_recon
        self.n_channels = n_channels
        self.channel_scheme = channel_scheme
        # 1) 先行确定 d_emb_new
        d_emb = int(d_emb)
        # print(enc_hidden_states, d_emb, n_enc_layer)
        self.encoder_list = nn.ModuleList([Encoder_oneview(d_list[v], enc_hidden_states[v], d_emb, n_enc_layer, dropout=0.1, n_channels=n_channels,
        channel_scheme=self.channel_scheme 
        ) for v in range(self.n_view)])

        self.d_emb_new = self.encoder_list[0].concat_dim
        self.d_emb = d_emb
        # 2) Encoder：输出 [B,V,d_emb_new]
        # self.encoder = MultiChannelVAEEncoder(
        #     d_list=d_list,
        #     d_emb=d_emb,
        #     d_emb_new=d_emb_new,
        #     dropout=dropout,
        #     hidden_ratio=hidden_ratio,
        #     n_channels=n_channels,
        #     channel_scheme=channel_scheme
        # )

        # 3) MoE + 分类器
        self.experts = MixtureOfExperts(num_experts=self.n_view, d_emb=self.d_emb_new, use_quality_net=use_quality_net)
        self.cls = Classifier(self.d_emb_new, n_cls, dropout=dropout)

        # 4) 逐视图解码器（输入维度为 d_emb_new）
        if with_recon:
            self.decoders = nn.ModuleList([Decoder(d_list[v], dec_hidden_states[v], self.d_emb_new, n_dec_layer, dropout) for v in range(self.n_view)])
        else:
            self.decoders = None

    def reparametrize(self, mu, logvar):
        if self.training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def infer(self, data_v):
        ch_fus_mu_list, ch_fus_lv_list,  = [], []
        ch_split_mu_list, ch_split_lv_list = [[] for _ in range(self.n_channels)],[[] for _ in range(self.n_channels)]
        for enc_i, enc in enumerate(self.encoder_list):
            view_mu, view_logvar = enc(data_v[enc_i])
            for ch in range(len(view_mu)):
                ch_split_mu_list[ch].append(view_mu[ch])
                ch_split_lv_list[ch].append(view_logvar[ch])
            ch_fus_mu_list.append(torch.cat(view_mu,dim=-1)) # list [tensor [B,d_emb_new]]
            ch_fus_lv_list.append(torch.cat(view_logvar,dim=-1)) # list [tensor [B,d_emb_new]]
        ch_fus_mu = torch.stack(ch_fus_mu_list, dim=1)  # tensor [B,V,d_emb_new]
        ch_fus_lv = torch.stack(ch_fus_lv_list, dim=1)  # tensor [B,V,d_emb_new]
        ch_split_mu = [torch.stack(ch_split_mu_list[ch],dim=1) for ch in range(len(ch_split_mu_list))]
        ch_split_lv = [torch.stack(ch_split_lv_list[ch],dim=1) for ch in range(len(ch_split_lv_list))]
        return ch_fus_mu,ch_fus_lv,ch_split_mu,ch_split_lv
    def forward(self, data_v, mask_v):
        """
        输入:
          - data_v: list 长度 V, 每个 [B, D_v]
          - mask_v: [B,V]，0 表缺失
        返回（保持字段顺序与数量以兼容原代码）:
          pred, recons, view_mu, view_logvar, None, None, z_views, None, weights, quality_scores
        """
        prop = 0.25
        if self.training is True:
            for i, X in enumerate(data_v):
                bit_mask_len = int(prop * X.size(-1))
                bit_mask = torch.ones_like(X)
                for j in range(mask_v.shape[0]):
                    zero_indices = torch.randperm(bit_mask.shape[1])[:bit_mask_len]
                    bit_mask[j, zero_indices] = 0
                data_v[i] = data_v[i].mul(bit_mask)
        
        # 1) 编码
        # head_view_mu_, head_view_lv_ , channel_cat_mu, channel_cat_lv = self.encoder_list(data_v)  # [[[B,d_k]]]
        ch_fus_mu,ch_fus_lv,ch_split_mu,ch_split_lv = self.infer(data_v)

        

        # 2) MoE 融合
        fused_mu, fused_logvar, weights = self.experts(ch_fus_mu, ch_fus_lv, mask_v)  # [B,d_emb_new], [B,d_emb_new], [B,V,1]
        z = self.reparametrize(fused_mu, fused_logvar)
        
        z_views = self.reparametrize(ch_fus_mu, ch_fus_lv)
        # 质量分数（若启用）
        quality_scores = self.experts.quality_net(ch_fus_mu, ch_fus_lv, mask_v) if hasattr(self.experts, "quality_net") else None

        # 3) 分类
        pred = self.cls(z)
        pred_views_list=[]
        for v in range(self.n_view):
            pred_views_list.append(self.cls(z_views[:,v,:]))
        pred_views = torch.stack(pred_views_list,dim=1) #[B V C]
        # 4) 可选重构（逐视图）
        recons = None
        z_views = None
        
        
        if self.decoders is not None:
            z_views = self.reparametrize(ch_fus_mu, ch_fus_lv)  # [B,V,d_emb_new]
            recons = []
            for v in range(self.n_view):
                x_hat_v = self.decoders[v](z_views)    # [B,V, D_v]
                recons.append(x_hat_v)

        return pred, recons, [ch_fus_mu], [ch_fus_lv], pred_views, z_views, weights, quality_scores
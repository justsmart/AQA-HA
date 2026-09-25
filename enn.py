import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Dict, Tuple
# ========== 1) Evidential Generation Module (EGM)：直接从 z 产生证据 ==========
class EvidentialGenerationModuleFromZ(nn.Module):
    """
    z: [B, D] -> e: [B, 2K], 非负激活(ReLU/Softplus)
    """
    def __init__(self, z_dim: int, num_classes: int, hidden: int = 0, activation: str = "relu"):
        super().__init__()
        self.num_classes = num_classes
        layers = []
        if hidden and hidden > 0:
            layers += [nn.Linear(z_dim, hidden), nn.ReLU(inplace=True)]
            last = hidden
        else:
            last = z_dim
        layers += [nn.Linear(last, 2 * num_classes)]
        self.fc = nn.Sequential(*layers)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "softplus":
            self.act = nn.Softplus(beta=1, threshold=20)
        else:
            raise ValueError("activation must be relu or softplus")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        e_raw = self.fc(z)        # [B, 2K]
        e = self.act(e_raw)       # 保证 e>=0
        return e

def compute_ciw_from_frequency(y_binary: torch.Tensor, eps: float = 1e-6,
                               mode: str = "inv", normalize: str = "zscore",
                               gamma: float = 2.0, bias: float = 0.0) -> torch.Tensor:
    """
    y_binary: [N, K] 多标签0/1，N样本数, K类别数
    mode:
      - "inv": w_k = 1 / (f_k + eps)
      - "log": w_k = log((N + eps) / (n_k + eps))
      - "sqrt_inv": w_k = 1 / sqrt(f_k + eps)  # 更平滑
    normalize:
      - "zscore": (w - mean)/std
      - "minmax": (w - min)/(max - min) 映射到[0,1]，随后再线性到[-1,1]
      - "none": 不做标准化（不建议）
    gamma: 缩放系数，用于 EBRA 前的线性缩放
    bias: 平移项
    returns:
      ciw: [K]，未过sigmoid的“对数几率”型权重（将送入 EBRA 的 sigmoid）
    """
    N, K = y_binary.shape
    n_pos = y_binary.sum(dim=0)               # [K]
    f = n_pos / (N + eps)                     # 频率

    if mode == "inv":
        w = 1.0 / (f + eps)
    elif mode == "log":
        w = torch.log((N + eps) / (n_pos + eps))
    elif mode == "sqrt_inv":
        w = 1.0 / torch.sqrt(f + eps)
    else:
        raise ValueError("mode must be inv/log/sqrt_inv")

    # 归一化
    if normalize == "zscore":
        mu, std = w.mean(), w.std().clamp_min(1e-6)
        w_norm = (w - mu) / std
    elif normalize == "minmax":
        w_min, w_max = w.min(), w.max()
        w_norm = (w - w_min) / (w_max - w_min + 1e-6)  # [0,1]
        w_norm = w_norm * 2 - 1                        # 映射到[-1,1]
    elif normalize == "none":
        w_norm = w
    else:
        raise ValueError("normalize must be zscore/minmax/none")

    # 变为“对数几率”风格的 ciw，后续 EBRA 会做 sigmoid(ciw)
    ciw = gamma * w_norm + bias
    return ciw



# ========== 2) EBRA：由 CIW 计算 â ==========
class EBRA(nn.Module):
    """
    â^+_k = σ(CIW_k), â^-_k = 1 - σ(CIW_k)
    """
    def __init__(self, ciw: torch.Tensor):
        super().__init__()
        self.register_buffer("ciw", ciw.clone().float())

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:
        s = torch.sigmoid(self.ciw)  # [K]
        a_pos = s
        a_neg = 1.0 - s
        return a_pos, a_neg


# ========== 3) Head：从 e 与 â 计算 Dirichlet/SL 量（不含损失） ==========
class TMSDCHead(nn.Module):
    """
    - α^i_k = e^i_k + â^i_k * W, W=2（二元）
    - u_k = W / (α^+_k + α^-_k)
    - p^+_k = (e^+ + â^+ W) / (W + e^+ + e^-)
    """
    def __init__(self, num_classes: int, W: float = 2.0):
        super().__init__()
        self.num_classes = num_classes
        self.W = W

    def split_e(self, e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = e.size(0)
        e = e.view(B, 2, self.num_classes)
        e_pos = e[:, 0, :]
        e_neg = e[:, 1, :]
        return e_pos, e_neg

    def forward(self, e: torch.Tensor, a_pos: torch.Tensor, a_neg: torch.Tensor) -> Dict[str, torch.Tensor]:
        e_pos, e_neg = self.split_e(e)                # [B,K]
        a_pos = a_pos.unsqueeze(0).expand_as(e_pos)   # [B,K]
        a_neg = a_neg.unsqueeze(0).expand_as(e_neg)   # [B,K]

        alpha_pos = e_pos + a_pos * self.W
        alpha_neg = e_neg + a_neg * self.W
        sum_alpha = alpha_pos + alpha_neg

        u = self.W / (sum_alpha + 1e-8)
        p_pos = (e_pos + a_pos * self.W) / (self.W + e_pos + e_neg + 1e-8)

        # 可选：信念质量 b（非必须）
        b_pos = (alpha_pos - a_pos * self.W) / (sum_alpha + 1e-8)
        b_neg = (alpha_neg - a_neg * self.W) / (sum_alpha + 1e-8)

        return {
            "alpha_pos": alpha_pos,
            "alpha_neg": alpha_neg,
            "sum_alpha": sum_alpha,
            "u": u.clamp(0, 1),
            "p": p_pos.clamp(0, 1),
            "b_pos": b_pos.clamp(0, 1),
            "b_neg": b_neg.clamp(0, 1),
        }


# ========== 4) 不确定性聚合 ==========
def aggregate_uncertainty(u_per_class: torch.Tensor, method: str = "max", topk: int = 5) -> torch.Tensor:
    if method == "max":
        return u_per_class.max(dim=1).values
    elif method == "sum":
        return u_per_class.sum(dim=1)
    elif method == "topk":
        k = min(topk, u_per_class.size(1))
        vals, _ = torch.topk(u_per_class, k, dim=1, largest=True, sorted=False)
        return vals.sum(dim=1)
    else:
        raise ValueError("method must be max/sum/topk")

# ========== 6) 独立的损失函数（论文式(10)的 Type-II 近似） ==========
def loss_typeII_log_approx(alpha_pos: torch.Tensor, alpha_neg: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
    """
    依据论文式(10)（对数近似形式）：
    L_k = y^+ [log(Ŝ_k) - log(α̂^+)] + (1 - y^+) [log(Ŝ_k) - log(α̂^-)]
    对 batch & K 取均值
    输入:
      alpha_pos, alpha_neg: [B,K]
      y_pos: [B,K] ∈ {0,1}
    """
    sum_alpha = alpha_pos + alpha_neg
    log_S = torch.log(sum_alpha + 1e-8)
    log_alpha_pos = torch.log(alpha_pos + 1e-8)
    log_alpha_neg = torch.log(alpha_neg + 1e-8)
    loss = y_pos * (log_S - log_alpha_pos) + (1.0 - y_pos) * (log_S - log_alpha_neg)
    return loss.mean()


# ========== 7) 可选：digamma 版本（Beta-EDL 常用形式） ==========
def beta_edl_loss(alpha_pos: torch.Tensor, alpha_neg: torch.Tensor, y_pos: torch.Tensor) -> torch.Tensor:
    """
    Beta-EDL (digamma) 版损失（与另一篇Beta-ENN等价）：
    L = y*(ψ(α_sum)-ψ(α_pos)) + (1-y)*(ψ(α_sum)-ψ(α_neg))
    """
    psi = torch.special.digamma
    sum_alpha = alpha_pos + alpha_neg
    term_pos = psi(sum_alpha) - psi(alpha_pos)
    term_neg = psi(sum_alpha) - psi(alpha_neg)
    loss = y_pos * term_pos + (1 - y_pos) * term_neg
    return loss.mean()


# ========== 6) 整体模型（仅负责前向/不含损失） ==========
class TMSDCFromZ(nn.Module):
    """
    直接输入 z: [B, D]
    """
    def __init__(self, z_dim: int, num_classes: int, label: torch.Tensor, egm_hidden: int = 0, egm_act: str = "relu"):
        super().__init__()
        self.num_classes = num_classes
        self.egm = EvidentialGenerationModuleFromZ(z_dim, num_classes, hidden=egm_hidden, activation=egm_act)
        ciw = compute_ciw_from_frequency(label, mode="log", normalize="zscore", gamma=2.0, bias=0.0)
        self.ebra = EBRA(ciw=ciw)     # â
        self.head = TMSDCHead(num_classes)

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        z: [B, D]
        returns: e, a_pos, a_neg, alpha_pos, alpha_neg, sum_alpha, u, p_pos, b_pos, b_neg
        """
        e = self.egm(z)                        # [B, 2K]
        a_pos, a_neg = self.ebra()             # [K], [K]
        out = self.head(e, a_pos, a_neg)
        out.update({"e": e, "a_pos": a_pos, "a_neg": a_neg})
        return out
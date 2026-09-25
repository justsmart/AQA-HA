import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, alpha, gamma, ev_reg=0.1, consistency_weight=1.0, diversity_weight=1.0, temperature=0.1,
                 quality_weight=0.1, use_focal=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.consistency_weight = consistency_weight
        self.diversity_weight = diversity_weight
        self.temperature = temperature
        self.quality_weight = quality_weight
        self.use_focal = use_focal
        self.bce = nn.BCELoss(reduction='none')

    def weighted_mse_loss(self, mask_v, data, rec_v):
        n_view = len(rec_v)
        mse1 = 0
        for i in range(n_view):
            mse1 += torch.mean(torch.pow(data[i].detach() - rec_v[i], 2) * mask_v[:, i:i + 1])
        return self.alpha * mse1

    def weighted_bce_loss(self, pred, label, mask_l):
        mask_sum = torch.sum(mask_l)
        if mask_sum == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        if self.use_focal:
            loss = self.focal(pred, label, mask_l)
            return loss
        else:
            return torch.sum(self.bce(pred, label) * mask_l) / mask_sum

    def contrastive_loss(self, shared_mu, mask_v, weights, temperature=None):
        if temperature is None:
            temperature = self.temperature
            
        batch_size, n_view, d_emb = shared_mu.shape
        total_loss = 0
        valid_pairs = 0
        
        for i in range(n_view):
            for j in range(i + 1, n_view):
                valid_mask = (mask_v[:, i] > 0) & (mask_v[:, j] > 0)
                valid_indices = torch.where(valid_mask)[0]

                if len(valid_indices) < 2:
                    continue

                anchor = shared_mu[valid_indices, i, :]
                positive = shared_mu[valid_indices, j, :]
                weights_i = weights[valid_indices, i, :]
                weights_j = weights[valid_indices, j, :]

                anchor = F.normalize(anchor, p=2, dim=1)
                positive = F.normalize(positive, p=2, dim=1)

                all_sim = torch.mm(anchor, positive.t()) / temperature
                all_sim *= torch.mm(weights_i, weights_j.t())

                pos_sim = torch.diag(all_sim)
                log_sum_exp = torch.logsumexp(all_sim, dim=1)

                losses = -pos_sim + log_sum_exp
                total_loss += torch.sum(losses)

                valid_pairs += len(valid_indices)
        
        return total_loss / max(valid_pairs, 1) if valid_pairs > 0 else torch.tensor(0.0, device=shared_mu.device)

    def shared_consistency_loss(self, shared_mu, mask_v):
        return torch.tensor(0.0, device=shared_mu.device)
    

    def shared_private_independence_loss(self, shared_z, private_z, mask_v, eps=1e-8):
        batch_size, n_view, d_emb = shared_z.shape
        independence_loss = 0
        valid_count = 0
        
        for v in range(n_view):
            valid_mask = mask_v[:, v] == 1
            
            if valid_mask.sum() == 0:
                continue
            valid_shared = shared_z[valid_mask, v, :]
            valid_private = private_z[valid_mask, v, :]
            shared_norm = F.normalize(valid_shared, dim=1, eps=eps)
            private_norm = F.normalize(valid_private, dim=1, eps=eps)
            
            correlation = torch.sum(shared_norm * private_norm, dim=1)
            modal_independence_loss = torch.mean(correlation ** 2)
            independence_loss += modal_independence_loss
            valid_count += 1

        if valid_count > 0:
            independence_loss = independence_loss / valid_count
        else:
            independence_loss = torch.tensor(0.0, device=shared_z.device)
        return independence_loss

    
    def kl_divergence_loss(self, shared_mu, shared_logvar, mask_v):
        shared_kld = torch.mean(-0.5 * torch.sum(
            (1 + shared_logvar - shared_mu.pow(2) - shared_logvar.exp()) * mask_v.unsqueeze(-1), dim=2
        ))

        return shared_kld * 1e-3
    
    def quality_supervision_loss(self, data, rec_v, mask_v, quality_scores):
        n_view = len(data)
        batch_size = data[0].shape[0]

        reconstruction_errors = []
        for i in range(n_view):
            mse_error = torch.mean((data[i] - rec_v[i])**2, dim=-1)
            reconstruction_errors.append(mse_error)
        
        reconstruction_errors = torch.stack(reconstruction_errors, dim=1)
        
        target_quality = torch.exp(-reconstruction_errors)

        target_quality = target_quality * mask_v
        target_quality = target_quality / (torch.sum(target_quality, dim=1, keepdim=True) + 1e-8)
        
        pred_quality = quality_scores * mask_v
        pred_quality = pred_quality / (torch.sum(pred_quality, dim=1, keepdim=True) + 1e-8)
        quality_loss = torch.mean((pred_quality - target_quality)**2 * mask_v)
        
        return quality_loss
    
    # @staticmethod
    def beta_loss(self, alpha_pos: torch.Tensor, alpha_neg: torch.Tensor, y_pos: torch.Tensor):
        """
        Beta loss 对应论文式(2)-(3)，对每类独立二元，取 batch-mean。
        y: [B, K], 0/1 二值标签（多标签任务）
        """
        psi = torch.special.digamma
        sum_alpha = alpha_pos + alpha_neg
        term_pos = psi(sum_alpha) - psi(alpha_pos)
        term_neg = psi(sum_alpha) - psi(alpha_neg)
        loss = y_pos * term_pos + (1 - y_pos) * term_neg
        return loss.mean()
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
    def total_disentangled_loss(self, pred, label, inc_L_ind, data, rec_v, inc_V_ind,
                               shared_mu, shared_logvar, private_mu, private_logvar, shared_z, private_z, weights,
                               quality_scores=None):
        mse_loss = self.weighted_mse_loss(inc_V_ind, data, rec_v)
        bce_loss = self.weighted_bce_loss(pred, label, inc_L_ind)
        # bce_loss = self.beta_loss(alpha, beta, label)
        kld_loss = self.kl_divergence_loss(shared_mu, shared_logvar,  inc_V_ind) \
            # + self.kl_divergence_loss(private_mu, private_logvar, inc_V_ind)

         
        consistency_loss = self.contrastive_loss(shared_mu, inc_V_ind, weights.detach())
        # independence_loss = self.shared_private_independence_loss(shared_z, private_z, inc_V_ind)
        
        if quality_scores is not None:
            quality_loss = self.quality_supervision_loss(data, rec_v, inc_V_ind, quality_scores)
        else:
            quality_loss = torch.tensor(0.0, device=label.device)

        total_loss = (bce_loss + mse_loss + kld_loss + 
                     self.consistency_weight * consistency_loss + 
                    #  self.diversity_weight * independence_loss +
                     self.quality_weight * quality_loss)
        
        return total_loss, {
            'bce_loss': bce_loss.item(),
            'mse_loss': mse_loss.item(),
            'kld_loss': kld_loss.item(),
            'contrastive_loss': consistency_loss.item() if isinstance(consistency_loss, torch.Tensor) else consistency_loss,
            # 'independence_loss': independence_loss.item() if isinstance(independence_loss, torch.Tensor) else independence_loss,
            'quality_loss': quality_loss.item() if isinstance(quality_loss, torch.Tensor) else quality_loss,
            'total_loss': total_loss.item()
        }



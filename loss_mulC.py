import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss(nn.Module):
    def __init__(self, alpha, gamma, lammda, ev_reg=0.1, consistency_weight=1.0, diversity_weight=1.0, temperature=0.1,
                 quality_weight=0.1, use_focal=False):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lammda = lammda
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
        return mse1

    def weighted_cross_mse_loss(self, mask_v, data, rec_v):
        # data [V,B,D] rec_v list[[V,B,D]*n_view]
        n_view = len(rec_v)
        mse1 = mse2 = 0
        for i in range(n_view):
            mse1 += torch.mean(torch.pow(data[i].detach() - rec_v[i][:, i, :], 2) * mask_v[:, i:i + 1])
            for j in range(n_view):
                if i != j:
                    mse2 += torch.mean(
                        torch.pow(data[i].detach() - rec_v[i][:, j, :], 2) * mask_v[:, i:i + 1] * mask_v[:, j:j + 1])
        return  mse1 / n_view +  self.lammda* mse2 / (n_view * n_view)
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
                # print("weights:",weights.min(),weights.max())
                anchor = F.normalize(anchor, p=2, dim=1)
                positive = F.normalize(positive, p=2, dim=1)
                # losses = ((anchor-positive)**2).sum(-1)
                
                all_sim = torch.mm(anchor, positive.t()) / temperature
                all_sim *= torch.mm(weights_i, weights_j.t())

                pos_sim = torch.diag(all_sim)
                log_sum_exp = torch.logsumexp(all_sim, dim=1)
                losses = -pos_sim + log_sum_exp
                # losses = -pos_sim
                
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

    def quality_supervision_loss(self, data, rec_v, label, pred_views, mask_v, mask_l, quality_scores):
        n_view = len(data)
        batch_size = data[0].shape[0]

        reconstruction_errors = []
        for i in range(n_view):
            mse_error = torch.mean((data[i] - rec_v[i][:,i])**2, dim=-1)
            reconstruction_errors.append(mse_error)
        
        reconstruction_errors = torch.stack(reconstruction_errors, dim=1)
        
        cls_errors = []
        for v in range(pred_views.shape[1]):
            cls_errors.append((self.bce(pred_views[:,v,:],label) * mask_l).sum(dim=-1).detach())
        cls_errors = torch.stack(cls_errors,dim=1)
        
        target_quality = torch.exp(-reconstruction_errors)
        target_quality = target_quality * mask_v
        target_quality = target_quality / (torch.sum(target_quality, dim=1, keepdim=True) + 1e-8)
        
        target_quality_cls = torch.exp(-cls_errors)
        target_quality_cls = target_quality_cls * mask_v
        target_quality_cls = target_quality_cls / (torch.sum(target_quality_cls, dim=1, keepdim=True) + 1e-8)
        target_quality = target_quality*0.5 + target_quality_cls*0.5
        
        pred_quality = quality_scores * mask_v 
        pred_quality = pred_quality / (torch.sum(pred_quality, dim=1, keepdim=True) + 1e-8)
        quality_loss = torch.mean((pred_quality - target_quality)**2 * mask_v)
        
        return quality_loss
    

    def total_disentangled_loss(self, pred, label, inc_L_ind, data, rec_v, pred_views, inc_V_ind,
                               shared_mu_list, shared_logvar_list, weights,
                               quality_scores=None):
        shared_mu = shared_mu_list[0]
        shared_logvar = shared_logvar_list[0]
        mse_loss = self.weighted_cross_mse_loss(inc_V_ind, data, rec_v)
        bce_loss = self.weighted_bce_loss(pred, label, inc_L_ind)
        # bce_loss = self.beta_loss(alpha, beta, label)
        kld_loss = 0
        for head in range(len(shared_mu_list)):
            kld_loss += self.kl_divergence_loss(shared_mu_list[head], shared_logvar_list[head],  inc_V_ind) 
            # + self.kl_divergence_loss(private_mu, private_logvar, inc_V_ind)
        kld_loss = kld_loss / len(shared_mu_list)
         
        consistency_loss = self.contrastive_loss(shared_mu, inc_V_ind, weights.detach())
        # independence_loss = self.shared_private_independence_loss(shared_z, private_z, inc_V_ind)
        
        if quality_scores is not None:
            quality_loss = self.quality_supervision_loss(data, rec_v, label, pred_views, inc_V_ind, inc_L_ind, quality_scores)
        else:
            quality_loss = torch.tensor(0.0, device=label.device)

        total_loss = (bce_loss + mse_loss * self.alpha + kld_loss *self.gamma+ 
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



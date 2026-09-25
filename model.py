import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter


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


class Encoder(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([Layer(d_v, hidden_states[0])] +
                                    [Layer(hidden_states[_], hidden_states[_ + 1]) for _ in range(n_layer - 1)])

        self.shared_mu = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        self.shared_logvar = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        
        self.private_mu = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        self.private_logvar = nn.Sequential(nn.Linear(hidden_states[-1], d_emb), nn.Dropout(dropout))
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        
        shared_mu = self.shared_mu(x)
        shared_logvar = self.shared_logvar(x)
        private_mu = self.private_mu(x)
        private_logvar = self.private_logvar(x)
        
        return shared_mu, shared_logvar, private_mu, private_logvar


class Decoder(nn.Module):
    def __init__(self, d_v, hidden_states, d_emb, n_layer, dropout=0.1):
        super().__init__()
        self.first = nn.Linear(d_emb * 2, hidden_states[-1])
        self.mid = nn.ModuleList(
            [Layer(hidden_states[n_layer - 1 - _], hidden_states[n_layer - 2 - _]) for _ in range(n_layer - 1)])
        self.last = nn.Linear(hidden_states[0], d_v)
        self.dropout = nn.Dropout(dropout)

    def forward(self, shared_z, private_z):
        x = torch.cat([shared_z, private_z], dim=-1)
        x = self.first(x)
        for layer in self.mid:
            x = layer(x)
        x = self.dropout(x)
        x = self.last(x)
        return x


class Classifier(nn.Module):
    def __init__(self, d_emb, n_cls, dropout=0.2):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(d_emb, d_emb),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_emb, n_cls),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class MVAE(nn.Module):
    def __init__(self, d_list, d_emb, n_enc_layer, n_dec_layer, n_cls, theta, dropout=0.1):
        super(MVAE, self).__init__()
        n_view = len(d_list)
        enc_hidden_states = []
        dec_hidden_states = []

        for _ in range(n_view):
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

        self.encoders = nn.ModuleList([Encoder(d_list[v], enc_hidden_states[v], d_emb, n_enc_layer, dropout) for v in range(n_view)])
        self.decoders = nn.ModuleList([Decoder(d_list[v], dec_hidden_states[v], d_emb, n_dec_layer, dropout) for v in range(n_view)])
        self.cls = Classifier(d_emb, n_cls)
        self.experts = MixtureOfExperts(n_view, d_emb, use_quality_net=True)
        self.d_emb = d_emb
        self.n_view = n_view
        self.attn = nn.MultiheadAttention(self.d_emb, num_heads=2, batch_first=True)

    def infer(self, data_v, mask_v):
        shared_mu, shared_logvar, private_mu, private_logvar = [], [], [], []

        for enc_i, enc in enumerate(self.encoders):
            s_mu, s_logvar, p_mu, p_logvar = enc(data_v[enc_i])
            shared_mu.append(s_mu.unsqueeze(1))
            shared_logvar.append(s_logvar.unsqueeze(1))
            private_mu.append(p_mu.unsqueeze(1))
            private_logvar.append(p_logvar.unsqueeze(1))
            
        shared_mu = torch.cat(shared_mu, dim=1)
        shared_logvar = torch.cat(shared_logvar, dim=1)
        private_mu = torch.cat(private_mu, dim=1)
        private_logvar = torch.cat(private_logvar, dim=1)
        
        return shared_mu, shared_logvar, private_mu, private_logvar

    def reparametrize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, data_v, mask_v):
        shared_mu, shared_logvar, private_mu, private_logvar = self.infer(data_v, mask_v)

        shared_z = self.reparametrize(shared_mu, shared_logvar)
        private_z = self.reparametrize(private_mu, private_logvar)

        data_rec = []
        for dec_i, dec in enumerate(self.decoders):
            rec_data = dec(shared_z[:, dec_i, :], private_z[:, dec_i, :])
            data_rec.append(rec_data)

        fused_shared_mu, fused_shared_logvar, weights = self.experts(shared_mu, shared_logvar, mask_v)
        fused_shared_z = self.reparametrize(fused_shared_mu, fused_shared_logvar)
        pred = self.cls(fused_shared_z)

        if self.experts.use_quality_net:
            quality_scores = self.experts.quality_net(shared_mu, shared_logvar, mask_v)
            return pred, data_rec, shared_mu, shared_logvar, private_mu, private_logvar, shared_z, private_z, weights, quality_scores
        else:
            return pred, data_rec, shared_mu, shared_logvar, private_mu, private_logvar, shared_z, private_z


class QualityAssessmentNetwork(nn.Module):
    def __init__(self, d_emb, hidden_dim=128):
        super().__init__()
        self.quality_net = nn.Sequential(
            nn.Linear(d_emb+1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, mu, logvar, mask_v):
        batch_size, n_view, d_emb = mu.shape

        uncertainty = torch.mean(logvar, dim=-1, keepdim=True)
        features = torch.cat([mu, uncertainty], dim=-1)
        quality_scores = self.quality_net(features).squeeze(-1)
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
            quality_scores = self.quality_net(mu, logvar, mask_v)
            alpha = quality_scores.unsqueeze(-1)
            alpha = alpha / (torch.sum(alpha, dim=1, keepdim=True) + eps)
        else:
            alpha = torch.softmax(torch.ones(mu.shape).to(mu.device).masked_fill(mask_v.unsqueeze(2) == 0, -1e9), dim=1)
        var = torch.exp(logvar) + eps
        weighted_mu = torch.sum(alpha * mu, dim=1)
        weighted_var = torch.sum(alpha * alpha * var, dim=1)
        weighted_logvar = torch.log(weighted_var + eps)
        return weighted_mu, weighted_logvar, alpha * self.num_experts
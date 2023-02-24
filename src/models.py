import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.categorical import Categorical

from utils import *


class Dam(nn.Module):
    """
    Class of DAM model
    """
    def __init__(self, alpha, n_user, n_item, n_bundle, emb_dim,
                 lr=1e-3, decay=1e-5, batch_size=512):
        """
        Initialize the class
        """
        super(Dam, self).__init__()
        self.user_embs = nn.Embedding(n_user, emb_dim)
        self.item_embs = nn.Embedding(n_item, emb_dim)
        self.item_embs_b = nn.Embedding(n_item, emb_dim)
        self.bundle_embs = nn.Embedding(n_bundle, emb_dim)

        self.dense1 = torch.nn.Linear(emb_dim * 2, emb_dim * 2)
        self.dense2 = torch.nn.Linear(emb_dim * 2, emb_dim * 2)
        self.pred_i = torch.nn.Linear(emb_dim * 2, 1)
        self.pred_b = torch.nn.Linear(emb_dim * 2, 1)

        self.alpha = alpha

        # initialization
        nn.init.xavier_normal_(self.user_embs.weight)
        nn.init.xavier_normal_(self.item_embs.weight)
        nn.init.xavier_normal_(self.item_embs_b.weight)
        nn.init.xavier_normal_(self.bundle_embs.weight)
        init_weights(self.dense1)
        init_weights(self.dense2)
        init_weights(self.pred_i)
        init_weights(self.pred_b)

        # optimizer
        self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=decay)
        self.batch_size = batch_size

    def forward(self, x, bundle=False):
        """
        Forward function
        """
        x = torch.relu(self.dense1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = torch.relu(self.dense2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        if bundle:
            x = torch.relu(self.pred_b(x))
        else:
            x = torch.relu(self.pred_i(x))
        return x

    def get_user_embs(self, u):
        """
        Get user embeddings
        """
        return self.user_embs(u)

    def get_item_embs(self, i):
        """
        Get item embeddings
        """
        return self.item_embs(i)

    def get_bundle_embs(self, u, b, DEVICE):
        """
        Get bundle embeddings
        """
        scores = torch.matmul(self.user_embs(u), self.item_embs_b.weight.T)
        mask = spy_sparse2torch_sparse(self.bundle_item[b.cpu().numpy()]).to(DEVICE)
        weights = sparse_masked_softmax(scores, mask)
        b_embs = torch.sparse.mm(weights, self.item_embs.weight)
        return b_embs

    def get_dataset(self, n_user, n_item, n_bundle, bundle_item, user_item,
                    user_bundle_trn, user_bundle_vld, vld_user_idx, user_bundle_test,
                    user_bundle_test_mask):
        """
        Get dataset
        """
        self.n_user = n_user
        self.n_item = n_item
        self.n_bundle = n_bundle
        self.bundle_item = bundle_item
        self.user_item = user_item
        self.user_bundle_trn = user_bundle_trn
        self.user_bundle_vld = user_bundle_vld
        self.vld_user_idx = vld_user_idx
        self.user_bundle_test = user_bundle_test
        self.user_bundle_test_mask = user_bundle_test_mask
        self.bundle_freq = np.array(user_bundle_trn.sum(0)).squeeze().astype(float)

    def get_scores(self, u_idx, b_idx):
        """
        Get user-bundle scores
        """
        n_bundle = b_idx.shape[0]
        u_idx = np.tile(u_idx, (n_bundle, 1)).transpose()
        b_idx = np.tile(np.arange(n_bundle), (u_idx.shape[0], 1))
        u_idx = u_idx.flatten()
        b_idx = b_idx.flatten()
        u_idx = torch.LongTensor(u_idx).to(TRN_DEVICE)
        b_idx = torch.LongTensor(b_idx).to(TRN_DEVICE)
        u_batch = self.get_user_embs(u_idx)
        b_batch = self.get_bundle_embs(u_idx, b_idx, EVA_DEVICE)
        ub = torch.cat((u_batch, b_batch), 1)
        ub = self.forward(ub, bundle=True)
        ub = torch.reshape(ub, (-1, n_bundle))
        return ub

    def update_model(self):
        """
        Update the model
        """
        self.train()
        # u-i interactions
        trn_ui_loss = 0.
        train_item_u, train_item_i = self.user_item.nonzero()
        num_trn = train_item_u.shape[0]
        idx_list = list(range(num_trn))
        np.random.shuffle(idx_list)
        for batch_idx, start_idx in enumerate(range(0, num_trn, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, num_trn)
            self.optimizer.zero_grad()
            u_idx = train_item_u[idx_list[start_idx:end_idx]]
            p_idx = train_item_i[idx_list[start_idx:end_idx]]
            n_idx = np.random.choice(range(self.n_item), u_idx.shape[0], replace=True)
            u_idx = torch.LongTensor(u_idx).to(TRN_DEVICE)
            p_idx = torch.LongTensor(p_idx).to(TRN_DEVICE)
            n_idx = torch.LongTensor(n_idx).to(TRN_DEVICE)
            u_batch = self.get_user_embs(u_idx)
            p_batch = self.get_item_embs(p_idx)
            n_batch = self.get_item_embs(n_idx)
            up = torch.cat((u_batch, p_batch), 1)
            un = torch.cat((u_batch, n_batch), 1)
            up = self.forward(up)
            un = self.forward(un)
            loss = -torch.sum(torch.log(torch.sigmoid(up - un)))
            trn_ui_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        trn_ui_loss = trn_ui_loss / (batch_idx + 1)

        # u-b interactions
        trn_ub_loss = 0.
        train_bundle_u, train_bundle_b = self.user_bundle_trn.nonzero()
        num_trn = train_bundle_b.shape[0]
        idx_list = list(range(num_trn))
        np.random.shuffle(idx_list)
        for batch_idx, start_idx in enumerate(range(0, num_trn, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, num_trn)
            self.optimizer.zero_grad()
            u_idx = train_bundle_u[idx_list[start_idx:end_idx]]
            p_idx = train_bundle_b[idx_list[start_idx:end_idx]]
            # popularity-based negative sampling
            bundle_freq_batch = torch.tile(torch.tensor(self.bundle_freq), (u_idx.shape[0], 1))
            user_interacted = naive_sparse2tensor(self.user_bundle_trn[u_idx])
            bundle_freq_batch *= (1 - user_interacted)
            bundle_prob_batch = torch.div(bundle_freq_batch, bundle_freq_batch.sum(1, keepdim=True))
            uniform = torch.ones_like(bundle_prob_batch)
            uniform = torch.div(uniform, uniform.sum(1, keepdim=True))
            prob_batch = self.alpha * bundle_prob_batch + (1-self.alpha) * uniform
            m = Categorical(prob_batch)
            n_idx = m.sample()

            u_idx = torch.LongTensor(u_idx).to(TRN_DEVICE)
            p_idx = torch.LongTensor(p_idx).to(TRN_DEVICE)
            n_idx = torch.LongTensor(n_idx).to(TRN_DEVICE)
            u_batch = self.get_user_embs(u_idx)
            p_batch = self.get_bundle_embs(u_idx, p_idx, TRN_DEVICE)
            n_batch = self.get_bundle_embs(u_idx, n_idx, TRN_DEVICE)
            up = torch.cat((u_batch, p_batch), 1)
            un = torch.cat((u_batch, n_batch), 1)
            up = self.forward(up, bundle=True)
            un = self.forward(un, bundle=True)
            loss = -torch.sum(torch.log(torch.sigmoid(up - un)))
            trn_ub_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        trn_ub_loss = trn_ub_loss / (batch_idx + 1)

        return trn_ui_loss + trn_ub_loss, 0

    def evaluate_val(self, ks, div=False):
        """
        Evaluate on the validation set
        """
        batch_size = 100
        n_target = 100
        self.eval()
        with torch.no_grad():
            recall_list, map_list, freq_list = [], [], []
            for batch_idx, start_idx in enumerate(range(0, len(self.vld_user_idx), batch_size)):
                end_idx = min(start_idx + batch_size, len(self.vld_user_idx))
                u_idx = self.vld_user_idx[start_idx:end_idx]
                u_idx = np.tile(u_idx, (n_target, 1)).transpose()
                b_idx = self.user_bundle_vld[start_idx:end_idx]
                u_idx = u_idx.flatten()
                b_idx = b_idx.flatten()
                u_idx = torch.LongTensor(u_idx).to(EVA_DEVICE)
                b_idx = torch.LongTensor(b_idx).to(EVA_DEVICE)
                u_batch = self.get_user_embs(u_idx)
                b_batch = self.get_bundle_embs(u_idx, b_idx, EVA_DEVICE)
                ub = torch.cat((u_batch, b_batch), 1)
                ub = self.forward(ub, bundle=True)
                ub = torch.reshape(ub, (-1, n_target))
                pos_idx = torch.zeros(ub.shape[0]).long().to(EVA_DEVICE)
                recalls, maps, freqs = evaluate_metrics(ub, pos_idx.unsqueeze(1), self.bundle_item, ks=ks, div=div)
                recall_list.append(recalls)
                map_list.append(maps)
                freq_list.append(freqs)
            recalls = list(np.array(recall_list).sum(axis=0)/len(self.vld_user_idx))
            maps = list(np.array(map_list).sum(axis=0)/len(self.vld_user_idx))
            freqs = torch.stack(freq_list).sum(dim=0)
        return recalls, maps

    def evaluate_test(self, ks, div=True):
        """
        Evaluate on the test set
        """
        batch_size = 1
        self.eval()
        ubs_origin, ubs_filtered = [], []
        with torch.no_grad():
            recall_list, map_list, freq_list = [], [], []
            user_idx, _ = np.nonzero(np.sum(self.user_bundle_test, 1))
            test_pos_idx = np.nonzero(self.user_bundle_test[user_idx].toarray())[1]
            ub_masks = self.user_bundle_test_mask[user_idx]
            for batch_idx, start_idx in enumerate(range(0, len(user_idx), batch_size)):
                end_idx = min(start_idx + batch_size, len(user_idx))
                u_idx = user_idx[start_idx:end_idx]
                u_idx = np.tile(u_idx, (self.n_bundle, 1)).transpose()
                b_idx = np.tile(np.arange(self.n_bundle), (u_idx.shape[0], 1))
                u_idx = u_idx.flatten()
                b_idx = b_idx.flatten()
                u_idx = torch.LongTensor(u_idx).to(EVA_DEVICE)
                b_idx = torch.LongTensor(b_idx).to(EVA_DEVICE)
                u_batch = self.get_user_embs(u_idx)
                b_batch = self.get_bundle_embs(u_idx, b_idx, EVA_DEVICE)
                ub = torch.cat((u_batch, b_batch), 1)
                ub = self.forward(ub, bundle=True)
                ub = torch.reshape(ub, (-1, self.n_bundle))
                ub_filtered = ub.masked_fill(naive_sparse2tensor(ub_masks[start_idx:end_idx]).bool().to(EVA_DEVICE), -float('inf'))
                pos_idx = torch.LongTensor(test_pos_idx[start_idx:end_idx]).to(EVA_DEVICE)
                recalls, maps, freqs = evaluate_metrics(ub_filtered, pos_idx.unsqueeze(1), self.bundle_item, ks=ks, div=div)
                ubs_filtered.append(ub_filtered.cpu())
                ubs_origin.append(ub.cpu())
                recall_list.append(recalls)
                map_list.append(maps)
                freq_list.append(freqs)
            recalls = list(np.array(recall_list).sum(axis=0)/len(user_idx))
            maps = list(np.array(map_list).sum(axis=0)/len(user_idx))
            freqs = torch.stack(freq_list).sum(dim=0)
        return recalls, maps, torch.cat(ubs_origin, dim=0), torch.cat(ubs_filtered, dim=0)

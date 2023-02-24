from tqdm import tqdm

from utils import *


class PopCon(object):
    """
    Class of PopCon reranking
    """
    def __init__(self, beta, n):
        """
        Initialize the class
        """
        super(PopCon, self).__init__()
        self.beta = beta
        self.n = n

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

        self.bundle_item_dense_tensor = spy_sparse2torch_sparse(self.bundle_item).to_dense()
        self.max_ent = torch.log2(torch.tensor(self.n_item))

    def delta_bundle_batch(self, cur_item_freq, cand_idx_batch):
        """
        Get gains of entropy and coverage
        """
        cur_ent = self.get_entropy(cur_item_freq.unsqueeze(0))
        bi = self.bundle_item_dense_tensor[cand_idx_batch.flatten()]
        nex_item_freq = cur_item_freq.repeat(bi.shape[0], 1) + bi
        nex_ent = self.get_entropy(nex_item_freq)
        delta_bundle_ent = nex_ent - cur_ent.repeat(bi.shape[0])
        delta_bundle_ent /= self.max_ent

        cur_cov = self.get_coverage(cur_item_freq.unsqueeze(0))
        nex_cov = self.get_coverage(nex_item_freq)
        delta_bundle_cov = nex_cov - cur_cov.repeat(bi.shape[0])
        return delta_bundle_ent.reshape(
            cand_idx_batch.shape), delta_bundle_cov.reshape(
            cand_idx_batch.shape)

    def get_entropy(self, item_freq):
        """
        Compute entropy
        """
        prob = item_freq / item_freq.sum(dim=1).unsqueeze(1)
        ent = -prob * torch.log2(prob)
        ent = torch.sum(ent, dim=1)
        return ent

    def get_coverage(self, item_freq):
        """
        Compute coverage
        """
        num_nz = (item_freq >= 1).sum(dim=1)
        cov = num_nz / item_freq.shape[1]
        return cov

    def rerank(self, results, ks):
        """
        Reranking algorithm
        """
        cand_scores, cand_idxs = torch.topk(results, dim=1, k=self.n)
        cand_scores_sigmoid = torch.sigmoid(cand_scores)
        cur_item_freq = torch.zeros(self.n_item) + 1e-9
        rec_list = []
        user_batch_size = 10
        adjust = torch.zeros_like(cand_scores_sigmoid)
        for i in range(1, max(ks)+1):
            user_idx = list(range(cand_scores_sigmoid.shape[0]))
            np.random.shuffle(user_idx)
            rec_list_one = torch.zeros(len(user_idx)).long()
            for batch_idx, start_idx in tqdm(enumerate(range(0, len(user_idx), user_batch_size))):
                end_idx = min(start_idx + user_batch_size, len(user_idx))
                u_batch = user_idx[start_idx:end_idx]
                cand_score_batch = cand_scores_sigmoid[u_batch]
                cand_idxs_batch = cand_idxs[u_batch]
                adjust_batch = adjust[u_batch]
                cand_div_ent_batch, cand_div_cov_batch = self.delta_bundle_batch(cur_item_freq, cand_idxs_batch)
                cand_score_batch_scaled = torch.pow(cand_score_batch, self.beta)
                total_score_batch = cand_score_batch_scaled +\
                                    (1 - cand_score_batch_scaled) * (cand_div_ent_batch + cand_div_cov_batch) +\
                                    adjust_batch
                rec_idx_rel = torch.argmax(total_score_batch, axis=1).unsqueeze(1)
                rec_idx_org = torch.gather(cand_idxs_batch, dim=1, index=rec_idx_rel)
                freq_gain = self.bundle_item[rec_idx_org.squeeze()].sum(0)
                cur_item_freq += torch.tensor(freq_gain).squeeze()
                adjust[u_batch, rec_idx_rel.squeeze()] = -np.inf
                rec_list_one[u_batch] = rec_idx_org.squeeze()
            rec_list.append(rec_list_one.unsqueeze(1))
        rec_list = torch.cat(rec_list, dim=1)
        return rec_list

    def evaluate_test(self, results, ks, div=True):
        """
        Evaluate the results
        """
        rec_list = self.rerank(results, ks)
        recall_list, map_list, freq_list = [], [], []
        user_idx, _ = np.nonzero(np.sum(self.user_bundle_test, 1))
        test_pos_idx = np.nonzero(self.user_bundle_test[user_idx].toarray())[1]
        pos_idx = torch.LongTensor(test_pos_idx).unsqueeze(1)
        batch_size = 1000
        for batch_idx, start_idx in tqdm(enumerate(range(0, rec_list.shape[0], batch_size))):
            end_idx = min(start_idx + batch_size, rec_list.shape[0])
            result = rec_list[start_idx:end_idx]
            pos = pos_idx[start_idx:end_idx]
            recalls, maps, freqs = evaluate_metrics(result, pos, self.bundle_item, ks=ks, div=True, score=False)
            recall_list.append(recalls)
            map_list.append(maps)
            freq_list.append(freqs)
        recalls = list(np.array(recall_list).sum(axis=0) / len(user_idx))
        maps = list(np.array(map_list).sum(axis=0) / len(user_idx))
        freqs = torch.stack(freq_list).sum(dim=0)
        covs, ents, ginis = evaluate_diversities(freqs, div=div)
        return recalls, maps, covs, ents, ginis


class Origin(object):
    """
    Class of returning original results
    """
    def __init__(self):
        super(Origin, self).__init__()

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

    def evaluate_test(self, results, ks, div=True):
        """
        Evaluate the results
        """
        user_idx, _ = np.nonzero(np.sum(self.user_bundle_test, 1))
        test_pos_idx = np.nonzero(self.user_bundle_test[user_idx].toarray())[1]
        pred_ranks = torch.topk(results, max(ks), dim=1, sorted=True)[1]
        recalls, maps, freqs = [], [], []
        for k in ks:
            pred_rank = pred_ranks[:, :k]
            recall, map, freq = get_metrics(pred_rank, torch.LongTensor(test_pos_idx).unsqueeze(1), k, self.bundle_item, div=div)
            recalls.append(recall)
            maps.append(map)
            freqs.append(freq)
        recalls = np.array(recalls) / user_idx.shape[0]
        maps = np.array(maps) / user_idx.shape[0]
        covs, ents, ginis = evaluate_diversities(torch.stack(freqs), div=div)
        return recalls, maps, covs, ents, ginis

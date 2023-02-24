import time

import click

from reranking_models import *
from utils import *


def load_mat_dataset(dataname):
    """
    Load dataset
    """
    path = f'../data/{dataname}'
    user_bundle_trn = load_obj(f'{path}/train.pkl')
    user_bundle_vld = load_obj(f'{path}/valid.pkl')
    user_bundle_test = load_obj(f'{path}/test.pkl')
    user_item = load_obj(f'{path}/user_item.pkl')
    bundle_item = load_obj(f'{path}/bundle_item.pkl')
    user_bundle_neg = np.array(load_obj(f'{path}/neg.pkl'))
    n_user, n_item = user_item.shape
    n_bundle, _ = bundle_item.shape

    user_bundle_test_mask = user_bundle_trn + user_bundle_vld

    # filtering
    user_bundle_vld, vld_user_idx = user_filtering(user_bundle_vld,
                                                   user_bundle_neg)

    return n_user, n_item, n_bundle, bundle_item, user_item,\
           user_bundle_trn, user_bundle_vld, vld_user_idx, user_bundle_test,\
           user_bundle_test_mask


def user_filtering(csr, neg):
    """
    Aggregate ground-truth targets and negative targets
    """
    idx, _ = np.nonzero(np.sum(csr, 1))
    pos = np.nonzero(csr[idx].toarray())[1]
    pos = pos[:, np.newaxis]
    neg = neg[idx]
    arr = np.concatenate((pos, neg), axis=1)
    return arr, idx


@click.command()
@click.option('--data', type=str, default='steam')
@click.option('--base', type=str, default='dam')
@click.option('--model', type=str, default='popcon')
@click.option('--beta', type=float, default=10000)
@click.option('--n', type=int, default=100)
@click.option('--seed', type=int, default=0)
def main(data, base, model, beta, n, seed):
    """
    Main function
    """
    set_seed(seed)
    n_user, n_item, n_bundle, bundle_item, user_item,\
    user_bundle_trn, user_bundle_vld, vld_user_idx, user_bundle_test,\
    user_bundle_test_mask = load_mat_dataset(data)
    ks = [1, 3, 5]
    result_path = f'../out/{data}/{base}_results.pt'
    results = torch.load(result_path).to('cpu')

    print('=========================== LOADED ===========================')

    if model == 'origin':
        model = Origin()
        model.get_dataset(n_user, n_item, n_bundle, bundle_item, user_item,
                          user_bundle_trn, user_bundle_vld, vld_user_idx,
                          user_bundle_test, user_bundle_test_mask)

    elif model == 'popcon':
        model = PopCon(beta=beta, n=n)
        model.get_dataset(n_user, n_item, n_bundle, bundle_item, user_item,
                          user_bundle_trn, user_bundle_vld, vld_user_idx,
                          user_bundle_test, user_bundle_test_mask)

    test_start_time = time.time()
    test_recalls, test_maps, test_covs, test_ents, test_ginis = model.evaluate_test(results, ks, div=True)
    test_elapsed = time.time() - test_start_time

    test_content = form_content(0, [0, 0],
                                test_recalls, test_maps, test_covs, test_ents, test_ginis,
                                [0, test_elapsed, test_elapsed])
    print(test_content)


def form_content(epoch, losses, recalls, maps, covs, ents, ginis, elapses):
    """
    Format of logs
    """
    content = f'{epoch:7d}| {losses[0]:10.4f} {losses[1]:10.4f} |'
    for item in recalls:
        content += f' {item:.4f} '
    content += '|'
    for item in maps:
        content += f' {item:.4f} '
    content += '|'
    for item in covs:
        content += f' {item:.4f} '
    content += '|'
    for item in ents:
        content += f' {item:.4f} '
    content += '|'
    for item in ginis:
        content += f' {item:.4f} '
    content += f'| {elapses[0]:7.1f} {elapses[1]:7.1f} {elapses[2]:7.1f} |'
    return content


if __name__ == '__main__':
    main()

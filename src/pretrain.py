import time

import click

from models import *
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
@click.option('--seed', type=int, default=0)
@click.option('--epochs', type=int, default=200)
@click.option('--alpha', type=float, default=0.1)
def main(data, base, seed, epochs, alpha):
    """
    Main function
    """
    set_seed(seed)
    n_user, n_item, n_bundle, bundle_item, user_item,\
    user_bundle_trn, user_bundle_vld, vld_user_idx, user_bundle_test,\
    user_bundle_test_mask = load_mat_dataset(data)
    ks = [1, 3, 5]
    config = {
        'alpha': alpha,
        'n_item': n_item,
        'n_user': n_user,
        'n_bundle': n_bundle,
        'lr': 1e-3,
        'decay': 1e-5,
        'batch_size': 1000,
        'emb_dim': 20}

    result_filtered_path = f'../out/{data}/{base}_results.pt'
    model_path = f'../out/{data}/{base}_model.pt'
    model = None

    if base == 'dam':
        model = Dam(**config)
        model.get_dataset(n_user, n_item, n_bundle, bundle_item, user_item,
                          user_bundle_trn, user_bundle_vld, vld_user_idx,
                          user_bundle_test, user_bundle_test_mask)

    ks_str = ','.join(f'{k:2d}' for k in ks)
    header = f' Epoch |         losses        |     Recall@{ks_str}    |' \
             f'       MAP@{ks_str}     |          elapse         |'
    print(header)
    start_time = time.time()
    best_vld_acc, best_vld_content, best_test_content = 0., '', ''
    for epoch in range(1, epochs+1):
        trn_start_time = time.time()
        model = model.to(TRN_DEVICE)
        trn_rec_loss, trn_div_loss = model.update_model()
        epoch_trn_elapsed = time.time() - trn_start_time
        eval_start_time = time.time()
        model = model.to(EVA_DEVICE)
        vld_recalls, vld_maps = model.evaluate_val(ks, div=False)
        epoch_eval_elapsed = time.time() - eval_start_time
        total_elapsed = time.time() - start_time
        vld_content = form_content(epoch, [trn_rec_loss, 0],
                               vld_recalls, vld_maps,
                               [epoch_trn_elapsed, epoch_eval_elapsed, total_elapsed])

        if vld_maps[0] > best_vld_acc:
            best_vld_acc = vld_maps[0]
            torch.save(model.state_dict(), model_path)
            best_vld_content = vld_content

        if epoch % 1 == 0:
            print(vld_content)

        if epoch % 20 == 0:
            print('============================ BEST ============================')
            print(best_vld_content)
            print('=================================================================')

    test_start_time = time.time()
    test_recalls, test_maps, ubs_origin, ubs_filtered = model.evaluate_test(ks, div=True)
    test_elapsed = time.time() - test_start_time
    test_content = form_content(0, [0, 0],
                                test_recalls, test_maps,
                                [0, test_elapsed, test_elapsed])
    print(test_content)

    torch.save(ubs_filtered, result_filtered_path)


def form_content(epoch, losses, recalls, maps, elapses):
    """
    Format of logs
    """
    content = f'{epoch:7d}| {losses[0]:10.4f} {losses[1]:10.4f} |'
    for item in recalls:
        content += f' {item:.4f} '
    content += '|'
    for item in maps:
        content += f' {item:.4f} '
    content += f'| {elapses[0]:7.1f} {elapses[1]:7.1f} {elapses[2]:7.1f} |'
    return content


if __name__ == '__main__':
    main()

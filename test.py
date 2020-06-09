# import itertools
from scipy import spatial
from datasets import *
# import matplotlib.pyplot as plt
from utils_ import *
from architectures import *
import numpy_indexed as npi


become_deterministic(0)
transformation = transform_AMOS_resize
# transformation = transform_AMOS_resize48
# transformation = transform_AMOS_resize64

def tpfp(scores, labels, numpos=None): # code from HPatches
    # count labels
    p = int(np.sum(labels))
    n = len(labels) - p

    if numpos is not None:
        assert (numpos >= p), \
            'numpos smaller that number of positives in labels'
        extra_pos = numpos - p
        p = numpos
        scores = np.hstack((scores, np.repeat(-np.inf, extra_pos)))
        labels = np.hstack((labels, np.repeat(1, extra_pos)))

    perm = np.argsort(-scores, kind='mergesort', axis=0)

    scores = scores[perm]
    # assume that data with -INF score is never retrieved
    stop = np.max(np.where(scores > -np.inf))

    perm = perm[0:stop + 1]

    labels = labels[perm]
    # accumulate true positives and false positives by scores
    tp = np.hstack((0, np.cumsum(labels == 1)))
    fp = np.hstack((0, np.cumsum(labels == 0)))
    return tp, fp, p, n, perm

def get_pr(scores, labels, numpos=None): # code from HPatches
    [tp, fp, p, n, perm] = tpfp(scores, labels, numpos)
    # compute precision and recall
    small = 1e-10
    recall = tp / float(np.maximum(p, small))
    precision = np.maximum(tp, small) / np.maximum(tp + fp, small)
    return precision, recall, np.trapz(precision, recall)

# def test_pair(query, target):
#     dists = spatial.distance.cdist(query, target,'euclidean')
#     idxs = np.argmin(dists, axis=1)
#     gt = np.arange(dists.shape[0])
#     right = np.equal(idxs, gt)
#
#     # m_d = dists[gt, idxs]
#     # pr,rc,ap = get_pr(-m_d,right,numpos=right.shape[0])
#     return right

def test_pair_(query, target, q_centers=None, t_centers=None, collisions=None, second_nearest=False):
    dists = spatial.distance.cdist(query, target,'euclidean')
    dists_centers = spatial.distance.cdist(q_centers, t_centers,'euclidean')
    aux = np.min(dists_centers, axis=1)
    close = aux<2.0
    # print('kept', np.sum(aux<2.0), 'removed', np.sum(aux>=2.0))
    dists = dists[close]
    dists_centers = dists_centers[close]

    idxs = np.argmin(dists, axis=1)
    gt = np.argmin(dists_centers, axis=1)
    right = idxs==gt
    if second_nearest:
        dists1 = np.min(dists, axis=1)
        for i in range(len(query)): # remove the absolutely nearest ones
            dists[i,np.argmin(dists[i,:])] = sys.maxsize
        for i,a in enumerate(collisions): # remove the inconsistent ones
            dists[i, a] = sys.maxsize
        dists2 = np.min(dists, axis=1)
        # if np.sum(dists2==0)>0:
        #     print('second with dist=0 found (!)')
            # aux = np.argmin(dists[np.argmin(dists2),:])
            # print(aux)
            # print(np.argmin(dists2))

            # print(dists[np.argmin(dists2),:])
            # print(dists[np.argmin(dists2),aux-1])
            # print(dists[np.argmin(dists2),aux])
            # print(dists[np.argmin(dists2),aux+1])
            # input()
        dists2[dists2==0] = 0.000001 # but this is hack, investigate WHY DISTANCE IS ZERO
        right = right * ((dists1 / dists2) < 0.8)

    m_d = dists[np.arange(len(query)), idxs]
    pr,rc,ap = get_pr(-m_d,right,numpos=len(query))
    return right, close, ap

def run_matching(amos, model, file_out, max_imgs=10, second_nearest=False, bsize=2000):
    model.eval()
    printc.green('processing patches ...')
    descs = get_descs(model, amos.patch_sets, bsize=bsize)
    precs = []
    losses = np.zeros(descs.shape[:2])
    counts = np.zeros(descs.shape[:2])
    if amos.patch_sets.shape[1]==1: # this means one image per folder #TODO change to mAP
        printc.red('running split version')
        view_names = [c.split('-')[0] for c in amos.data['view_names']]
        gb = npi.group_by(view_names)
        idx_cams = gb.split_array_as_list(np.arange(len(view_names)))
        gb = npi.group_by(amos.cam_idxs)
        all_idxs = gb.split_array_as_list(np.arange(len(amos.patch_sets)))

        printc.green('evaluating ...')

        rights_all = []
        for i, cur_cams in enumerate(idx_cams):
            cur_cams = cur_cams[:max_imgs]
            combs = list(itertools.permutations(np.array(cur_cams), 2)) # includes (a,b), (b,a)
            # combs = list(itertools.combinations(aux, 2)) # only (a,b)
            rights = []
            for c in tqdm(combs, desc='running pairs'):
                # print(descs[all_idxs[c[0]],:].shape) # (2237, 1, 128)
                q_centers = amos.data['LAFs'].data.cpu().numpy()[all_idxs[c[0]]][:,:,2]
                t_centers = amos.data['LAFs'].data.cpu().numpy()[all_idxs[c[1]]][:,:,2]
                right, mask, AP = test_pair_(descs[all_idxs[c[0]],0], descs[all_idxs[c[1]],0], q_centers, t_centers, second_nearest=second_nearest)
                rights += [right]
                losses[all_idxs[c[0]][mask],0] += (1-rights[-1])
                counts[all_idxs[c[0]][mask],0] += 1

            precs += [np.mean(np.concatenate(rights))]
            rights_all += rights
            print(amos.data['view_names'][i], 'prec= {:.2f}'.format(precs[-1] * 100))
            print(amos.data['view_names'][i], 'prec= {:.2f}'.format(precs[-1] * 100), file=file_out)

        # printc.green('overall prec={:.6f}'.format(np.mean(np.array(precs))) )
        rights_all = np.concatenate(rights_all)
        print('kept',len(rights_all),'from',amos.patch_sets.shape[0],'->', 100.0*len(rights_all)/amos.patch_sets.shape[0],'%')
        # print(np.concatenate(rights_all).shape)
        print('overall prec={:.6f}'.format(np.mean(rights_all)))
        print('overall prec all={:.6f}'.format(np.sum(rights_all) / amos.patch_sets.shape[0]))
        print('overall prec={:.6f}'.format(np.mean(rights_all)), file=file_out)
        print('overall prec all={:.6f}'.format(np.sum(rights_all) / amos.patch_sets.shape[0]), file=file_out)

        out = {}
        out['losses'] = losses
        out['counts'] = counts
        out['data_path'] = amos.data_path
        out['type'] = 'matching'
        return out

    printc.red('running standard version')
    gb = npi.group_by(amos.cam_idxs)
    all_idxs = gb.split_array_as_list(np.arange(len(amos.patch_sets)))

    printc.green('evaluating ...')

    APs = []
    for i, idxs in enumerate(all_idxs):
        desc = descs[idxs][:,:max_imgs] # descs for one cam
        aux = np.arange(desc.shape[1])
        combs = list(itertools.permutations(aux, 2)) # includes (a,b), (b,a)
        # combs = list(itertools.combinations(aux, 2)) # only (a,b)
        rights = []
        q_centers = amos.data['LAFs'].data.cpu().numpy()[idxs][:,:,2]
        t_centers = amos.data['LAFs'].data.cpu().numpy()[idxs][:,:,2]
        oneAP = []
        for c in tqdm(combs, desc='running pairs'):
            colls = [np.array(amos.collisions[c])-idxs[0] for c in idxs] ### this should correct indices according to offset, we want idxs to current set
            # colls = [[list(idxs).index(c) for c in coll] for coll in colls] # maybe slow
            right, mask, AP = test_pair_(query=desc[:,c[0]], target=desc[:,c[1]], q_centers=q_centers, t_centers=t_centers, collisions=colls, second_nearest=second_nearest)
            oneAP += [AP]
            rights += [right]
            losses[idxs,c[0]] += (1-rights[-1])
            counts[idxs,c[0]] += 1

        APs += oneAP
        precs += [np.mean(np.concatenate(rights))]
        print(amos.data['view_names'][i], 'correct rate= {:.2f}%'.format(precs[-1] * 100.0))
        print(amos.data['view_names'][i], 'correct rate= {:.2f}%'.format(precs[-1] * 100.0), file=file_out)
        print(amos.data['view_names'][i], 'avg prec= {:.2f}%'.format(100.0 * np.mean(np.array(oneAP))))
        print(amos.data['view_names'][i], 'avg prec= {:.2f}%'.format(100.0 * np.mean(np.array(oneAP))), file=file_out)

    printc.green('mean correct rate={:.6f}%'.format(100.0*np.mean(np.array(precs))) )
    print('mean correct rate={:.6f}%'.format(100.0*np.mean(np.array(precs))), file=file_out)

    APs = np.array(APs)
    printc.green('mAP={:.6f}%'.format(100*np.mean(APs)) )
    print('mAP={:.6f}%'.format(100*np.mean(APs)), file=file_out)

    out = {}
    out['losses'] = losses
    out['counts'] = counts
    out['data_path'] = amos.data_path
    out['type'] = 'matching'
    return out

def get_3_fcs(descs, cam_idxs, npts=100):
    all_idxs = list(np.arange(descs.shape[0]))
    ps_idxs = random.sample(all_idxs, npts)
    set_idxs = [random.choice(list(np.arange(descs.shape[1]))) for _ in range(npts)]

    gb = npi.group_by(list(cam_idxs))
    idxs_per_cam = [list(c) for c in gb.split_array_as_list(all_idxs)]
    a,b,c,d = [],[],[],[]
    ea,eb,ec = [],[],[]
    for ps_idx, set_idx in tqdm(zip(ps_idxs, set_idxs), desc='Running queries', total=npts):
        pos_idx = list(range(descs.shape[1]))
        pos_idx.remove(set_idx)
        pos_idx = random.choice(pos_idx)
        other_set = list(range(descs.shape[1]))
        other_set.remove(set_idx)
        other_set.remove(pos_idx)
        cam_idx = cam_idxs[list(np.arange(descs.shape[0])).index(ps_idx)]
        in_cam_idxs = copy(idxs_per_cam[cam_idx])
        in_cam_idxs.remove(ps_idx)
        out_cam_idxs = list(set(all_idxs).difference(set(idxs_per_cam[cam_idx])))

        # other_idxs = list(np.arange(descs.shape[0]))
        # other_idxs.remove(ps_idx)

        query_desc = np.expand_dims(descs[ps_idx, set_idx], 0)
        descs_img = descs[in_cam_idxs, pos_idx] # is only one
        aux = descs[in_cam_idxs][other_set]
        descs_cam = np.reshape(aux, (-1, descs.shape[-1]))
        descs_other = np.reshape(descs[out_cam_idxs, :], (-1, descs.shape[-1]))
        descs_pos = np.expand_dims(descs[ps_idx, pos_idx], 0)

        a += [find_nearest(query_desc, descs_img)]
        b += [find_nearest(query_desc, descs_cam)]
        c += [find_nearest(query_desc, descs_other)]
        d += [find_nearest(query_desc, descs_pos)]

        ea += [d[-1]-a[-1]]
        eb += [d[-1]-b[-1]]
        ec += [d[-1]-c[-1]]
    return np.array((a,b,c,d)), np.array((ea,eb,ec))

def find_nearest(descs_query, descs_target):
    dists = spatial.distance.cdist(descs_query, descs_target, 'euclidean')
    return np.amin(dists, axis=1)[0]

def find_mean(descs_query, descs_target):
    dists = spatial.distance.cdist(descs_query, descs_target, 'euclidean')
    return np.mean(dists)

def fce(p):
    p = torch.from_numpy(p).float() # numpy -> tensor -> numpy, because pool.map on huge tensor would fail on "too many open files"
    return transformation(p).data.cpu().numpy()

def get_descs(model, patch_sets, bsize=2000):
    print('get_descs function begin')
    patches = patch_sets.view(-1, patch_sets.shape[-3], patch_sets.shape[-2], patch_sets.shape[-1])

    pool = multiprocessing.Pool(10)
    # torch.multiprocessing.set_sharing_strategy('file_system')
    inputs = list(tqdm(pool.imap(fce, patches.data.cpu().numpy()), total=len(patches), desc='Transforming patches'))
    # inputs = []
    # for c in tqdm(patches.data.cpu().numpy()):
    #     inputs += [fce(c)]
    printc.green('stacking ...')
    inputs = torch.from_numpy(np.stack(inputs)).float()
    printc.green('finished')
    sys.stdout.flush()

    idxs = np.arange(len(inputs))
    splits = np.array_split(idxs, max(1, (patches.shape[0] // bsize) ** 2))
    preds = []
    printc.green('finished')
    sys.stdout.flush()
    with torch.no_grad():
        for spl in tqdm(splits, desc='Getting descriptors'):
            preds += [model(inputs[spl].cuda()).data.cpu().numpy()]
    preds = np.concatenate(preds)
    preds = np.reshape(preds, (patch_sets.shape[0], patch_sets.shape[1], -1))
    print('get_descs function end')
    return preds

def get_avg_dist(descs, npts=100):
    all_idxs = list(np.arange(descs.shape[0]))
    Aps_idxs = random.sample(all_idxs, npts)
    Aset_idxs = [random.choice(list(np.arange(descs.shape[1]))) for _ in range(npts)]
    Bps_idxs = random.sample(all_idxs, npts)
    Bset_idxs = [random.choice(list(np.arange(descs.shape[1]))) for _ in range(npts)]
    obs = []
    for Aps_idx, Aset_idx, Bps_idx, Bset_idx in tqdm(zip(Aps_idxs, Aset_idxs, Bps_idxs, Bset_idxs), desc='Running queries', total=npts):
        query_desc = np.expand_dims(descs[Aps_idx, Aset_idx], 0)
        target_desc = np.expand_dims(descs[Bps_idx, Bset_idx], 0)
        obs += [find_nearest(query_desc, target_desc)]
    return obs

def get_amos(data_path, AMOS_RGB=False, depths='', only_D=False):
    return AMOS_dataset(transform=transformation,
                        data_path=data_path,
                        Npositives=1,
                        AMOS_RGB=AMOS_RGB,
                        depths=depths,
                        only_D=only_D,
                        use_collisions=True,
                        )

def data_from_type(type):
    if type in ['AMOS-views-v4_pairs-match']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4-pairs/AMOS-views-v4-pairs_PS:0_WF:uniform_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:60000_WF:Hessian_PG:meanImg_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4_hess_fair']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:0_WF:Hessian_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4_uni']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:60000_WF:uniform_PG:meanImg_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4_uni_fair']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:0_WF:uniform_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-views-v4_uni_fair_mini']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_PS:0_WF:uniform_PG:meanImg_minsets:100_masks:AMOS-masks.pt'
    elif type in ['AMOS-test-1']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1/AMOS-test-1_PS:0_WF:uniform_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-test-1-pairs']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1_pairs/AMOS-test-1_pairs_PS:0_WF:uniform_PG:meanImg_minsets:1000_masks:AMOS-masks.pt'
    elif type in ['AMOS-test-1-new']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1/AMOS-test-1_maxsets:1000_WF:Hessian_PG:new_masks:AMOS-masks.pt'

    elif type in ['sift']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1-downsized/AMOS-test-1-downsized_WF:Hessian_PG:sift_masks:AMOS-masks.pt'
    elif type in ['sift-split']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1-downsized-split/AMOS-test-1-downsized-split_WF:Hessian_PG:sift_masks:AMOS-masks.pt'
    elif type in ['sift-RGB']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1-downsized/AMOS-test-1-downsized_WF:Hessian_PG:sift_RGB_masks:AMOS-masks.pt'
    elif type in ['sift-D']:
        data_path = 'Datasets/AMOS-views/AMOS-test-1-downsized/AMOS-test-1-downsized_WF:Hessian_PG:sift_RGB_depths_masks:AMOS-masks.pt'
    elif type in ['full']:
        data_path = 'Datasets/AMOS-views/AMOS-views-v4/AMOS-views-v4_maxsets:2000_sigmas-v:v14_thr:0.00016_WF:Hessian_PG:new_depths_masks:AMOS-masks.pt'
    else: assert False, 'invalid test type'
    return data_path

class Interface:
    def match(self,
              # model_name='id:103_arch:h1_ds:v4_loss:tripletMargin_mpos:1.0_mneg:1.0_PS:60000_WF:uniform_PG:meanImg_tps:20000000_CamsB:5_masks_ep:1',
                model_name='id:213_arch:h1_ds:v4_loss:tripletMargin_mpos:1.0_mneg:1.0_maxsets:2000_sigmas-v:v14_thr:0.00016_WF:Hessian_PG:new_depths_masks:AMOS-masks_tps:5000000_CamsB:5_ep:10',
                # type='AMOS-views-v4_uni_fair',
                # type='AMOS-test-1-new',
                type='sift',
                SN=False,
                only_D=False,
                bs=2000,
                ):
        printc.yellow('\n'.join(['Input arguments:'] + [str(x) for x in sorted(locals().items()) if x[0] != 'self']))
        model = load_hardnet(model_name)
        path_out = os.path.join('Models', model_name, 'Matching', type+'.txt')
        os.makedirs(os.path.dirname(path_out), exist_ok=True)
        if type in ['sift-D']: # channels must be averaged
            amos = get_amos(data_from_type(type), AMOS_RGB=False, depths='dummy', only_D=only_D)
        else:
            amos = get_amos(data_from_type(type), AMOS_RGB=False, only_D=only_D)
        out = run_matching(amos, model, open(path_out, 'w'), second_nearest=SN, bsize=bs)
        np.save(os.path.join('Models', model_name, 'Matching/info_{}_matching.npy'.format(type)), out)

    def three_fcs(self,
                  model_name='id:102_ds:v4_loss:tripletMargin_mpos:1.0_mneg:1.0_PS:60000_WF:Hessian_PG:meanImg_tps:20000000_CamsB:5_masks_ep:1',
                  type='AMOS-views-v4_uni_fair_mini',
                  ):
        model = load_hardnet(model_name)
        amos = get_amos(data_from_type(type))
        descs = get_descs(model, amos.patch_sets)
        r, e = get_3_fcs(descs, amos.cam_idxs.long(), 100)
        a = np.argsort(r[0])
        r = [r[0][a], r[1][a], r[2][a], r[3][a]]
        a = np.argsort(e[0])
        e = [e[0][a], e[1][a], e[2][a]]

        fig = plt.figure(figsize=(20, 10))
        plt.plot(r[0])
        plt.plot(r[1])
        plt.plot(r[2])
        # plt.plot(res[3], linestyle='dotted')
        plt.plot(r[3], 'o')
        plt.legend(['in image', 'in view', 'other views', 'positives'])
        plt.xlabel('point')
        plt.ylabel('distance')
        plt.title('distances')
        dir_out = os.path.join('Models', model_name, 'Graphs')
        os.makedirs(dir_out, exist_ok=True)
        fig.savefig(os.path.join(dir_out, '_'.join([type, 'dists.png'])), dpi=fig.dpi)

        fig = plt.figure(figsize=(20, 10))
        plt.plot(e[0])
        plt.plot(e[1])
        plt.plot(e[2])
        plt.legend(['in image', 'in view', 'other views'])
        plt.xlabel('point')
        plt.ylabel('edge')
        plt.title('edges')
        dir_out = os.path.join('Models', model_name, 'Graphs')
        os.makedirs(dir_out, exist_ok=True)
        fig.savefig(os.path.join(dir_out, '_'.join([type, 'edges.png'])), dpi=fig.dpi)

    def avg_dist(self,
                  model_name='id:103_arch:h1_ds:v4_loss:tripletMargin_mpos:1.0_mneg:1.0_PS:60000_WF:uniform_PG:meanImg_tps:20000000_CamsB:5_masks_ep:1',
                  type='AMOS-views-v4_uni_fair',
                 ):
        model = load_hardnet(model_name)
        amos = get_amos(data_from_type(type))
        descs = get_descs(model, amos.patch_sets)
        res = get_avg_dist(descs, 1000)
        print('avg dist:', np.mean(np.array(res)))
        print('min dist:', np.min(np.array(res)))
        print('max dist:', np.max(np.array(res)))


if __name__ == "__main__":
    Fire(Interface)
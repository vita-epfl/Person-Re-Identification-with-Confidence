from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import os
import os.path as osp
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .iotools import mkdir_if_missing
from collections import defaultdict
from sortedcontainers import SortedDict

import pickle
from .tsne import tsne_wrapper as tsne
from collections import Counter
import operator
from matplotlib import colors as mcolors

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import pdb

def load_pickle(path):
  """Check and load pickle object.
  According to this post: https://stackoverflow.com/a/41733927, cPickle and
  disabling garbage collector helps with loading speed."""
  assert osp.exists(path)
  # gc.disable()
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  # gc.enable()
  return ret

def save_pickle(obj, path):
  """Create dir and save file."""
  mkdir_if_missing(osp.dirname(osp.abspath(path)))
  with open(path, 'wb') as f:
    pickle.dump(obj, f, protocol=2)

def drawLineGraph(dict_mAP_cmc,src, file_name, title='',ylabel =''):
    ind = list(dict_mAP_cmc.keys())
    delaTheta = list(dict_mAP_cmc.values())

    fig = plt.figure(figsize=(25,8))
    ax = fig.add_subplot(111)

    line1 = ax.plot(ind, delaTheta, label='delaTheta', color='c', marker='o')

    ax.set_ylabel(ylabel)

    plt.xticks(ind)
    plt.xlabel('Delta Rotation')

    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(1.05,1))
    ax.grid('on')

    for x, deltaRot in zip(ind,delaTheta):
        ax.annotate('%.2f'%(deltaRot), (x,deltaRot))
    plt.title(title, pad=25)
    plt.savefig(osp.join(src,file_name))

def plot_deltaTheta(distmat, dataset,save_dir='log/deltaTheta_results', min_rank=1):
    if isinstance(distmat, tuple):
        distmat = distmat[0]
    num_q, num_g = distmat.shape
    root_angle = 45
    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    num_q, num_g = distmat.shape

    print("Plotting D(Theta) with min ranks {}".format(min_rank))
    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Saving plot to '{}'".format(save_dir))

    assert num_q == len(dataset.query)
    assert num_g == len(dataset.gallery)

    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    dict_delta = defaultdict(int)
    dict_sum = defaultdict(int)
    dict_imgs = defaultdict(list)


    for q_idx in range(num_q):

        qimg_path, qpid, qrotation = dataset.query[q_idx]


        min_delta_rot = 360
        min_delta_rank = min_rank
        rank_idx = 1
        for g_idx in indices[q_idx,:]:

            gimg_path, gpid, grotation = dataset.gallery[g_idx]
            invalid = (qpid == gpid) & (qrotation == grotation)

            if not invalid :
                if qpid == gpid and rank_idx >min_rank:
                    sub_rot = abs(qrotation*root_angle - grotation*root_angle)
                    delta_rot = 360 - sub_rot  if sub_rot >180 else sub_rot

                    if delta_rot < min_delta_rot:
                        min_delta_rot = delta_rot
                        min_delta_rank = rank_idx
                        min_gimg = gimg_path
                        min_gidx = g_idx
                elif qpid == gpid:
                    break
                rank_idx += 1
        if min_delta_rank >min_rank:
            dict_delta[min_delta_rot] += 1
            dict_sum[min_delta_rot] += distmat[q_idx, min_gidx]
            qdir = osp.join(save_dir,'Delta_Rot_{}'.format(str(min_delta_rot)),str(min_delta_rank)+'__'+ os.path.splitext(osp.basename(qimg_path))[0])
            mkdir_if_missing(qdir)
            #dict_imgs[min_delta_rot].append((qimg_path,min_gimg))
            _cp_img_to(qimg_path, qdir, rank=0, prefix='query')
            _cp_img_to(min_gimg, qdir, rank=min_delta_rank, prefix='gallery')

    sorted_dict_delta = SortedDict(dict_delta)
    drawLineGraph(sorted_dict_delta,save_dir,'delta_rot_minRank_{}.png'.format(min_rank), 'Number of errors vs Delta Rotation', ylabel='Error Number')

    dict_mean = defaultdict(int)

    for key in dict_delta.keys():
        dict_mean[key] = dict_sum[key]/dict_delta[key]
    sorted_dict_mean = SortedDict(dict_mean)
    drawLineGraph(sorted_dict_mean,save_dir,'delta_mean_minRank_{}.png'.format(min_rank), 'Mean distance vs Delta Rotation', ylabel='Mean distance')

    save_pickle(sorted_dict_mean,osp.join(save_dir,'delta_mean_minRank_{}.pickle'.format(min_rank)))
    save_pickle(sorted_dict_delta,osp.join(save_dir,'delta_rot_minRank_{}.pickle'.format(min_rank)))


def visualize_ranked_results(distmat, dataset, save_dir='log/ranked_results', topk=20):
    """
    Visualize ranked results

    Support both imgreid and vidreid

    Args:
    - distmat: distance matrix of shape (num_query, num_gallery).
    - dataset: has dataset.query and dataset.gallery, both are lists of (img_path, pid, camid);
               for imgreid, img_path is a string, while for vidreid, img_path is a tuple containing
               a sequence of strings.
    - save_dir: directory to save output images.
    - topk: int, denoting top-k images in the rank list to be visualized.
    """
    rot_dict = {}
    if isinstance(distmat, tuple):
        rot_dict = distmat[1]
        distmat = distmat[0]
    num_q, num_g = distmat.shape

    print("Visualizing top-{} ranks".format(topk))
    print("# query: {}\n# gallery {}".format(num_q, num_g))
    print("Saving images to '{}'".format(save_dir))

    assert num_q == len(dataset.query)
    assert num_g == len(dataset.gallery)

    indices = np.argsort(distmat, axis=1)
    mkdir_if_missing(save_dir)

    def _cp_img_to(src, dst, rank, prefix):
        """
        - src: image path or tuple (for vidreid)
        - dst: target directory
        - rank: int, denoting ranked position, starting from 1
        - prefix: string
        """
        if isinstance(src, tuple) or isinstance(src, list):
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3))
            mkdir_if_missing(dst)
            for img_path in src:
                shutil.copy(img_path, dst)
        else:
            dst = osp.join(dst, prefix + '_top' + str(rank).zfill(3) + '_name_' + osp.basename(src))
            shutil.copy(src, dst)

    for q_idx in range(num_q):
        prefix='query'
        if rot_dict:
            prefix = '0_query_rotp_'+ str(rot_dict['query'][q_idx])
        qimg_path, qpid, qcamid = dataset.query[q_idx]
        qdir = osp.join(save_dir, os.path.splitext(osp.basename(qimg_path))[0])
        qdir_incorrect = osp.join(save_dir,'incorrect', os.path.splitext(osp.basename(qimg_path))[0])
        mkdir_if_missing(qdir)
        _cp_img_to(qimg_path, qdir, rank=0, prefix=prefix)
        incorrect = False
        rank_idx = 1
        for g_idx in indices[q_idx,:]:
            prefix='gallery'
            if rot_dict:
                prefix = str(rank_idx)+'_gallery_rotp_'+ str(rot_dict['gallery'][g_idx])
            gimg_path, gpid, gcamid = dataset.gallery[g_idx]
            invalid = (qpid == gpid) & (qcamid == gcamid)
            if not invalid:
                if rank_idx == 1 and qpid != gpid:
                    incorrect = True
                    mkdir_if_missing(qdir_incorrect)
                    _cp_img_to(qimg_path, qdir_incorrect, rank=0, prefix='0_query')
                _cp_img_to(gimg_path, qdir, rank=rank_idx, prefix=prefix)
                if incorrect:
                    _cp_img_to(gimg_path, qdir_incorrect, rank=rank_idx, prefix=str(rank_idx)+'_' + prefix)
                rank_idx += 1
                if rank_idx > topk:
                    break

    print("Done")


def drawTSNE(query_f,gallery_f,query_pids, gallery_pids, query_camids, gallery_camids,q_imgPath, g_imgPath,num_clusters,src):
    # unique_pids = Counter(words).keys()
    # pid_count = Counter(words).values()
    perplexity = 60
    file_name = "tsne_plot_{}cls_p{}.png".format(num_clusters,perplexity)

    file_name_withImg = "tsne_plot_{}cls_p{}_withImg.png".format(num_clusters,perplexity)
    #pdb.set_trace()
    print("Drawing T-SNE Plot")
    total_pids = np.concatenate((query_pids, gallery_pids))
    #total_f = np.concatenate((query_f, gallery_f))
    #total_camids = np.concatenate((query_camids, gallery_camids))
    counter = Counter(total_pids)
    chosen_pids = sorted(counter.items(), key=operator.itemgetter(1))
    chosen_pids.reverse()
    chosen_pids = dict(chosen_pids[1:num_clusters+1])
    #indx = np.array([True if j in chosen_pids.keys() else False for j in total_pids])

    #pdb.set_trace()
    query_indx = np.array([True if j in chosen_pids.keys() else False for j in query_pids])
    gallery_indx = np.array([True if j in chosen_pids.keys() else False for j in gallery_pids])
    chosen_imgPath = q_imgPath[query_indx]
    #pdb.set_trace()
    total_f = np.concatenate((query_f.numpy()[query_indx], gallery_f.numpy()[gallery_indx]))
    total_pids = np.concatenate((query_pids[query_indx], gallery_pids[gallery_indx]))
    total_camids = np.concatenate((query_camids[query_indx], gallery_camids[gallery_indx]))
    n_clusters = len(list(chosen_pids.keys()))
    ## Calculating TSNE
    label_dict = {k:i for i,k in enumerate(list(chosen_pids.keys()))}
    new_labels = [float(label_dict[j]) for j in total_pids]
    #pdb.set_trace()
    Y = tsne(total_f, no_dims=2, initial_dims=total_f.shape[1], perplexity=perplexity,seed=1)

    cmap = plt.get_cmap('gist_rainbow', n_clusters)

    fig, ax = plt.subplots(figsize=(8,8))
    scatter = ax.scatter(Y[:, 0], Y[:, 1], 20, new_labels,cmap=cmap)#,cmap=cmap, norm=norm)
    ax.set_title('T-SNE for {} different ids'.format(num_clusters), pad=25)

    #ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])

    #cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    # cb = ax2.colorbar()
    # loc = np.arange(0,len(list(chosen_pids.keys())),len(list(chosen_pids.keys())))
    # cb.set_ticks(loc)
    # cb.set_ticklabels(list(map(str, chosen_pids.keys())))
    cbar = plt.colorbar(scatter)
    ticks_locs = (np.arange(n_clusters) + 0.5)*(n_clusters-1)/n_clusters
    cbar.set_ticks(ticks_locs)

    # set tick labels (as before)
    cbar.set_ticklabels(list(chosen_pids.keys()))
    #ax2.set_ylabel('ID', size=12)
    plt.savefig(osp.join(src,file_name))

    plt.close()
    ## Plotting
    #pdb.set_trace()
    fig, ax = plt.subplots(figsize=(8,8))
    # define the colormap
    #cmap = plt.cm.jet
    # extract all colors from the .jet map
    #cmaplist = [cmap(i) for i in range(cmap.N)]
    # force the first color entry to be grey
    # create the new map
    #cmap = cmap.from_list('Custom cmap', cmaplist[:len(list(chosen_pids.keys()))], len(list(chosen_pids.keys())))


    # define the bins and normalize
    #bounds = np.linspace(0,len(list(chosen_pids.keys())),len(list(chosen_pids.keys()))+1)
    #norm = matplotlib.colors.BoundaryNorm(bounds, len(list(chosen_pids.keys())))
    #pdb.set_trace()


    for indx_q, (x0, y0) in enumerate(zip(Y[:len(chosen_imgPath), 0], Y[:len(chosen_imgPath), 1])):
        try:
            image = plt.imread(chosen_imgPath[indx_q])
        except TypeError:
            # Likely already an array...
            pass
        im = OffsetImage(image, zoom=0.3)
        ab = AnnotationBbox(im, (x0, y0), xycoords='data', frameon=False)
        ax.add_artist(ab)
    ax.update_datalim(np.column_stack([Y[:, 0], Y[:, 1]]))
    ax.autoscale()
    scatter = ax.scatter(Y[:, 0], Y[:, 1], 20, new_labels,cmap=cmap)#,cmap=cmap, norm=norm)
    ax.set_title('T-SNE for {} different ids'.format(num_clusters), pad=25)

    #ax2 = fig.add_axes([0.95, 0.1, 0.03, 0.8])

    #cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, spacing='proportional', ticks=bounds, boundaries=bounds, format='%1i')
    # cb = ax2.colorbar()
    # loc = np.arange(0,len(list(chosen_pids.keys())),len(list(chosen_pids.keys())))
    # cb.set_ticks(loc)
    # cb.set_ticklabels(list(map(str, chosen_pids.keys())))
    cbar = plt.colorbar(scatter)
    ticks_locs = (np.arange(n_clusters) + 0.5)*(n_clusters-1)/n_clusters
    cbar.set_ticks(ticks_locs)

    # set tick labels (as before)
    cbar.set_ticklabels(list(chosen_pids.keys()))
    #ax2.set_ylabel('ID', size=12)
    plt.savefig(osp.join(src,file_name_withImg))

    print("T-SNE Plot Drawn")

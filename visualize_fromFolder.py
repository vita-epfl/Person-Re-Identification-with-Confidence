import matplotlib
matplotlib.use("Agg")
import argparse
import sys
import os
from collections import defaultdict
import pickle
import re
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
import pdb
import numpy as np
import time
parser = argparse.ArgumentParser(description='Train image model with cross entropy loss')

parser.add_argument('--root', type=str, default='/data/george-data/Dataset',
                    help="root path to directory to ranked_results")

parser.add_argument('-m', '--market', action='store_true',
                    help='Using Market dataset (Default False)')
parser.add_argument('-ms', '--msmt17', action='store_true',
                    help='Using msmt17 dataset (Default False)')

def may_make_dir(path):
  """
  Args:
    path: a dir, or result of `osp.dirname(osp.abspath(file_path))`
  Note:
    `osp.exists('')` returns `False`, while `osp.exists('.')` returns `True`!
  """
  # This clause has mistakes:
  # if path is None or '':

  if path in [None, '']:
    return
  if not os.path.exists(path):
    os.makedirs(path)

def getRank(sorted_image_list,args):
    if args.market:
        pattern = re.compile(r'(?:gallery?|query?)_top[-\d]+_name[_]+([-\d]+)_c([-\d]+)s([-\d]+)_[-\d]+_[-\d]+.jpg')
        query_id, camera, scene = map(float, pattern.search(sorted_image_list[0]).groups())
    elif args.msmt17:
        pattern = re.compile(r'([-\d]+)_(?:gallery?|query?)_rotp_([-\d]+)_top[-\d]+_name[_]+([-\d]+)_[-\d]+_([-\d]+)_[-\d]+[a-z]+_[-\d]+_[-\d]+[_]*[a-z]*.jpg')
        _, _,query_id, camera = map(float, pattern.search(sorted_image_list[0]).groups())

    for i,image_name in enumerate(sorted_image_list[1:]):
        if args.market:
            if args.without_rotation:
                id, _, _, = map(float, pattern.search(image_name).groups())
            else:
                _, _,id, _, _, = map(float, pattern.search(image_name).groups())
        elif args.msmt17:
            _, _,id, _ = map(float, pattern.search(image_name).groups())
        else:
            if args.without_rotation:
                town, _, id, _, _ = map(float, pattern.search(image_name).groups())
            else:
                _,_,town, _, id, _, _ = map(float, pattern.search(image_name).groups())
            id = town*100+id

        if query_id == id:
            return i +1
    return 1000

def save_pickle(obj, path):
  """Create dir and save file."""
  may_make_dir(os.path.dirname(os.path.abspath(path)))
  with open(path, 'wb') as f:
    pickle.dump(obj, f, protocol=2)

def load_pickle(path):
  """Check and load pickle object.
  According to this post: https://stackoverflow.com/a/41733927, cPickle and
  disabling garbage collector helps with loading speed."""
  assert os.path.exists(path)
  # gc.disable()
  with open(path, 'rb') as f:
    ret = pickle.load(f)
  # gc.enable()
  return ret

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def main(args):
    total_queries = 0
    args = parser.parse_args(args)
    src = os.path.join(args.root,'ranked_results')
    query_folders = os.listdir(src)
    if args.market:
        pattern = re.compile(r'(?:gallery?|query?)_top([-\d]+)_name[_]+([-\d]+)_c([-\d]+)s([-\d]+)_[-\d]+_[-\d]+.jpg')
    elif args.msmt17:
        pattern = re.compile(r'(?:gallery?|query?)_top([-\d]+)_name[_]+([-\d]+)_[-\d]+_([-\d]+)_[-\d]+[a-z]+_[-\d]+_[-\d]+[_]*[a-z]*.jpg')

    def get_numb(s):
        rank,_,_,_ = map(float, pattern.search(s).groups())
        return rank

    fig=plt.figure(figsize=(10, 10))
    count = 0
    pic_count = 1
    columns = 6
    rows = 4
    file_name = 'rank_{}_'
    total_count = 0
    query_folder_ranks = defaultdict(list)
    query_folders.sort()
    for query_folder in query_folders:
        sorted_files = os.listdir(os.path.join(src,query_folder))
        if args.without_rotation:
            sorted_files.sort(key=get_numb)
        else:
            sorted_files.sort(key=alphanum_key)
        try:
            if args.market:
                _,query_id, camera, _ = map(float, pattern.search(sorted_files[0]).groups())
            elif args.msmt17:
                _, _,query_id, camera = map(float, pattern.search(sorted_files[0]).groups())
        except:
            pdb.set_trace()
        rank = getRank(sorted_files,args)
        query_folder_ranks[rank].append(query_folder)
        if rank<6:
            total_queries +=1

    print('Done building dict, Number of Queries with Rank <6: {}'.format(total_queries))
    printed = False
    step = 0
    st = time.time()
    prefix_camera = '_gt_'
    if args.market or args.msmt17:
        prefix_camera = '_cam_'
    for i in query_folder_ranks.keys():
        if i < 6:
            folder_save = os.path.join(args.root,'rank_{}'.format(i))
            may_make_dir(folder_save)
            k=0
            for query_folder in query_folder_ranks[i]:
                last_time = time.time()
                step+=1
                sorted_files = os.listdir(os.path.join(src, query_folder))
                sorted_files.sort(key=alphanum_key)
                if args.market:
                    _,query_id, camera, _ = map(float, pattern.search(sorted_files[0]).groups())
                elif args.msmt17:
                    _, _,query_id, camera = map(float, pattern.search(sorted_files[0]).groups())
                gt_rot = camera

                for im_num ,img_file in enumerate(sorted_files[:6]):
                    if args.market:
                        _,id, camera, _ = map(float, pattern.search(img_file).groups())
                    elif args.msmt17:
                        _,_,id, camera = map(float, pattern.search(img_file).groups())

                    if query_id == id:
                        color = 'green'
                    else:
                        color = 'red'
                    #img = plt.imread(os.path.join(src, query_folder, img_file))
                    #img = Image.fromarray(np.uint8(img*255))
                    img = Image.open(os.path.join(src, query_folder, img_file)).convert('RGB')
                    img = ImageOps.expand(img, border=10, fill=color)
                    fig.add_subplot(rows, columns, pic_count)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.annotate(str(int(id)) +prefix_camera+str(int(camer)), (0,0), (-10,0), xycoords='axes fraction', textcoords='offset points', va='top')
                    if (k != 0 and (k+1)%(columns*rows) == 0) or (k == len(query_folder_ranks[i])*6-1 and im_num ==5):
                        plt.savefig(os.path.join(folder_save, file_name.format(i) +str(count)+'.png'))
                        plt.close(fig)
                        fig=plt.figure(figsize=(10, 10))
                        count += 1
                        pic_count = 0

                    pic_count += 1
                    k += 1

                if not printed:
                  printed = True
                else:
                  # Clean the current line
                  sys.stdout.write("\033[F\033[K")
                print('{}/{} queries done, +{:.2f}s, total {:.2f}s'
                      .format(step, total_queries,
                              time.time() - last_time, time.time() - st))

    print('Saving pickle')
    save_pickle(query_folder_ranks,os.path.join(args.root,'rank_perQuery.pkl'))


if __name__ == '__main__':
    main(sys.argv[1:])

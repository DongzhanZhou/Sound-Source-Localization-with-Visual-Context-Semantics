import lmdb
import os
import sys
import librosa
from tqdm import tqdm
import pickle
import cv2

def main(lmdb_path, lmdb_name, data_path, file_list, modal='image'):
    
    opt = {
            'name': lmdb_name,
            'file_list': file_list,
            'file_folder': data_path,
            'lmdb_save_path': os.path.join(lmdb_path, lmdb_name),
            'commit_interval': 100,  # After commit_interval images, lmdb commits
            'num_workers': 8,
            'suffix': '.flac' # audio only, choose between ['flac', 'wav', 'mp3']
        }

    if modal == 'image':
        general_image_folder(opt)
    elif modal == 'audio':
        general_audio_folder(opt)
    else:
        raise NotImplementedError

def general_audio_folder(opt):
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)
    
    # read all the file paths to a list
    print('Reading audio path list ...')
    with open(opt['file_list'], 'r') as f:
        lines = f.readlines()
    keys = [line.strip() for line in lines]
    all_audio_list = [os.path.join(opt['file_folder'], key+opt['suffix']) for key in keys]
    
    # create lmdb environment
    data_size_per_audio = librosa.load(all_audio_list[0], sr=24000)[0].nbytes
    print('data size per audio is: ', data_size_per_audio)
    data_size = data_size_per_audio * len(all_audio_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 10)
    # map_size：
    # Maximum size database may grow to; used to size the memory mapping. If database grows larger
    # than map_size, an exception will be raised and the user must close and reopen Environment.

    # write data to lmdb

    txn = env.begin(write=True)
    tqdm_iter = tqdm(enumerate(zip(all_audio_list, keys)), total=len(all_audio_list), leave=False)
    for idx, (path, key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))

        key_byte = key.encode('ascii')
        data, _ = librosa.load(all_audio_list[idx], sr=24000)

        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            # begin again after committing
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    # create meta information
    # check whether all the images are the same size
    meta_info['keys'] = keys

    pickle.dump(meta_info, open(os.path.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

def general_image_folder(opt):
    """
    Create lmdb for general image folders
    If all the images have the same resolution, it will only store one copy of resolution info.
        Otherwise, it will store every resolution info.
    """
    lmdb_save_path = opt['lmdb_save_path']
    meta_info = {'name': opt['name']}

    if not lmdb_save_path.endswith('.lmdb'):
        raise ValueError("lmdb_save_path must end with 'lmdb'.")
    if os.path.exists(lmdb_save_path):
        print('Folder [{:s}] already exists. Exit...'.format(lmdb_save_path))
        sys.exit(1)

    print('Reading image path list ...')
    with open(opt['file_list'], 'r') as f:
        lines = f.readlines()
    videos = [line.strip() for line in lines]
    keys = []
    all_img_list = []
    for video in videos:
        for i in range(12):
            keys.append("%s/%03d.jpg" %(video, i+1))
    
    all_img_list = [os.path.join(opt['file_folder'], key) for key in keys]

    data_size_per_img = cv2.imread(all_img_list[0], cv2.IMREAD_UNCHANGED).nbytes
    print('data size per image is: ', data_size_per_img)
    data_size = data_size_per_img * len(all_img_list)
    env = lmdb.open(lmdb_save_path, map_size=data_size * 5)

    txn = env.begin(write=True)
    resolutions = []
    tqdm_iter = tqdm(enumerate(zip(all_img_list, keys)), total=len(all_img_list), leave=False)
    for idx, (path,key) in tqdm_iter:
        tqdm_iter.set_description('Write {}'.format(key))

        key_byte = key.encode('ascii')
        data = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        data = cv2.resize(data, dsize=(256, 256))

        if data.ndim == 2:
            H, W = data.shape
            C = 1
        else:
            H, W, C = data.shape
        resolutions.append('{:d}_{:d}_{:d}'.format(C, H, W))

        txn.put(key_byte, data)
        if (idx + 1) % opt['commit_interval'] == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.commit()
    env.close()
    print('Finish writing lmdb.')

    # create meta information
    # check whether all the images are the same size
    #assert len(keys) == len(resolutions)
    if len(set(resolutions)) <= 1:
        meta_info['resolution'] = [resolutions[0]]
        meta_info['keys'] = all_img_list
        print('All images have the same resolution. Simplify the meta info.')
    else:
        meta_info['resolution'] = resolutions
        meta_info['keys'] = all_img_list
        print('Not all images have the same resolution. Save meta info for each image.')

    pickle.dump(meta_info, open(os.path.join(lmdb_save_path, 'meta_info.pkl'), "wb"))
    print('Finish creating lmdb meta info.')

if __name__ == '__main__':
    main(lmdb_path='data/lmdb_folder', 
            lmdb_name='train_images.lmdb', 
            data_path='data/video_frames', 
            file_list='data/train_list.txt', 
            modal='image')
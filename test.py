import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
from model import TripletNet, TripletNet2
import argparse

parser= argparse.ArgumentParser()
parser.add_argument('--device_num', type=int, help='WHICH GPU')
args = parser.parse_args()
print(args)
args.cuda = torch.cuda.is_available()

def get_mel(mel_path, songid):
    arr = np.array([])
    try :
        arr = np.load(mel_path + songid + '.npy')
    except:
        print ("mel file not found.")
    return arr

# go through the songs 
def find_topN_songs(songid, songs_npy, N, model_path, mel_path, id_to_artist_song_file):
    # given this song, calculate distance with all the other songs and get shortest distance songs  
    
    id_to_artistsong = pickle.load(open(id_to_artist_song_file, 'rb'))

    print ("Root song is {}, artist is {}".format(id_to_artistsong[songid][0], id_to_artistsong[songid][1]))

    model = TripletNet2(0)
    if args.cuda : 
        model = model.cuda()

    model.load_state_dict(torch.load(model_path))
    
    root_mel = get_mel(mel_path, songid)
    if root_mel.size ==0:
        return

    root_mel = Variable(torch.from_numpy(root_mel), volatile=True)
    if args.cuda :
        root_mel = root_mel.cuda()
    root_mel = root_mel.view([1, root_mel.shape[0], root_mel.shape[1]])
    root_feature_vec = model.forward_single(root_mel)
    
    dist_list = {}

    for songid in songs_npy:
        cmp_mel = get_mel(mel_path, songid)
        if cmp_mel.size ==0:
            continue
        cmp_mel = Variable(torch.from_numpy(cmp_mel), volatile=True)
        if args.cuda:
            cmp_mel = cmp_mel.cuda()
        cmp_mel = cmp_mel.view([1, cmp_mel.shape[0], cmp_mel.shape[1]])
        cmp_feature_vec = model.forward_single(cmp_mel)
        if cmp_feature_vec.data[0].cpu().numpy().all() == 0:
            continue
        dist = F.pairwise_distance(root_feature_vec, cmp_feature_vec)
        dist_list[songid] = float(dist.data[0])

        print ("song {}, dist {}".format(id_to_artistsong[songid][0].encode("utf-8"), float(dist[0])))

    # sort by dist
    dist_list = sorted(dist_list.items(), key=lambda x:x[1])
    
    counter = N
    topN = []
    for i in dist_list:
        if counter > 0:
            songid = i[0]
            artist = id_to_artistsong[songid][0]
            song = id_to_artistsong[songid][1]
            topN.append([song, artist, i[1]])
        counter -=1

    return topN

if __name__=='__main__':
    msd_data_path = '/media/bach4/kylee/MSD_mel/MSD_split/'
    net_data_path = '/media/bach4/kylee/triplenet-data/'
    model_path = '/home/kylee/dev/audio-content-tripletnet/'
    songs_npy= np.load(net_data_path+'songs.npy')
    random_song = songs_npy[2110]
    r = find_topN_songs(random_song, songs_npy, 10, model_path+'TripletNet.pt', net_data_path+'mel_dir/', msd_data_path+'echonest_id_to_artist_song.pkl')
    print (r)




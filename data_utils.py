# coding: utf-8

## Taste profile subset data
# shape : (1019318, 384546)  
# sparsity : 99.9876%  

import os
import csv
import numpy as np
import pandas as pd
import scipy.sparse as sparse
import _pickle as cP


'''
Create a sparse matrix from Taste profile subset data.
Format of txtfile is 
user | song | play count
'''
def create_sparse_matrix(txtfile):
    '''
    returns :
        data : dataframe containing user play history
        user_song_sparse : csr_matrix
        users : list of unique users
        songs : list of unique songs
    '''
    data = pd.read_csv(txtfile, sep="\t", header=None, names=['user', 'song', 'play_count'])
    print (data.columns)
    print (data.info(verbose=True, null_counts=True))
    
    # users = list(np.sort(data.user.unique()))
    users = list(data.user.unique())
    songs = list(data.song.unique()) # is this index preserved in the csr matrix and is used for mapping to audio CNN?
    plays = list(data.play_count.astype(np.float))

    rows = data.user.astype('category', categories = users).cat.codes 
    # Get the associated row indices
    cols = data.song.astype('category', categories = songs).cat.codes 
    # Get the associated column indices
    user_song_sparse = sparse.csr_matrix((plays, (rows, cols)), shape=(len(users), len(songs)))
    print (user_song_sparse.shape)
    
    return data, user_song_sparse, users, songs

'''
Calculate how sparse the matrix is.
'''
def calculate_sparsity(user_song_sparse):
    matrix_size = user_song_sparse.shape[0]*user_song_sparse.shape[1]
    total_plays = len(user_song_sparse.nonzero()[0])
    sparsity = 100*(1 - (total_plays/matrix_size))
    print(sparsity)
    if sparsity > 99.5:
        print ("Matrix may be too sparse")



''' For a given user, create list of playcount >1 and playcount ==1 '''
def _divide_playcount_list (df, user_playcount_list, max_len):
    '''
    return : 
        multiples_pc_list : list of songs that have been played more than once 
        ones_pc_list : list of songs only played once (used as negative samples)
    '''

    sorted_list = np.argsort(user_playcount_list.data)[::-1]
    one_idx = -1

    for i in range(len(sorted_list)):  
        if (sorted_list[i] == 1): # first song with a playcount of 1
            one_idx = i
            break
    if one_idx == -1 : 
        print ("No playcount of ones...Need to do random select")
    
    # print("Number of ones : %d/%d"%(len(sorted_list) - one_idx, len(sorted_list)))
    ones_pc_list = []
    multiples_pc_list = []
        
    for i in sorted_list : 
        if i < one_idx:
            multiples_pc_list.append(df.iloc[i]['song'])
        else :
            ones_pc_list.append(df.iloc[i]['song'])
    
    # if the playlist is too long cut it to top 50 listened songs
    if (len(multiples_pc_list) > max_len) :
        multiples_pc_list = multiples_pc_list[:max_len]
    
    return multiples_pc_list, ones_pc_list
    
''' Make a triplet pair (positive,positive,negative) and save to the txt file '''
def _make_triplet_data(data_dir, df, user_song_matrix_subset, max_len):
    if os.path.isfile(data_dir + '.txt'):
        print ("File already exists")
        return
    
    f = open (data_dir + '.txt', 'w')
    
    for user_idx in range(user_song_matrix_subset.shape[0]):
        multiples, ones = _divide_playcount_list(df, user_song_matrix_subset[user_idx], max_len)
        for i in range(len(multiples)) :
            for j in range(i+1, len(multiples)):
                f.write(multiples[i] + ',')
                f.write(multiples[j] + ',')
                randomidx = np.random.choice(len(ones))
                f.write(ones[randomidx] + '\n')
    f.close()


''' Split data into training, valid, test set and save to the correct textfile'''
def split_data(training_data_path, df, user_song_matrix, max_len):
#     user_song_matrix = user_song_matrix[:100]
    train_len = int (user_song_matrix.shape[0] * 0.8)
    test_len = user_song_matrix.shape[0] - train_len
    valid_len = int (train_len * 0.2)
    train_len = train_len - valid_len
    print ("train %d, valid %d, test %d"%(train_len, valid_len, test_len))

    _make_triplet_data(training_data_path + 'train', df, user_song_matrix[:train_len], max_len) # numpy array
    _make_triplet_data(training_data_path + 'valid', df, user_song_matrix[train_len:train_len+valid_len], max_len)
    _make_triplet_data(training_data_path + 'test', df, user_song_matrix[train_len+valid_len:], max_len)

''' When getting item from dataloader, return the mel array '''
def get_mel(mel_path, echonest_song_id) :
    arr = np.load(mel_path + echonest_song_id + '.npy')
    return arr

if __name__ == '__main__':
    train_triplets = '/media/bach4/kylee/Deep-content-data/train_triplets.txt'
    training_data_path = '/media/bach4/kylee/triplenet-data/'

    # create a sparse matrix 
    df, user_song_matrix, users, songs= create_sparse_matrix(train_triplets)
    calculate_sparsity(user_song_matrix)

    split_data(training_data_path, df, user_song_matrix, 5)



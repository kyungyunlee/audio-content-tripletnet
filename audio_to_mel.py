import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as ms
import IPython.display as ipd
import librosa
import librosa.display
import cPickle as cP


# ### Process and save melspectrogram 
# As done in paper "AUTOMATIC TAGGING USING DEEP CONVOLUTIONAL NEURAL NETWORKS"   
# sampling rate downsampled to 12kHz  
# audio length trimmed to 29.1s  
# mel bins 96  
# hop size 256 samples  

def create_melspectrogram(song_list, base_dir, mel_dir, song_dir, SR, n_fft, hop_size, n_mels, audio_len):

     
    id7d_to_path = cP.load(open(base_dir + '7D_id_to_path.pkl','r'))
    idmsd_to_id7d = cP.load(open(base_dir + 'MSD_id_to_7D_id.pkl','r'))
    idechonest_to_idmsd = cP.load(open(base_dir + 'echonest_id_to_MSD_id.pkl', 'r'))

    for i in range(len(song_list)):
        song_tag = song_list[i]
        
        save_file = mel_dir + song_tag + '.npy'
        audio_file = song_dir  + id7d_to_path[idmsd_to_id7d[idechonest_to_idmsd[song_tag]]]
        
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
        if os.path.isfile(save_file)==1:
            print ("{} , file already exists".format(save_file))
            continue
            
        try:
            y, sr = librosa.load(audio_file)
            # perform short time fourier transform
            S = librosa.core.stft(y, n_fft=n_fft, hop_length=hop_size)
            S = np.abs(S) # get magnitude
            
            mel_basis = librosa.filters.mel(SR, n_fft=n_fft, n_mels=n_mels)
            mel_S = np.dot(mel_basis,S)
            # log!
            mel_S = np.log10(1+10*mel_S)
            mel_S = mel_S.astype(np.float32)
            print (mel_S.shape)
            
            # make sure all processed data have same shape (same number of frames)
            
            num_frames = int ((audio_len * SR)/hop_size)
            # print num_frames
            if mel_S.shape[1] < num_frames : 
                mel_S = np.concatenate((mel_S,mel_S),axis=1)
                # print mel_S.shape
            if mel_S.shape[1] > num_frames: 
                mel_S = mel_S[:, :num_frames]
            
            print ("{} : mel_shape={}, file_name={}".format(i, mel_S.shape, save_file))
            np.save(save_file, mel_S)
                
        except Exception:
            print ("Error : file {} not found".format(save_file))
            continue
            print ("[%d/%d] done" %(i+1, len(song_list)))

    print ("Finished!")


if __name__ =='__main__':

    n_mels = 96
    hop_size = 256
    n_fft = 512
    SR = 12000
    audio_len=29.15
    
    base_dir = '/media/bach4/kylee/MSD_mel/MSD_split/'
    song_path = "/media/bach2/dataset/MSD/songs/"
    my_path = '/media/bach4/kylee/triplenet-data/'
    song_list = np.load(my_path + 'songs.npy')
    song_list = song_list.tolist()
    create_melspectrogram(song_list, my_path+'mel_dir/', song_path, SR, n_fft, hop_size, n_mels, audio_len)


import pickle

file_path = r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\data_emotion.p'
data, W, vocab, word_idx_map, max_sentence_length, label_index = pickle.load(open(file_path, 'rb'))

file_path = r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\data_sentiment.p'
data, W, vocab, word_idx_map, max_sentence_length, label_index = pickle.load(open(file_path, 'rb'))

#------------------------------------------------------------------------

file_path = r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\text_glove_average_sentiment.pkl'
train_text_avg_emb, val_text_avg_emb, test_text_avg_emb = pickle.load(open(file_path, 'rb'))

file_path = r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\text_glove_average_emotion.pkl'
train_text_avg_emb, val_text_avg_emb, test_text_avg_emb = pickle.load(open(file_path, 'rb'))

#------------------------------------------------------------------------

file_path = r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\audio_embeddings_feature_selection_emotion.pkl'
train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(file_path, 'rb'))

file_path = r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\audio_embeddings_feature_selection_sentiment.pkl'
train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(open(file_path, 'rb'))

#------------------------------------------------------------------------

train_text_CNN_emb, val_text_CNN_emb, test_text_CNN_emb = pickle.load(open(
    r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\text_glove_CNN_emotion.pkl', 'rb'))
train_text_CNN_emb, val_text_CNN_emb, test_text_CNN_emb = pickle.load(open(
    r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\text_glove_CNN_sentiment.pkl', 'rb'))

#------------------------------------------------------------------------

train_text_emb, val_text_emb, test_text_emb = pickle.load(open(r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\text_emotion.pkl', 'rb'))
train_text_emb, val_text_emb, test_text_emb = pickle.load(open(r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\text_sentiment.pkl', 'rb'))

#-------------------------------------------------------------------------

file_path = r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\audio_emotion.pkl'
with open(file_path, 'rb') as file:
    train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(file)

file_path = r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\audio_sentiment.pkl'
with open(file_path, 'rb') as file:
    train_audio_emb, val_audio_emb, test_audio_emb = pickle.load(file)

#-------------------------------------------------------------------------

train_bimodal_emb, val_bimodal_emb, test_bimodal_emb = pickle.load(open(
    r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Features.Models\features\\bimodal_sentiment.pkl', 'rb'))

# Reading the datasets

#Read Meld

import pandas as pd
df_train = pd.read_csv(r'C:\Users\saury\PycharmProjects\FacialMMT\MELD.Raw\train.tar\\train_sent_emo.csv')
#print(df_train.head(1000))

utt = df_train['Utterance'].tolist() # load the list of utterances
dia_id = df_train['Dialogue_ID'].tolist() # load the list of dialogue id's
utt_id = df_train['Utterance_ID'].tolist() # load the list of utterance id's
for i in range(len(utt)):
    print ('Utterance: ' + utt[i]) # display utterance
    print ('Video Path: train_splits/dia' + str(dia_id[i]) + '_utt' + str(utt_id[i]) + '.mp4') # display the video file path
    print ()

# Read emorynlp

df_train = pd.read_csv(r'C:\Users\saury\PycharmProjects\FacialMMT\emorynlp\\emorynlp_test_final.csv') # load the .csv file, specify the appropriate path
utt = df_train['Utterance'].tolist() # load the list of utterances
sea = df_train['Season'].tolist() # load the list of season no.
ep = df_train['Episode'].tolist() # load the list of episode no.
sc_id = df_train['Scene_ID'].tolist() # load the list of scene id's
utt_id = df_train['Utterance_ID'].tolist() # load the list of utterance id's
for i in range(len(utt)):
    print ('Utterance: ' + utt[i]) # display utterance
    print ('Video Path: emorynlp_train_splits/sea' + str(sea[i]) + '_ep' + str(ep[i]) + '_sc' + str(sc_id[i]) + '_utt' + str(utt_id[i]) + '.mp4') # display the video file path
    print ()


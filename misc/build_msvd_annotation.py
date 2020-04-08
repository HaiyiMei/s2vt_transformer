import json
import sys
import pandas as pd


video_data = pd.read_csv('./data/video_corpus.csv', sep=',')
#filte Language==English
video_data = video_data[video_data['Language'] == 'English']
#add video path
video_data['video_path'] = video_data.apply(lambda row: row['VideoID']+'_'+str(int(row['Start']))+'_'+str(int(row['End'])), axis=1)

sents_anno = []
for name, desc in zip(video_data['video_path'], video_data['Description']):
    d = {}
    d['caption'] = desc
    d['video_id'] = name
    sents_anno.append(d)

anno = {'sentences': sents_anno}
with open('./feats_MSVD/test.json', 'w') as f:
    json.dump(anno, f)
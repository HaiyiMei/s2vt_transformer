### environment
python2.7
python3.7
pytorch

### prepare for train
1. download the video clips from [here](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/)

2. download caffe and VGG model
download caffe from [BVLC/caffe](https://github.com/BVLC/caffe) to "/root/workspace/"    
download [VGG16](http://www.robots.ox.ac.uk/~vgg/software/very_deep/caffe/VGG_ILSVRC_16_layers.caffemodel) model and its [prototxt](https://gist.githubusercontent.com/ksimonyan/211839e770f7b538e2d8/raw/0067c9b32f60362c74f4c445a080beed06b07eb3/VGG_ILSVRC_16_layers_deploy.prototxt) from [caffe/model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo#models-used-by-the-vgg-team-in-ilsvrc-2014) and put it into "/root/workspace/caffe/models/vgg_16"

3. extract feature from video clip
```
python extract_RGB_feats.py
```
then put these feature data into training data (put into this folder "./rgb_train_features") and test data (put into this folder "./rgb_test_features").

### how to train/test
```
python s2vt.py train
python s2vt.py test
```
and you will get a file named "test_result.txt" in the root folder of this project.

### how to evaluation
1. install java
```shell
conda install -c cyclus java-jdk
conda install matplotlib
conda install scikit-image
```
2. download data
MSVD data set:[Microsoft Research Video Description Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52422&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F38cf15fd-b8df-477e-a4e4-a4680caa75af%2Fdefault.aspx)
put it to ./data folder.

3. parse video csv
```
python ./coco_eval/parse_video_csv.py
```
4. create reference and result .json file
```
python ./coco_eval/create_reference.py
python ./coco_eval/create_result_json.py
```
5. evalute the result
```
python ./coco_eval/eval.py
```

and you will get the evalution result.

### my model
[here]() is a pytorch model trained by myself
### acknowledgement
some code copy from chenxinpeng(https://github.com/chenxinpeng/S2VT)

### note
the project does not work now(20180119)

### related paper
Sequence to Sequence - Video to Text(ICCV2015)(https://vsubhashini.github.io/s2vt.html)

### note
any questions, please contact me: jiguo.li@vipl.ict.ac.cn or jgli@pku.edu.cn





-------------------
batch_size = 100
image_dim = 4096
img_embed_dim = 1000
word_embed_dim = 1000
hidden_dim = 1000
video_step = 80
caption_step = 20
epoches = 1001
csv_path = "./data/video_corpus.csv"
input_video_shape = (80, 4096)

output1.shape torch.Size([10, 1000])
output2.shape torch.Size([10, 1000])
state1.shape torch.Size([10, 1000])
state2.shape torch.Size([10, 1000])


LSTM1:
    input:(1, 1000) output:(1, 1000)
LSTM2:
    input:(1, 2000) output:(1, 1000)


ValueError: Object arrays cannot be loaded when allow_pickle=False
解决办法：讲numpy版本降到1.16.2或以下

目前存在的问题：
1. decoder在训练阶段输入到LSTM2中的词向量是label的词向量还是上一个step生成的词的向量
2. 关于caption_mask的功能，将大于句子词数后面的pad直接乘以0会不会导致实际训练的loss与表现出来的不一样
    ['baby husky with a husky',
    'A woman is playing a flue',
    'A woman is exercising',
    'An older man takes a boy by the arm and throws him into a chair',
    'A woman is mixing food in a bowl',
    'A woman cuts vegetables',
    'The chopping and preperation of vedgetables',
    'Someone is slicing fruit',
    'A woman is pouring an egg mixture into a frying pan',
    'Somebody is cooking']

    caption_mask
    [[1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [1. 1. 1. 1. 1. 1. 1. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]


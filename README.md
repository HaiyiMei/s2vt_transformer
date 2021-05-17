# s2vt_transformer

install package
```Shell
conda install -c bioconda java-jdk
pip install git+https://github.com/ildoonet/pytorch-gradual-warmup-lr.git
```

link the file
```Shell
ln -s /data/ feats
```

`feats` should look like:
```Shell
├── feats
    ├── MSVD
        ├── videos
    ├── MSRVTT
        ├── videos
```

extract frames:
```
python misc/video2image.py feats/MSRVTT/videos feats/MSRVTT/frames && \
python misc/video2image.py feats/MSVD/videos feats/MSVD/frames
```

extract bounding box:
```Shell
python misc/extract_bbox.py --dataset MSVD && \
python misc/extract_bbox.py --dataset MSRVTT
```


get [temporal shift module](https://github.com/mit-han-lab/temporal-shift-module) and then extract features:
```Shell
git clone https://github.com/mit-han-lab/temporal-shift-module

python misc/extract_feature.py --dataset MSVD --sample_len 32 --model tsn --mode frame --clip_len 16 --output_dir uniform_frame && \
python misc/extract_feature.py --dataset MSVD --sample_len 32 --model resnet --mode frame --output_dir uniform_frame && \
python misc/extract_feature.py --dataset MSRVTT --sample_len 32 --model tsn --mode frame --output_dir uniform_frame && \
python misc/extract_feature.py --dataset MSRVTT --sample_len 32 --model resnet --mode frame --output_dir uniform_frame
```

```Shell
python misc/extract_feature.py --dataset MSVD --model senet
python misc/extract_feature.py --dataset MSVD --model pnasnet
python misc/extract_feature.py --dataset MSVD --model i3d --clip_len 32 --mode clip

```

train:
```Shell
python train.py --cfg configs/msvd_base.yml
python train.py --cfg configs/msvd_fusion.yml
```
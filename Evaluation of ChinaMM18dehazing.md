# The evaluation of mAP in ChinaMM18dehazing

We use faster-rcnn for detection task. Specificaclly, we load weights of e2e_faster_rcnn_R-101-FPN_2x from Detectron/MODEL_ZOO.

## Installation
Follow the instruction of README.md of this repo to complete the installation of Detectron.pytorch.

## Evaluation
**Notice: Please DO NOT change any setting parameter in this repo.**

### Data preparation
Create data folders under the repo,
```
cd {repo_root}
mkdir -p data/pretrained_model/e2e_faster_rcnn_R-101-FPN_2x
mkdir reside_results
```
The evaluation result will be stored in ```reside_results```.

Download the pre-trained faster rcnn model,
```
cd data/pretrained_model/e2e_faster_rcnn_R-101-FPN_2x
wget -c https://s3-us-west-2.amazonaws.com/detectron/35857952/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml.01_39_49.JPwJDh92/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
```

Download [RTTS dataset](http://www.google.com/url?q=http%3A%2F%2Ft.cn%2FRHP3eXg&sa=D&sntz=1&usg=AFQjCNF2ll2T1XuV-nCFD2aV0VV0P5PReg). Named it as ```RTTS``` and put it under ```{repo_root}```.

### Evaluate the generate results.

#### Step 1:
```
python tools/infer_simple2text.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml --load_detectron data/pretrained_model/e2e_faster_rcnn_R-101-FPN_2x/model_final.pkl --image_dir {generated dehazed result}/real_detection/ --output_dir reside_results/{your team number}
```

#### Step 2:
```
python tools/voc_eval.py --results reside_results/{your team number}_results --annotations RTTS/Annotations/ --file_list RTTS_file_list.txt
```
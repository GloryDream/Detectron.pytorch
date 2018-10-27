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
```

Download the pre-trained faster rcnn model,
```
cd data/pretrained_model/e2e_faster_rcnn_R-101-FPN_2x
wget -c https://s3-us-west-2.amazonaws.com/detectron/35857952/12_2017_baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml.01_39_49.JPwJDh92/output/train/coco_2014_train%3Acoco_2014_valminusminival/generalized_rcnn/model_final.pkl
```

### Evaluate the generate results.

#### Step 1:
```
python tools/infer_simple2text.py --dataset coco2017 --cfg configs/baselines/e2e_faster_rcnn_R-101-FPN_2x.yaml --load_detectron data/pretrained_model/e2e_faster_rcnn_R-101-FPN_2x/model_final.pkl --image_dir {your image folder} --output_dir {your results folder} --name {the name of the json file} 
```

#### Step 2:
```
python tools/voc_eval.py --results reside_results/{your team number}_results --annotations RTTS/Annotations/ --file_list RTTS_file_list.txt
```
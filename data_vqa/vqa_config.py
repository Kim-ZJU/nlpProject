from easydict import EasyDict as edict

train = edict({
    "image":"./data/images/train",
    "annotation":"./data/annotations/train.json",
    "question":"./data/questions/train.json",
})

test = edict({
   "image":"./data/images/test",
   "annotation":"./data/annotations/test.json",
   "question":"./data/questions/test.json"
})

val = edict({
    "image":"./data/images/val",
    "annotation":"./data/annotations/val.json",
    "question":"./data/questions/val.json",
})

model_cfg = edict({
    "cnn_ckpt_19":"./data/vgg19_ascend_v111_imagenet2012_research_cv_bs64_acc74.ckpt",
    "cnn_ckpt_16":"./data/vgg16_ascend_v120_imagenet2012_official_cv_bs32_acc73.ckpt",
    "log_path":"./outputs",

    "device_target": 'Ascend',
    "per_batch_size": 32,
    "graph_ckpt":1,
    "rank": 0,
    "group_size":1
})
import math
import os
import json
import numpy as np

import mindspore.dataset.vision.py_transforms as vision
import mindspore.dataset as de
from PIL import Image
from mindspore import Tensor, nn, Model
from easydict import EasyDict as edict

def img2tensor(img):
    # mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    # std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = edict({
        "ToPIL": vision.ToPIL(),
        "Decode": vision.Decode(),
        "Resize": vision.Resize((512, 512)),
        "CenterCrop": vision.CenterCrop(448),
        "ToTensor":vision.ToTensor(),
        # "Rescale":vision.Rescale(1.0/255.0,0.0),
        "Normalize": vision.Normalize(mean=mean, std=std),
        "HWC2CHW": vision.HWC2CHW(),
    })
    # print("input image:",img.shape)
    img = transform.HWC2CHW(img) / 255
    # print("before normaize",img)
    # img = (img - mean[:, None, None]) / std[:, None, None]
    img = transform.Normalize(img)
    # print('img after normalize:',img)
    # print("totensor:",img)
    # img = [img]
    # tensor = Tensor(img,mstype.float32)
    # return tensor
    return img
    
def create_dataset(batch_size,mode = 'train',drop_remainder=True,q_dict = None,a_dict=None):
    dataset = VQADataset(mode,q_dict = q_dict,a_dict = a_dict)
    sampler = DistributedSampler(dataset)

    de_dataset = de.GeneratorDataset(dataset, ["image", "question","label"],shuffle=False,sampler=sampler)
    de_dataset = de_dataset.map(operations=img2tensor, input_columns="image", num_parallel_workers=8)
    # de_dataset = de_dataset.map(operations=None, input_columns="question", num_parallel_workers=8)
    # de_dataset = de_dataset.map(operations=None, input_columns="label", num_parallel_workers=8)
    de_dataset = de_dataset.project(columns=['image','question','label'])
    de_dataset = de_dataset.batch(batch_size, drop_remainder=drop_remainder)
    
    return de_dataset
    
class DistributedSampler():
    """
    sampling the dataset.
    Args:
    Returns:
        num_samples, number of samples.
    """
    def __init__(self, dataset, rank=0, group_size=1, shuffle=True, seed=0):
        self.dataset = dataset
        self.rank = rank
        self.group_size = group_size
        self.dataset_length = len(self.dataset)
        self.num_samples = int(math.ceil(self.dataset_length * 1.0 / self.group_size))
        self.total_size = self.num_samples * self.group_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            self.seed = (self.seed + 1) & 0xffffffff
            np.random.seed(self.seed)
            indices = np.random.permutation(self.dataset_length).tolist()
        else:
            indices = list(range(len(self.dataset_length)))

        indices += indices[:(self.total_size - len(indices))]
        indices = indices[self.rank::self.group_size]
        return iter(indices)

    def __len__(self):
        return self.num_samples

class VQADataset:
    """
    Args:
    mode: train/val/test
    cfg: 
    Returns:
        de_dataset.
    """
    def __init__(self, mode = "train",q_dict =None,a_dict = None):
        super(VQADataset, self).__init__()
        self.Resize =  vision.Resize((512, 512))
        self.CenterCrop =  vision.CenterCrop(448)
        self.images = []   # image paths
        self.questions = []   #  questions 应该是已经转换成one-hot的编码
        self.answers = []   # answers   对应的正确答案
        self.q_dict = q_dict
        self.a_dict = a_dict
        ann = {}
        ques = {}
        if mode == "train":
            ann = json.load(open(train.annotation,'r'))
            ques = json.load(open(train.question,'r'))
        elif mode == 'test':
            ann = json.load(open(test.annotation,'r'))
            ques = json.load(open(test.question,'r'))
        else:
            ann = json.load(open(val.annotation,'r'))
            ques = json.load(open(val.question,'r'))
        
        self.question_dict = {item['question_id']:item['question'] for item in ques['questions']}
        annotations = ann['annotations']
        img_not_exist_count = 0
        for index,item in enumerate(annotations):
            img_path = os.path.join(train.image,"COCO_{0}2014_{1}.jpg".format(mode,str(item['image_id']).zfill(12)))
            if not os.path.exists(img_path):  # filter non-existing image
                img_not_exist_count += 1
                continue
            
            answers = item['answers']     # filter blank answer
            if len(answers) == 0:
                continue
            
            election = dict() 
            for index,ans in enumerate(answers):
                if ans['answer_confidence'] == 'yes':
                    election[ans['answer']] = election.get(ans['answer'],0)+1
            if not election:
                continue
                
            elected = max(election, key=election.get)
            if not self.a_dict.has_key(elected):
                continue

            self.answers.append(elected)
            self.images.append(img_path)
            self.questions.append(item['question_id'])
            

    def __getitem__(self, index):
        print('index:',index)
        image = Image.open(self.images[index]).convert('RGB')
        image = self.Resize(image)
        image = self.CenterCrop(image)
        # image = self.images[index]

        question_str = self.question_dict[self.questions[index]]
        question = process_sentence(question_str)
        question = question.split()
        question = convert_sentence_to_vec(question,self.q_dict)

        answer = self.a_dict[self.answers[index]]
        return image, question,index

    def __len__(self):
        return len(self.questions)

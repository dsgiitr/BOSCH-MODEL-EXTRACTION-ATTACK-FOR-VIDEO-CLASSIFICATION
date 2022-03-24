# BOSCH's MODEL EXTRACTION ATTACK FOR VIDEO CLASSIFICATION

We present our work done on model extraction over Vision Transformers such as Video-Swin-T and MoViNeT-A2-Base on Video Action-Recognition tasks under Inter-IIT Tech Meet 10.0. We employ various video student models such as r3d, r(2+1)d and c3d. We also test out various classic techniques such as PRADA, MAZE and DFME.

Deep learning models have found their place in various applications in today’s world. Companies monetize these models as a service available to the end-users over the web. In this context, stealing the knowledge stored within this trained model is an attractive proposition for competitors. A ‘clone’ model can be trained with the victim model’s predictions to bring it close to the ‘victim’ model and can be used for monetary gains or to mount further attacks to improve the clone’s performance. Our solution to this challenge of model extraction attacks on video classification models is based on knowledge distillation. The student model learns by minimizing the difference between the teacher’s and its output logits. In model extraction attacks, the student is replaced by the clone model we are trying to train, whereas the teacher is replaced by the victim model queried. However, model extraction attacks cannot be taken as distillation problems directly because (a) we do not have access to the teacher model architecture, due to which backpropagation through it is not possible (b) we only have access to output logits of the victim model.

## Approach, <br>

![extraction](https://user-images.githubusercontent.com/18577165/159977307-778795fb-199e-4580-a6ca-a22af4932d14.jpeg)

## Experiments conducted, <br>
**Generating videos by stacking affine-transformed images** <br>
<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/18577165/159978268-86766a67-0d12-4cc5-a4cf-d74871cf9130.gif">
</p>
<br>

**Augmentations on the balanced 5% Kinetics dataset** <br>
<p align="center">
  <img width="65%" height="10%" src="https://user-images.githubusercontent.com/18577165/159974997-282e7073-a436-4b28-ac61-19dfe2ab9f4a.png">
</p>
<br>

**Using other action datasets and models** <br>
HMDB51            |  UCF101
:-------------------------:|:-------------------------:
![](https://user-images.githubusercontent.com/18577165/159976169-3f9f539c-93cf-4906-9a11-7d4fb45c1f4d.jpeg)  |  ![](https://user-images.githubusercontent.com/18577165/159976180-393c0267-6a7f-4098-9f17-6f4722acb3cb.jpeg)

<br>

## File Structure

- project_extraction
    - black_box
        - video_swin_blackbox
            - attack_video_swin.py
        - Video-Swin-Transformer
            - experiment_file_1
            - experiment_file_2
    - grey_box
        - video_swin_transformer_dependencies
        - ...
        - experiment_file_1
        - experiment_file_2
        - ...
        - eval_teacher.py
    - dependencies.sh
    - eval.only.py
    - dataset_folder_1
    - dataset_folder_2
    - ...
    - swin_weights
    - extracted_weights_1
    - extracted_weights_2
    - ...

## Setup

- Clone the repository <br>
```
git clone https://github.com/dsgiitr/BOSCH-MODEL-EXTRACTION-ATTACK-FOR-VIDEO-CLASSIFICATION project_extraction
```
- Set the environment <br>
```
cd project_extraction
bash dependencies.sh
export PYTHONOPTIMIZE='1' 
```
- Download and unzip the [datatset](#datasets) in project_extraction <br>
- Download the teacher-weights in project_extraction <br>
```
!wget https://github.com/SwinTransformer/storage/releases/download/v1.0.4/swin_base_patch244_window877_kinetics400_1k.pth
```
- (Optional) Download the extracted [weights](#extracted-weights) <br>


## Run code

- To run black_box code <br>
```
cd black_box
python3 code.py
```

- To run grey_box code <br>
```
cd grey_box
python3 code.py
```


## Datasets

Datasets such as kinetics, ucf101 and hmdb51 can be downloaded from [here](https://drive.google.com/drive/folders/1Nz8wSjpfAT_-w6bu7ik-tkMgY_973--i?usp=sharing) 


## Extracted Weights

Trained weights can be found [here](https://drive.google.com/drive/folders/16nTfSHd_XwvqHavlBDSCbl-4QV0A9jir)


## File Description

- [swin_to_r21_kd.py](grey_box/swin_to_r21_kd.py) :
    Final model used for the submission. Along with all the improvements made earlier, in this file, we calculate the loss not only by evaluating our student model's outputs against the outputs of the victim model but also against the true labels for the examples belonging to the relevant kinetics dataset. Along with this, we also increase the weightage of this loss against the true label (called as the "true_loss" of an example) in order to give greater emphasis to it.
    - Configuration :
        - Teacher : Swin-T
        - Student : r(2+1)d
        - Loss : KLDiv
        - Optim : AdamW with LR: 0.00003 on linear layers and 0.000003 on core layers
        - Scheduler : Reduce on plateau
        - Dataset : UCF101, HMDB51, kinetics_5percent

- [movinet_to_r21.py](grey_box/movinet_to_r21.py) :
    Final model used for the submission. Along with all the improvements made earlier, in this file, we calculate the loss not only by evaluating our student model's outputs against the outputs of the victim model but also against the true labels for the examples belonging to the relevant kinetics dataset. Along with this, we also increase the weightage of this loss against the true label (called as the "true_loss" of an example) in order to give greater emphasis to it.
    - Configuration :
        - Teacher : Movinet
        - Student : r(2+1)d
        - Loss : KLDiv
        - Optim : AdamW with LR: 0.00003 on linear layers and 0.000003 on core layers
        - Scheduler : Reduce on plateau
        - Dataset : UCF101, HMDB51, kinetics_5percent

- [eval_only.py](eval_only.py) :
    Used for validation against the k400/k600 validation dataset
    - Configuration
        - Dataset :
            - k400_validation (contains 10% class normalised kinetics 400 dataset)
            - k600_validation (contains 10% class normalised kinetics 600 dataset)

- [swin_to_c3d.py](grey_box/swin_to_C3D.py) :
    Uses c3d, a purely convolution based architechture, as a student with vanilla settings
    - Configuration :
        - Teacher : Swin-T
        - Student : c3d
        - Loss : KLDiv
        - Optim : AdamW with LR: 0.00003 on linear layers and 0.000003 on core layers
        - Scheduler : Reduce on plateau
        - FRAMES = 32

- [swin_to_r21.py](grey_box/swin_to_r21.py) :
    Changed the student to R(2+1)D, a model pretrained on IG65m dataset
    - Vanilla code for model extraction.
    - Configuration :
        - Teacher : Swin-T
        - Student : r(2+1)d
        - Loss : KLDiv
        - Optim : AdamW with LR: 0.00003 on linear layers and 0.000003 on core layers
        - Scheduler : Reduce on plateau

- [swin_to_r21_using_cosine.py](grey_box/swin_to_r21_cosine_agumented.py) :
    Changed the loss to cosine and included other datasets for better training
    - Configuration :
        - Teacher : Swin-T
        - Student : r(2+1)d
        - Loss : KLDiv
        - Optim : AdamW with LR: 0.00003 on linear layers and 0.000003 on core layers
        - Scheduler : Cosine

- [swin_to_r21_prada.py](grey_box/swin_to_r21_cosine_agumented.py) :
    Tried the techniques in PRADA [paper](https://arxiv.org/pdf/1805.02628.pdf) with vanilla settings
    - Configuration :
        - Teacher : Swin-T
        - Student : r(2+1)d
        - Loss : KLDiv
        - Optim : AdamW with LR: 0.00003 on linear layers and 0.000003 on core layers
        - Scheduler : Reduce on plateau

- [eval_teacher.py](grey_box/eval_teacher.py) :
    - Used to validate the teacher. Evaluates teacher on 10% class normalized kinetics400 dataset
    - Accpets both swin-T and movinet

- [attack_video_swin.py](black_box/video_swin_blackbox/attack_video_swin.py):
    Built upon MAZE. Add an extra temporal dimension and uses generator
    - Configuration :
        - Teacher : Swin-T
        - Student : r3d
        - Loss : KLDiv
        - Optim : Adam with LR: 0.001 of model and 0.00001 of generator
        - Scheduler : Cosine Annealing

- [attack_video_movinet.py](black_box/video_swin_blackbox/attack_video_movinet.py):
    Same as attack_video_swin, replaces teacher as movinet
    - Configuration :
        - Teacher : Movinet
        - Student : r3d
        - Loss : KLDiv
        - Optim : Adam with LR: 0.001 of model and 0.00001 of generator
        - Scheduler : Cosine Annealing

- [stacked_images_with_swin.py](black_box/Video-Swin-Transformer/stacked_images_with_swin.py)
    Applied affine transformation on images and stacked then to produce a video
    - Configuration :
        - Teacher : Swin
        - Student : r3d
        - Loss : KLDiv
        - Optim : Adam with LR: 0.001
        - Scheduler : Cosine Annealing

## Results

The following experiments were performed in the Video Swin Transformer. The best results from these experiments were then extended MoViNet as the victim, hence developing a common strategy as asked. 

**The accuracies are calculated on the Kinetics400/600 validation dataset (_true labels_)**.

### Grey Box setting
|Technique|Top-5 Accuracy|Top-1 Accuracy|
|----------------------|-|-|
|Augmented Kinetics with C3D|27.5|8.4|
|Augmented Kinetics with R(2+1)D|42.5|19.1|
|Concatenated dataset with R(2+1)D|51.8|30.6|
|Combining PRADA approach with R(2+1)|34.2|12.67|
|Combining KD techniques|**54.8**|**31.4**|

The final results for Video Swin Transformer victim were obtained using augmentations, dataset concatenation and KD techniques. The final results for MoViNet-A2 Base were obtained using augmentations and dataset concatenation.

|Victim|Clone|Top-5 Accuracy|Number of Queries|
|-------|------|-----------------|-|
|Video Swin Transformer|R(2+1)D|54.8|~4L|
|MoViNet-A2 Base|R(2+1)D|67.3|~4L|

### Black Box setting
|Technique|Top-5 Accuracy|Top-1 Accuracy|
|----------------------|-|-|
|Random normal sampling with ResNet3D|1.26|0.27|
|Training generator along with clone with ResNet3D|2.69|0.41|
|Training conditional GAN independently with ResNet3D|**4.85**|**0.84**|
|Stacking affine-transformed images with R(2+1)D|1.22|0.30|

The final experiment for Video Swin Transformer victim was using stacked affine-transformed images with R(2+1)D. But, we obtained results which were against our expectations. Hence we trained the first approach for more time in case of both victims.

|Victim|Clone|Top-5 Accuracy|Number of Queries|
|-------|------|-----------------|-|
|Video Swin Transformer| R(2+1)D|4.85|~1M|
|MoViNet-A2 Base| R(2+1)D|4.13|~1M|


## Citations
```
@article{carreira2019short,
  title={A short note on the kinetics-700 human action dataset},
  author={Carreira, Joao and Noland, Eric and Hillier, Chloe and Zisserman, Andrew},
  journal={arXiv preprint arXiv:1907.06987},
  year={2019}
}
@article{liu2021video,
  title={Video Swin Transformer},
  author={Liu, Ze and Ning, Jia and Cao, Yue and Wei, Yixuan and Zhang, Zheng and Lin, Stephen and Hu, Han},
  journal={arXiv preprint arXiv:2106.13230},
  year={2021}
}
@article{kondratyuk2021movinets,
  title={MoViNets: Mobile Video Networks for Efficient Video Recognition},
  author={Dan Kondratyuk, Liangzhe Yuan, Yandong Li, Li Zhang, Matthew Brown, and Boqing Gong},
  journal={arXiv preprint arXiv:2103.11511},
  year={2021}
}
```
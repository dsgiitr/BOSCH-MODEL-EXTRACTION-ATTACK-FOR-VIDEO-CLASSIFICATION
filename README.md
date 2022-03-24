# BOSCH-MODEL-EXTRACTION-ATTACK-FOR-VIDEO-CLASSIFICATION

We present our work done for the model extraction task in Inter IIT Tech Meet 10.0

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
    - our_model_weights_1
    - our_model_weights_2
    - ...

## Final Weights

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
    Uses c3d, a purely convalution based architechture, as a student with vanilla settings
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

A program that trains ImageNet and test results.

Contains the implementation of the ResNeXt and Inceotion_Resnet_v2.

Contains a pre-training ResNeXt. 

train model:

```sh
usage: train_model.py [-h] [--batch_size BATCH_SIZE]
                      [--pretrained_model PRETRAINED_MODEL] [--epochs EPOCHS]
                      [--CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES]                              [--epoch_size EPOCH_SIZE]
                      [--lr_schedule_file LR_SCHEDULE_FILE]
                      [--model_name MODEL_NAME] [--display_step DISPLAY_STEP]
                      [--evaluate_step EVALUATE_STEP]
                      [--logs_base_dir LOGS_BASE_DIR]
                      [--models_base_dir MODELS_BASE_DIR]
                      [--nrof_class NROF_CLASS] [--finetuning FINETUNING]
                      [--dic_loc DIC_LOC] [--only_weight ONLY_WEIGHT]
                      [--model_def MODEL_DEF]

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Number of images to process in a batch.
  --pretrained_model PRETRAINED_MODEL
                        Load a pretrained model before training starts.
  --epochs EPOCHS       Number of epochs to run.
  --CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
                        CUDA VISIBLE DEVICES
  --epoch_size EPOCH_SIZE
                        Number of batches per epoch.
  --lr_schedule_file LR_SCHEDULE_FILE
                        File containing the learning rate schedule
  --model_name MODEL_NAME
  --display_step DISPLAY_STEP
                        Step size showing the training situation
  --evaluate_step EVALUATE_STEP
                        Step size showing verification
  --logs_base_dir LOGS_BASE_DIR
                        Directory where to write event logs.
  --models_base_dir MODELS_BASE_DIR
                        Directory where to write trained models and
                        checkpoints.
  --nrof_class NROF_CLASS
                        Number of class category.
  --finetuning FINETUNING
                        Whether finetuning.
  --dic_loc DIC_LOC     Subclass dictionary location.
  --only_weight ONLY_WEIGHT
                        Whether only load pretrained model's weight.
  --model_def MODEL_DEF
                        Model definition. Points to a module containing the
                        definition of the inference graph.
```


test model:
```sh
usage: test_model.py [-h] [--pretrained_model PRETRAINED_MODEL]
                     [--CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES]
                     [--model_name MODEL_NAME] [--data_dir DATA_DIR]
                     [--nrof_class NROF_CLASS] [--crop_type CROP_TYPE] [--k K]
                     [--model_def MODEL_DEF]

optional arguments:
  -h, --help            show this help message and exit
  --pretrained_model PRETRAINED_MODEL
                        Load a pretrained model before training starts.
  --CUDA_VISIBLE_DEVICES CUDA_VISIBLE_DEVICES
                        CUDA VISIBLE DEVICES
  --model_name MODEL_NAME
  --data_dir DATA_DIR   Where to save the data.
  --nrof_class NROF_CLASS
                        Number of class category.
  --crop_type CROP_TYPE
                        Crop type for test.
  --k K                 k of random k crop.
  --model_def MODEL_DEF
                        Model definition. Points to a module containing the
                        definition of the inference graph.
```

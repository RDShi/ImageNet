
$$y=x+\sum_{i=1}^C T_i(x)$$


## Implementation details

### input
the input image is 224×224 randomly cropped from a resized image using the scale and aspect ratio augmentation.
The shortcuts are identity connections except for those increasing dimensions
which are projections. 

### train
We use SGD with a mini-batch size of 256 on 8 GPUs.
The weight decay is 0.0001 and the momentum is 0.9. 
We start from a learning rate of 0.1, and divide it by 10 for three times using the schedule. 
We adopt the weight initialization. 
We evaluate the error on the single 224×224 center crop from an image whose shorter side is 256.

### architecture:
Downsampling is done by stride-2 convolutions in the 3×3 layer of the first block in each stage.

在resnet的论文中可知：
The identity shortcuts can be directly used when the input and output are of the same dimensions. When the dimensions increase, we consider two options: (A) The shortcut still performs identity mapping, with extra zero entries padded for increasing dimensions. This option introduces no extra parameter; (B) The projection shortcut is used to match dimensions (done by 1×1 convolutions). For both options, when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2.

Conv-BN-ReLU
blocks of ResNeXt: 256d(in)-(256,1x1,128)-(3x3,32x4d)-(128,1x1,256)-256d(out)


result:
top 5 acc: 0.92708
top 1 acc: 0.7562


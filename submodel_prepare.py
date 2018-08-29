import numpy as np
from sklearn import metrics
import utils

thresold = 0

labp = np.fromfile('./data/resnext0810k_crop_predict.bin',dtype=np.int64)
logit = np.fromfile('./data/resnext0810k_crop_prob.bin',dtype=np.float32)
lab = np.fromfile('./data/resnext0810label.bin',dtype=np.int64)
con_mat = metrics.confusion_matrix(lab, labp)

n = len(lab)
labn = []
labpn = []
for i in range(n):
    if logit[i] > thresold:
        labn.append(lab[i])
        labpn.append(labp[i])
print(metrics.classification_report(labn, labpn, digits=4))


for i in range(1000):
    if float(con_mat[i,i])/sum(con_mat[:,i]) < 0.5 and float(con_mat[i,i])/sum(con_mat[i,:]) > 0.7:
        print(i)

i = 704
dic_loc = '/home/shirundong/ImageClassifiction/imagenet/data/'+str(i)
con_mat[np.argsort(con_mat[:,i])[::-1][:5], i]
wnid_labels, _ = utils.load_imagenet_meta('/data/srd/data/Image/ImageNet/ILSVRC2012_devkit_t12/data/meta.mat')
wnid_labels[np.argsort(con_mat[:,i])[::-1][:5]].tofile(dic_loc+'.dic')
np.argsort(con_mat[:,i])[::-1][:5].tofile(dic_loc+'.bin')



import sys
import os
import time
import tensorflow as tf
import tflearn
from sklearn.metrics import accuracy_score 
import numpy as np
import argparse
import threading
import importlib
import utils


def main(args):

    if not os.path.isdir(args.data_dir):  # Create the data directory if it doesn't exist
        os.makedirs(args.data_dir)

    print('pid:'+str(os.getpid()))
    gpu_list = args.CUDA_VISIBLE_DEVICES.split(",")
    os.environ['CUDA_VISIBLE_DEVICES']= args.CUDA_VISIBLE_DEVICES

    # tf.reset_default_graph()

    network = importlib.import_module(args.model_def)

    val_img_path = "/data/srd/data/Image/ImageNet/val"
    annotations = "/data/srd/data/Image/ImageNet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"

    x = tf.placeholder(dtype=tf.float32, shape=[None,224,224,3], name="input") #input

    pred = network.inference(x, n_class=args.nrof_class)
    prob = tf.nn.softmax(pred)

    restorer = tf.train.Saver()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.pretrained_model:
            restorer.restore(sess, args.pretrained_model)

        start_time = time.time()

        tflearn.is_training(False, session=sess)

        ## test process
        images_val = sorted(os.listdir(val_img_path))
        labels = utils.read_test_labels(annotations)
        label_predict = []
        label_prob = []
        top5_correct = []

        if args.crop_type == 'centre':
            ## centre crop
            start_time = time.time()
            for i, image in enumerate(images_val):
                test_im = utils.read_centre_crop(os.path.join(val_img_path, image))
                pr = sess.run(prob, feed_dict={x: test_im})
                label_prob.append(np.max(pr[0]))

                label_predict.append(np.argmax(pr[0]))
                top5 = np.argsort(pr[0])[-1:-6:-1]
                top5_correct.append(1 if labels[i] in top5 else 0)
                int_time = time.time()
                print('crop type: {} schedule:[{}/{}] Elapsed time: {}'.format('centre crop', i, 50000, utils.format_time(int_time - start_time)))

            print('centre crop top 5 acc:', np.mean(top5_correct))
            print('centre crop top 1 acc:', accuracy_score(labels, label_predict))
            np.array(labels).tofile(os.path.join(args.data_dir,args.model_name+'label.bin'))
            np.array(label_predict).tofile(os.path.join(args.data_dir,args.model_name+'centre_crop_predict.bin'))
            np.array(label_prob).tofile(os.path.join(args.data_dir,args.model_name+'centre_crop_prob.bin'))

            int_time = time.time()
            print('crop type: {} Elapsed time: {}'.format('centre crop', utils.format_time(int_time - start_time)))

        else:
            ## random k crop
            start_time = time.time()
            for i, image in enumerate(images_val):
                test_im = utils.read_k_patches(os.path.join(val_img_path, image), args.k)
                pr = sess.run(prob, feed_dict={x: test_im})
                pr = np.mean(pr,axis=0)
                label_prob.append(np.max(pr))

                label_predict.append(np.argmax(pr))
                top5 = np.argsort(pr)[-1:-6:-1]
                top5_correct.append(1 if labels[i] in top5 else 0)
                int_time = time.time()
                print('crop type: {} schedule:[{}/{}] Elapsed time: {}'.format('random k crop', i, 50000, utils.format_time(int_time - start_time)))

            print('random k crop top 5 acc:', np.mean(top5_correct))
            print('random k crop top 1 acc:', accuracy_score(labels, label_predict))
            np.array(label_predict).tofile(os.path.join(args.data_dir,args.model_name+'k_crop_predict.bin'))
            np.array(label_prob).tofile(os.path.join(args.data_dir,args.model_name+'k_crop_prob.bin'))

            int_time = time.time()
            print('crop type: {} Elapsed time: {}'.format('random k crop', utils.format_time(int_time - start_time)))




def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrained_model', type=str, help='Load a pretrained model before training starts.', default=None)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, help='CUDA VISIBLE DEVICES', default='9')
    parser.add_argument('--model_name', type=str, help='', default='test')
    parser.add_argument('--data_dir', type=str, help='Where to save the data.', default='./data')
    parser.add_argument('--nrof_class', type=str, help='Number of class category.', default=1000)
    parser.add_argument('--crop_type', type=str, help='Crop type for test.', default='centre')
    parser.add_argument('--k', type=int, help='k of random k crop.', default=3)
    parser.add_argument('--model_def', type=str, help='Model definition. Points to a module containing the definition of the inference graph.',
                         default='models.resnext')


    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



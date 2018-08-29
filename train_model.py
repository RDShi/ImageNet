import sys
import os
import time
import tensorflow as tf
import tflearn
import numpy as np
import argparse
import threading
import importlib
import utils


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # tower_grads = 
        # [[(grad0_gpu0, var0_gpu0),...,(gradN_gpu0, varN_gpu0)],...,[(grad0_gpuN, var0_gpuN),...,(gradN_gpuN, varN_gpuN)]]
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            if g == None: break
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

            # Average over the 'tower' dimension.
        if g == None: continue
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def lr_from_file(filename, epoch):
    learning_rate = 0.1
    try:
        with open(filename, 'r') as f:
            for line in f.readlines():
	        par = line.strip().split(':')
                e = int(par[0])
                lr = float(par[1])
                if e <= epoch:
                    learning_rate = lr
                else:
                    break
    finally:
        return learning_rate


def main(args):

    log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), args.model_name)
    if not os.path.isdir(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    model_dir = os.path.join(os.path.expanduser(args.models_base_dir), args.model_name)
    if not os.path.isdir(model_dir):  # Create the model directory if it doesn't exist
        os.makedirs(model_dir)
    model_path = model_dir+'/'+args.model_name+'.ckpt'

    print('pid:'+str(os.getpid()))
    gpu_list = args.CUDA_VISIBLE_DEVICES.split(",")
    os.environ['CUDA_VISIBLE_DEVICES']= args.CUDA_VISIBLE_DEVICES

    # tf.reset_default_graph()

    network = importlib.import_module(args.model_def)

    max_checkpoints = 3
    train_img_path = "/data/srd/data/Image/ImageNet/train"
    val_img_path = "/data/srd/data/Image/ImageNet/val"
    annotations = "/data/srd/data/Image/ImageNet/ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
    if args.epoch_size:
        num_batches = args.epoch_size
    else:
        # ts_size = utils.imagenet_size(train_img_path) # 1281167
        # num_batches = ts_size//args.batch_size
        num_batches = 1281167//args.batch_size

    if args.dic_loc:
        wnid_labels = np.fromfile(args.dic_loc+'.dic', dtype='<U9')
    else:
        wnid_labels, _ = utils.load_imagenet_meta('/data/srd/data/Image/ImageNet/ILSVRC2012_devkit_t12/data/meta.mat')

    dic_list = args.dic_loc+'.bin' if args.dic_loc else None

    x = tf.placeholder(dtype=tf.float32, shape=[None,224,224,3], name="input") #input
    y = tf.placeholder(tf.float32, [None, args.nrof_class]) # target
    lr = tf.placeholder(tf.float32) # learning rate

    # queue of examples being filled on the cpu
    with tf.device('/cpu:0'):
        q = tf.FIFOQueue(args.batch_size*3, [tf.float32, tf.float32], shapes=[[224, 224, 3], [args.nrof_class]])
        enqueue_op = q.enqueue_many([x, y])
        x_b, y_b = q.dequeue_many(args.batch_size)

    opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
    global_step = tf.Variable(0, trainable=False)
    epoch = tf.div(global_step, num_batches)

    num_gpus = len(gpu_list)
    x_splits = tf.split(x_b, num_gpus)
    y_splits = tf.split(y_b, num_gpus)
    tower_grads = []
    tower_cross_entropy = []
    tower_loss = []
    tower_accuracy = []
    counter = 0
    for d in gpu_list:
        with tf.device('/gpu:%s' % d):
            with tf.name_scope('%s_%s' % ('tower', counter)):
                pred = network.inference(x_splits[counter], n_class=args.nrof_class, finetuning=args.finetuning)
                cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y_splits[counter]), name='cross-entropy')
                l2_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='l2_loss')
                loss = l2_loss+cross_entropy
                # loss = cross_entropy
                correct = tf.equal(tf.argmax(pred, 1), tf.argmax(y_splits[counter], 1))
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
                counter += 1
                with tf.variable_scope("stuff"):
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
                    tower_cross_entropy.append(cross_entropy)
                    tower_loss.append(loss)
                    tower_accuracy.append(accuracy)
                    tf.get_variable_scope().reuse_variables()

    with tf.name_scope('loss'):
        mean_loss = tf.stack(axis=0, values=tower_loss)
        mean_loss = tf.reduce_mean(mean_loss, 0)
        tf.summary.scalar('train', mean_loss)
    with tf.name_scope('cross_entropy'):
        mean_cross_entropy = tf.stack(axis=0, values=tower_cross_entropy)
        mean_cross_entropy = tf.reduce_mean(mean_cross_entropy, 0)
        tf.summary.scalar('train', mean_cross_entropy)
    with tf.name_scope('accuracy'):
        mean_accuracy = tf.stack(axis=0, values=tower_accuracy)
        mean_accuracy = tf.reduce_mean(mean_accuracy, 0)
        tf.summary.scalar('train', mean_accuracy)

    # print(tower_grads)
    mean_grads = average_gradients(tower_grads)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = opt.apply_gradients(mean_grads, global_step=global_step, name="gradient_op")


    # merge ummaries to write them to file
    merged = tf.summary.merge_all()

    # checkpoint saver and restorer
    if args.only_weight:
        all_vars = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        all_vars += bn_moving_vars
        excl_vars = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
        to_restore = [item for item in all_vars if tflearn.utils.check_restore_tensor(item, excl_vars)]
    else: 
        all_vars = tf.global_variables()
        excl_vars = tf.get_collection(tf.GraphKeys.EXCL_RESTORE_VARS)
        to_restore = [item for item in all_vars if tflearn.utils.check_restore_tensor(item, excl_vars)]

    restorer = tf.train.Saver(var_list=to_restore, max_to_keep=max_checkpoints)

    saver = tf.train.Saver(max_to_keep=max_checkpoints)

    coord = tf.train.Coordinator()

    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True
    # tf.add_to_collection(tf.GraphKeys.GRAPH_CONFIG, config)
	
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        if args.pretrained_model:
            restorer.restore(sess, args.pretrained_model)

        # enqueuing batches procedure
        def enqueue_batches():
            while not coord.should_stop():
                im, l = utils.read_batch(args.batch_size, train_img_path, wnid_labels)
                sess.run(enqueue_op, feed_dict={x: im,y: l})

        # creating and starting parallel threads to fill the queue
        num_threads = 2
        for i in range(num_threads):
            t = threading.Thread(target=enqueue_batches)
            t.setDaemon(True)
            t.start()

        # operation to write logs for tensorboard visualization
        train_writer = tf.summary.FileWriter(log_dir, sess.graph)

        start_time = time.time()

        step = 0
		
        tflearn.is_training(True, session=sess)

        for e in range(sess.run(epoch), args.epochs):
            for i in range(num_batches):

                # learning rate schedule
                learning_rate = lr_from_file(args.lr_schedule_file, e)

                _, step = sess.run([train_op, global_step], feed_dict={lr: learning_rate})
                # train_writer.add_summary(summary, step)

                # display current training informations
                if step % args.display_step == 0:
                    tflearn.is_training(False, session=sess)
                    c, a, c_e = sess.run([mean_loss, mean_accuracy, mean_cross_entropy], feed_dict={lr: learning_rate})
                    int_time = time.time()
                    print('Epoch: {:03d} Step: {:09d} --- Loss: {:.7f} Cross Entropy: {:.07f} Training accuracy: {:.4f} Learning Rate: {} PID: {} Elapsed time: {}'
                          .format(e, step, c, c_e, a, learning_rate, os.getpid(), utils.format_time(int_time - start_time)))
                    result = sess.run(merged)
                    train_writer.add_summary(result, step)
                    tflearn.is_training(True, session=sess)

                # make test and evaluate validation accuracy
                if step % args.evaluate_step == 0:
                    tflearn.is_training(False, session=sess)
                    v_l_all = []
                    v_a_all = []
                    v_e_all = []
                    for _ in range(10):
                        val_im, val_cls = utils.read_validation_batch(args.batch_size, val_img_path, annotations, dic_list=dic_list)
                        v_c, v_a, v_e = sess.run([mean_loss, mean_accuracy, mean_cross_entropy], feed_dict={x_b:val_im, y_b:val_cls, lr: learning_rate})
                        v_l_all.append(v_c)
                        v_a_all.append(v_a)
                        v_e_all.append(v_e)
                    # intermediate time
                    int_time = time.time()
                    print ('Elapsed time: {}'.format(utils.format_time(int_time - start_time)))
                    print ('Loss: {:.7f} Cross Entropy: {:.07f} Validation accuracy: {:.04f}'.format(np.mean(v_l_all), np.mean(v_e_all), np.mean(v_a_all)))
                    # save weights to file
                    save_path = saver.save(sess, model_path)
                    print('Variables saved in file: %s' % save_path)
                    print('Logs saved in dir: %s' % log_dir)
                    summary = tf.Summary()
                    summary.value.add(tag='loss/val', simple_value=np.mean(v_l_all))
                    summary.value.add(tag='cross_entropy/val', simple_value=np.mean(v_e_all))
                    summary.value.add(tag='accuracy/val', simple_value=np.mean(v_a_all))
                    train_writer.add_summary(summary, step)
                    tflearn.is_training(True, session=sess)


        end_time = time.time()
        print ('Elapsed time: {}'.format(utils.format_time(end_time - start_time)))
        save_path = saver.save(sess, model_path)
        print('Variables saved in file: %s' % save_path)
        print('Logs saved in dir: %s' % log_dir)

        coord.request_stop()
        coord.join()
		

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=256)
    parser.add_argument('--pretrained_model', type=str, help='Load a pretrained model before training starts.', default=None)
    parser.add_argument('--epochs', type=int, help='Number of epochs to run.', default=90)
    parser.add_argument('--CUDA_VISIBLE_DEVICES', type=str, help='CUDA VISIBLE DEVICES', default='9')
    parser.add_argument('--epoch_size', type=int, help='Number of batches per epoch.', default=None)
    parser.add_argument('--lr_schedule_file', type=str, help='File containing the learning rate schedule', default='./data/learning_rate_schedule.txt')
    parser.add_argument('--model_name', type=str, help='', default='test')
    parser.add_argument('--display_step', type=int, help='Step size showing the training situation', default=10)
    parser.add_argument('--evaluate_step', type=int, help='Step size showing verification', default=500)
    parser.add_argument('--logs_base_dir', type=str, help='Directory where to write event logs.', default='~/logs/image/imagenet/')
    parser.add_argument('--models_base_dir', type=str, help='Directory where to write trained models and checkpoints.', default='~/models/image/imagenet/')
    parser.add_argument('--nrof_class', type=str, help='Number of class category.', default=1000)
    parser.add_argument('--finetuning', type=bool, help='Whether finetuning.', default=False)
    parser.add_argument('--dic_loc', type=str, help='Subclass dictionary location.', default=None)
    parser.add_argument('--only_weight', type=bool, help="Whether only load pretrained model's weight.", default=False)
    parser.add_argument('--model_def', type=str, help='Model definition. Points to a module containing the definition of the inference graph.',
                         default='models.resnext')

    return parser.parse_args(argv)
  

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))



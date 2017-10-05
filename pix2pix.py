'''
A simple inplementation of pix2pix
'''
import tensorflow as tf
import argparse
import glob
import os
import math
import time

EPS = 1e-12


parser = argparse.ArgumentParser()
parser.add_argument("input_dir", help="path to folder containing images")
# which_direction
parser.add_argument("--mode", choices=["train", "test", "export"])
parser.add_argument("output_dir", help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", default=None,
                    help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--max_steps", type=int, help="number of training steps")
parser.add_argument("--max_epochs", type=int, help="number of training epochs")
parser.add_argument("--summary_freq", type=int, default=100,
                    help="update summaries every summary_freq steps")
parser.add_argument("--progress_freq", type=int, default=50,
                    help="display progress every progress_freq steps")
parser.add_argument("--trace_freq", type=int, default=0,
                    help="trace execution every trace_freq steps")
parser.add_argument("--save_images_freq", type=int, default=0,
                    help="write current training images every save_images_freq steps")
parser.add_argument("--save_freq", type=int, default=5000,
                    help="save model every save_freq steps, 0 to disable")
parser.add_argument("--max_to_keep", type=int, default=3,
                    help="how many models you want to keep")

parser.add_argument("--aspect_ratio", type=float, default=1.0,
                    help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=1,
                    help="number of images in batch")
parser.add_argument("--which_direction", type=str,
                    default="AtoR", choices=["AtoR", "RtoA"])
parser.add_argument("--ngf", type=int, default=64,
                    help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64,
                    help="number of discriminator filters in first conv layer")
parser.add_argument("--lr", type=float, default=0.0002,
                    help="initial learning rate for adam")
parser.add_argument("--beta1", type=float, default=0.5,
                    help="momentum term of adam")
parser.add_argument("--l1_weight", type=float, default=100.0,
                    help="weight on L1 term for generator gradient")
parser.add_argument("--gan_weight", type=float, default=1.0,
                    help="weight on GAN term for generator gradient")

# export options
parser.add_argument("--output_filetype", default="png",
                    choices=["png", "jpeg"])
a = parser.parse_args()


def preprocess(img):
    # img has been scaled to (0,1), this func scales it to (-1,1)
    # print(img)
    with tf.name_scope('preprocess'):
        return img * 2 - 1


def depreprocess(img):
    # after calculation, img's scale is (-1,1), this func scales it to (0,1)
    with tf.name_scope('depreprocess'):
        return (img + 1) / 2


def create_conv_layer(input_batch, output_channels, stride):
    # 构造一个卷积层
    with tf.variable_scope('conv_layer'):
        # input_batch的数据形状是[batch_num, height, width, channel_num]
        input_channels = input_batch.get_shape()[3]
        filter = tf.get_variable('conv_filter',
                                 [4, 4, input_channels, output_channels], dtype=tf.float32,
                                 initializer=tf.random_normal_initializer(0, 0.02))
        # padded = tf.pad(input_batch, [[0, 0], [1, 1],
        #                               [1, 1], [0, 0]], mode="CONSTANT")
        # print(input_batch.get_shape())
        result = tf.nn.conv2d(
            input_batch, filter, [1, stride, stride, 1], padding='SAME')
        # print(result.get_shape())
        return result


def create_lrelu_layer(input_batch, a):
    # 构造一个leaky relu层
    with tf.name_scope('lrelu'):
        input = tf.identity(input_batch)
        return 0.5 * (1 + a) * input + 0.5 * (1 - a) * tf.abs(input)


def create_bn_layer(input_batch):
    # 构造batch_normalization层
    with tf.variable_scope('batch_norm'):
        input_channels = input_batch.get_shape()[3]
        gamma = tf.get_variable('bn_gamma',
                                [input_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1, 0.02))
        beta = tf.get_variable(
            'bn_beta', [input_channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        epsilon = 1e-5
        mean, variance = tf.nn.moments(input_batch, [0, 1, 2], keep_dims=False)
        result = tf.nn.batch_normalization(
            input_batch, mean, variance, beta, gamma, epsilon)
        return result


def create_deconv_layer(input_batch, output_channels, stride):
    with tf.variable_scope('deconv'):
        batch_num, input_height, input_width, input_channels = [int(d) for d in input_batch.get_shape()]
        filter = tf.get_variable('filter', [4, 4, output_channels, input_channels],
                                 dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # print('decove---',[batch_num, input_height * 2, input_width * 2, output_channels])
        result = tf.nn.conv2d_transpose(input_batch, filter, [batch_num, input_height * 2, input_width * 2, output_channels],
                                        [1, stride, stride, 1], padding="SAME")
        return result


def create_generator(generator_inputs, generator_output_channels):
    generator_layers = []
    # encoder_1 :[batch_num, 256, 256, image_channels] ---> [batch_num, 128, 128, ngf]
    # encoder_2 :[batch_num, 128, 128, ngf] ---> [batch_num, 64, 64, ngf * 2]
    # encoder_3 :[batch_num, 64, 64, ngf * 2] ---> [batch_num, 32, 32, ngf * 4]
    # encoder_4 :[batch_num, 32, 32, ngf * 4] ---> [batch_num, 16, 16, ngf * 8]
    # encoder_5 :[batch_num, 16, 16, ngf * 8] ---> [batch_num, 8, 8, ngf * 8]
    # encoder_6 :[batch_num, 8, 8, ngf * 8] ---> [batch_num, 4, 4, ngf * 8]
    # encoder_7 :[batch_num, 4, 4, ngf * 8] ---> [batch_num, 2, 2, ngf * 8]
    # encoder_8 :[batch_num, 2, 2, ngf * 8] ---> [batch_num, 1, 1, ngf * 8]

    # encoder 1层:
    with tf.variable_scope('encoder_1'):
        output = create_conv_layer(generator_inputs, a.ngf, stride=2)
        generator_layers.append(output)

    encoder_rest_layers_info = [a.ngf * 2, a.ngf * 4, a.ngf * 8, a.ngf * 8, a.ngf * 8, a.ngf * 8, a.ngf * 8]
    # encoder 2~8层:
    for i in range(len(encoder_rest_layers_info)):
        with tf.variable_scope('encoder_%d' % (len(generator_layers) + 1)):
            activated = create_lrelu_layer(generator_layers[-1], 0.2)
            convolved = create_conv_layer(activated, encoder_rest_layers_info[i], stride=2)
            bned = create_bn_layer(convolved)
            generator_layers.append(bned)
    # decoder_1 :[batch_num, 1, 1, ngf * 8] ---> [batch_num, 2, 2, ngf * 8 * 2]
    # decoder_2 :[batch_num, 2, 2, ngf * 8 * 2] ---> [batch_num, 4, 4, ngf * 8 * 2]
    # decoder_3 :[batch_num, 4, 4, ngf * 8 * 2] ---> [batch_num, 8, 8, ngf * 8 * 2]
    # decoder_4 :[batch_num, 8, 8, ngf * 8 * 2] ---> [batch_num, 16, 16, ngf * 8 * 2]
    # decoder_5 :[batch_num, 16, 16, ngf * 8 * 2] ---> [batch_num, 32, 32, ngf * 4 * 2]
    # decoder_6 :[batch_num, 32, 32, ngf * 4 * 2] ---> [batch_num, 64, 64, ngf * 2 * 2]
    # decoder_7 :[batch_num, 64, 64, ngf * 2 * 2] ---> [batch_num, 128, 128, ngf * 2]
    # decoder_8 :[batch_num, 128, 128, ngf * 2] ---> [batch_num, 256, 256, generator_output_channels]
    decoder_rest_layers_info = [(a.ngf * 8, 0.5), (a.ngf * 8, 0.5), (a.ngf * 8, 0.5), (a.ngf * 8, 0.0),
                                (a.ngf * 4, 0.0), (a.ngf * 2, 0.0), (a.ngf, 0.0)]
    layers_num_1 = len(generator_layers)
    # decoder 1~7层:
    for decoder_num, (output_channels, dropout) in enumerate(decoder_rest_layers_info):
        # U型的generator,就是把encoder得到的feature map也加到decoder里参与计算
        skip_layer_num = layers_num_1 - decoder_num - 1

        with tf.variable_scope('decoder_%d' % (decoder_num + 1)):
            if decoder_num == 0:
                input = generator_layers[-1]
            else:
                # print(generator_layers[-1].get_shape(),'----------------------',generator_layers[skip_layer_num].get_shape())
                input = tf.concat([generator_layers[-1], generator_layers[skip_layer_num]], axis=3)
            activated = tf.nn.relu(input)
            # 这里多连上前面对应层的结果，在deconv函数中input_channels是读出来的，没关系
            deconvolved = create_deconv_layer(activated, output_channels, stride=2)
            bned = create_bn_layer(deconvolved)
            if dropout > 0.0:
                bned = tf.nn.dropout(bned, keep_prob=1 - dropout)
            generator_layers.append(bned)

    # decoder 8层:
    with tf.variable_scope('decoder_8'):
        input = tf.concat([generator_layers[-1], generator_layers[0]], axis=3)
        activated = tf.nn.relu(input)
        deconvolved = create_deconv_layer(activated, generator_output_channels, stride=2)
        output = tf.tanh(deconvolved)
        generator_layers.append(output)
    return generator_layers[-1]


def create_discriminator(dis_inputs, dis_targets):
    discriminator_layers = []
    input = tf.concat([dis_inputs, dis_targets], axis=3)
    # layer_1 , [batch_num, 256, 256, input_channels * 2] ---> [batch_num, 128, 128, ndf]
    # layer_2 , [batch_num, 128, 128, ndf] ---> [batch_num, 64, 64, ndf * 2]
    # layer_3 , [batch_num, 64, 64, ndf * 2] ---> [batch_num, 32, 32, ndf * 4]
    # layer_4 , [batch_num, 32, 32, ndf * 4] ---> [batch_num, 31, 31, ndf * 8]
    # layer_5 , [batch_num, 31, 31, ndf * 8] ---> [batch_num, 30, 30, 1]

    # discriminator 1层:
    with tf.variable_scope('layer_1'):
        convolved = create_conv_layer(input, a.ndf, stride=2)
        activated = create_lrelu_layer(convolved, 0.2)
        discriminator_layers.append(activated)

    # discriminator 2~4层:
    dis_rest_layers_info = [a.ndf * 2, a.ndf * 4, a.ndf * 8]
    for i in range(len(dis_rest_layers_info)):
        with tf.variable_scope('layer_%d' % (i + 2)):
            stride = 1 if i == 2 else 2
            convolved = create_conv_layer(discriminator_layers[-1], dis_rest_layers_info[i], stride=stride)
            bned = create_bn_layer(convolved)
            activated = create_lrelu_layer(bned, 0.2)
            discriminator_layers.append(activated)

    # discriminator 5层:
    with tf.variable_scope('layer_5'):
        convolved = create_conv_layer(discriminator_layers[-1], 1, stride=1)
        activated = tf.sigmoid(convolved)
        discriminator_layers.append(activated)
    return discriminator_layers[-1]


def create_model(inputs, targets):

    # 生成Generator
    with tf.variable_scope('generator'):
        generator_output_channels = int(targets.get_shape()[-1])
        generator_outputs = create_generator(inputs, generator_output_channels)

    # 生成两个共享参数的Discriminator，因为要一边检测真的一边检测假的
    # D的结果是[batch_num, 30, 30, 1]
    with tf.name_scope('real_discriminator'):
        with tf.variable_scope('discriminator'):
            dis_real = create_discriminator(inputs, targets)

    with tf.name_scope('fake_discriminator'):
        with tf.variable_scope('discriminator', reuse=True):
            dis_fake = create_discriminator(inputs, generator_outputs)

    # 定义G的损失，分为通过D的损失和L1损失两部分
    with tf.name_scope('generator_loss'):
        gen_D_loss = tf.reduce_mean(-tf.log(dis_fake + EPS))
        gen_L1_loss = tf.reduce_mean(tf.abs(targets - generator_outputs))
        gen_total_loss = gen_D_loss * a.gan_weight + gen_L1_loss * a.l1_weight

    # 定义D的损失
    with tf.name_scope('discriminator_loss'):
        dis_loss = tf.reduce_mean(-(tf.log(dis_real + EPS) + tf.log(1 - dis_fake + EPS)))

    # GAN的算法是先算D再算G，所以这里先定义D的train
    with tf.name_scope('discriminator_train'):
        dis_vars = [var for var in tf.trainable_variables() if var.name.startswith('discriminator')]
        dis_optimizor = tf.train.AdamOptimizer(a.lr, a.beta1)
        dis_grads_vars = dis_optimizor.compute_gradients(dis_loss, dis_vars)
        dis_train = dis_optimizor.apply_gradients(dis_grads_vars)

    # 定义G的train
    with tf.name_scope('generator_train'):
        with tf.control_dependencies([dis_train]):
            gen_vars = [var for var in tf.trainable_variables() if var.name.startswith('generator')]
            gen_optimizor = tf.train.AdamOptimizer(a.lr, a.beta1)
            gen_grads_vars = gen_optimizor.compute_gradients(gen_total_loss, gen_vars)
            gen_train = gen_optimizor.apply_gradients(gen_grads_vars)

    # ema = tf.train.ExponentialMovingAverage(0.99)
    # update_losses = ema.apply([gen_D_loss, gen_L1_loss, dis_loss])

    global_step = tf.train.get_or_create_global_step()
    increase_global_step = tf.assign(global_step, global_step + 1)

    return {
        'generator_outputs': generator_outputs,
        'dis_real': dis_real,
        'dis_fake': dis_fake,
        'gen_D_loss': gen_D_loss,
        'gen_L1_loss': gen_L1_loss,
        'gen_total_loss': gen_total_loss,
        'dis_loss': dis_loss,
        'gen_grads_vars': gen_grads_vars,
        'dis_grads_vars': dis_grads_vars,
        # gen_train的dependencies是dis_train，所以最后这个只有gen_train是可以的
        'total_train': tf.group(increase_global_step, gen_train)
    }


def load_images():
    input_dir = a.input_dir
    if not input_dir or not os.path.exists(input_dir):
        raise Exception('input dir does not exist')
    input_paths = glob.glob(os.path.join(input_dir, '*.jpg'))
    with tf.name_scope('load_images'):
        path_queue = tf.train.string_input_producer(input_paths, shuffle=a.mode == 'train')
        reader = tf.WholeFileReader()
        paths, contents = reader.read(path_queue)
        raw_input = tf.image.decode_jpeg(contents)
        # 将图片数据归一化到(0,1)
        raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

        # assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
        # with tf.control_dependencies([assertion]):
        #     raw_input = tf.identity(raw_input)

        raw_input.set_shape([256, 512, 3])
        # print(raw_input)
        paths = [paths]
        width = raw_input.get_shape()[1]
        # print(width)
        real_images = preprocess(raw_input[:, :width // 2, :])
        abstract_images = preprocess(raw_input[:, width // 2:, :])
    if a.which_direction == 'AtoR':
        inputs, targets = abstract_images, real_images
    elif a.which_direction == 'RtoA':
        inputs, targets = real_images, abstract_images
    path_batch, input_batch, target_batch = tf.train.batch([paths, inputs, targets], batch_size=a.batch_size)
    steps_per_epoch = int(math.ceil(len(input_paths) / a.batch_size))

    return {
        'paths': path_batch,
        'inputs': input_batch,
        'targets': target_batch,
        'count': len(input_paths),
        'steps_per_epoch': steps_per_epoch,
    }


def main():

    output_dir = a.output_dir
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    input_data = load_images()
    inputs = input_data['inputs']
    targets = input_data['targets']
    model = create_model(inputs, targets)

    inputs_images = tf.image.convert_image_dtype(depreprocess(inputs), dtype=tf.uint8, saturate=True)
    targets_images = tf.image.convert_image_dtype(depreprocess(targets), dtype=tf.uint8, saturate=True)
    outputs_images = tf.image.convert_image_dtype(depreprocess(model['generator_outputs']), dtype=tf.uint8, saturate=True)
    # 写summary
    with tf.name_scope('loss_summary'):
        tf.summary.scalar('gen_D_loss', model['gen_D_loss'])
        tf.summary.scalar('gen_L1_loss', model['gen_L1_loss'])
        tf.summary.scalar('gen_total_loss', model['gen_total_loss'])
        tf.summary.scalar('dis_loss', model['dis_loss'])
    with tf.name_scope('input_summary'):
        tf.summary.image('inputs', inputs_images)
    with tf.name_scope('target_summary'):
        tf.summary.image('targets', targets_images)
    with tf.name_scope('output_summary'):
        tf.summary.image('outputs', outputs_images)

    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name + '/values', var)
    for grad, var in model['gen_grads_vars'] + model['dis_grads_vars']:
        tf.summary.histogram(var.op.name + '/values', var)
        tf.summary.histogram(var.op.name + '/gradients', grad)

    with tf.name_scope('parameter_num'):
        parameter_num = tf.reduce_sum([tf.reduce_prod(tf.shape(var_1)) for var_1 in tf.trainable_variables()])

    with tf.name_scope('image_save'):
        image_save_fetch = {
            'paths': input_data['paths'],
            'inputs': tf.map_fn(tf.image.encode_jpeg, inputs_images, dtype=tf.string),
            'outputs': tf.map_fn(tf.image.encode_jpeg, outputs_images, dtype=tf.string),
            'targets': tf.map_fn(tf.image.encode_jpeg, targets_images, dtype=tf.string),
        }

    saver = tf.train.Saver(max_to_keep=a.max_to_keep)
    sv = tf.train.Supervisor(logdir=a.output_dir, save_summaries_secs=0, saver=None)

    # 下面是sess部分
    with sv.managed_session() as sess:
        print('Total parameters count: ', sess.run(parameter_num))
        if a.checkpoint is not None:
            print('Load model from latest checkpoint...')
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)
        if a.max_epochs is not None:
            max_steps = a.max_epochs * input_data['steps_per_epoch']
        if a.max_steps is not None:
            max_steps = a.max_steps

        if a.mode == 'train':
            start_time = time.time()

            def should(freq, step):
                return freq > 0 and ((step + 1) % freq == 0 or step + 1 == max_steps)
            for step in range(max_steps):
                options = None
                run_metadata = tf.RunMetadata()

                fetch = {'train': model['total_train'],
                         'global_step': sv.global_step}
                if should(a.progress_freq, step):
                    fetch['dis_loss'] = model['dis_loss']
                    fetch['gen_D_loss'] = model['gen_D_loss']
                    fetch['gen_L1_loss'] = model['gen_L1_loss']
                    fetch['gen_total_loss'] = model['gen_total_loss']
                if should(a.summary_freq, step):
                    fetch['summary'] = sv.summary_op

                result = sess.run(fetch, options=options, run_metadata=run_metadata)

                if should(a.summary_freq, step):
                    print('Record summaries...')
                    sv.summary_writer.add_summary(result['summary'], global_step=result['global_step'])

                if should(a.trace_freq, step):
                    print('Record trace...')
                    sv.summary_writer.add_run_metadata(run_metadata, 'step_%d' % result['global_step'])

                if should(a.progress_freq, step):
                    print('\n========== Current Progress ==========\n')
                    epoch_num = int(math.ceil(result['global_step'] / input_data['steps_per_epoch']))
                    epoch_step_num = int((result['global_step'] - 1) % input_data['steps_per_epoch'] + 1)
                    average_step_time = (time.time() - start_time) / result['global_step']
                    remain_time = math.ceil((max_steps - result['global_step']) * average_step_time / 60)
                    print('current at epoch: %d, step: %d, average step time: %.4fs, minutes to finish: %dm'
                          % (epoch_num, epoch_step_num, average_step_time, remain_time))
                    print('dis_loss: ', result['dis_loss'])
                    print('gen_D_loss: ', result['gen_D_loss'])
                    print('gen_L1_loss: ', result['gen_L1_loss'])
                    print('gen_total_loss: ', result['gen_total_loss'])
                    print('\n======================================\n')

                if should(a.save_freq, step):
                    print('Save model...')
                    saver.save(sess, os.path.join(a.output_dir, 'model'), global_step=sv.global_step)

                if sv.should_stop():
                    break

        if a.mode == 'test':
            # 测试的时候记得在命令行参数里制定checkpoint路径，注意模式是test，注意模型大小设置一样

            test_results = sess.run(image_save_fetch)
            print(test_results['paths'])
            for i in range(input_data['steps_per_epoch']):
                for j in range(len(test_results['paths'])):
                    name, tail = os.path.splitext(os.path.basename(test_results['paths'][j][0].decode('utf-8')))
                    for kind in ['input', 'output', 'target']:
                        image_save_path = os.path.join(a.output_dir, name + '-' + kind + tail)
                        test_result_1 = test_results['%ss' % kind][j]
                        with open(image_save_path, 'wb') as f:
                            f.write(test_result_1)


main()
# test: python ./pix2pix.py F:\dataset\facades\tiny_test F:\projects\python\tf_test\pix2pix\output\test --mode test --batch_size 4 --checkpoint F:\projects\python\tf_test\pix2pix\output --ngf 16 --ndf 16
# train: python ./pix2pix.py F:\dataset\facades\tiny_train F:\projects\python\tf_test\pix2pix\output\ --mode train --max_steps 500 --summary_freq 10 --progress_freq 10 --batch_size 4 --ngf 16 --ndf 16
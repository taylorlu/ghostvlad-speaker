import tensorflow as tf
import numpy as np
weight_decay = 1e-4

def identity_block_2D(input_tensor, kernel_size, filters, stage, block, trainable=True):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    with tf.variable_scope(conv_name_1) as scope:
        x = tf.layers.conv2d(input_tensor, filters1, [1, 1],
                        kernel_initializer=tf.orthogonal_initializer(),
                        use_bias=False,
                        trainable=trainable,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn')(x, training=trainable)

    with tf.variable_scope('activation'):
        x = tf.nn.relu(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    with tf.variable_scope(conv_name_2) as scope:
        x = tf.layers.conv2d(x, filters2, [kernel_size, kernel_size],
                        padding='same',
                        kernel_initializer=tf.orthogonal_initializer(),
                        use_bias=False,
                        trainable=trainable,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn')(x, training=trainable)

    with tf.variable_scope('activation'):
        x = tf.nn.relu(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    with tf.variable_scope(conv_name_3) as scope:
        x = tf.layers.conv2d(x, filters3, [1, 1],
                        kernel_initializer=tf.orthogonal_initializer(),
                        use_bias=False,
                        trainable=trainable,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn')(x, training=trainable)

    with tf.variable_scope('add'):
        x = x + input_tensor
        x = tf.nn.relu(x)
    return x


def conv_block_2D(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), trainable=True):
    """A block that has a conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_1 = 'conv' + str(stage) + '_' + str(block) + '_1x1_reduce'
    with tf.variable_scope(conv_name_1) as scope:
        x = tf.layers.conv2d(input_tensor, filters1, [1, 1],
                        strides=strides,
                        kernel_initializer=tf.orthogonal_initializer(),
                        use_bias=False,
                        trainable=trainable,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn')(x, training=trainable)

    with tf.variable_scope('activation'):
        x = tf.nn.relu(x)

    conv_name_2 = 'conv' + str(stage) + '_' + str(block) + '_3x3'
    with tf.variable_scope(conv_name_2) as scope:
        x = tf.layers.conv2d(x, filters2, [kernel_size, kernel_size], padding='same',
                        kernel_initializer=tf.orthogonal_initializer(),
                        use_bias=False,
                        trainable=trainable,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn')(x, training=trainable)

    with tf.variable_scope('activation'):
        x = tf.nn.relu(x)

    conv_name_3 = 'conv' + str(stage) + '_' + str(block) + '_1x1_increase'
    with tf.variable_scope(conv_name_3) as scope:
        x = tf.layers.conv2d(x, filters3, [1, 1],
                        kernel_initializer=tf.orthogonal_initializer(),
                        use_bias=False,
                        trainable=trainable,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn')(x, training=trainable)

    conv_name_4 = 'conv' + str(stage) + '_' + str(block) + '_1x1_proj'
    with tf.variable_scope(conv_name_4) as scope:
        shortcut = tf.layers.conv2d(input_tensor, filters3, [1, 1], strides=strides,
                        kernel_initializer=tf.orthogonal_initializer(),
                        use_bias=False,
                        trainable=trainable,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
        shortcut = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn')(shortcut, training=trainable)

    with tf.variable_scope('add'):
        x = x + shortcut
        x = tf.nn.relu(x)
    return x


def resnet_2D_v1(inputs, trainable=True):
    bn_axis = 3

    # ===============================================
    #            Convolution Block 1
    # ===============================================
    with tf.variable_scope('conv1_1/3x3_s1'):
        x1 = tf.layers.conv2d(inputs, 64, [7, 7],
                        kernel_initializer=tf.orthogonal_initializer(),
                        use_bias=False,
                        trainable=trainable,
                        kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                        padding='same')
        x1 = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn')(x1, training=trainable)

    x1 = tf.nn.relu(x1)
    x1 = tf.layers.max_pooling2d(x1, [2, 2], [2, 2])

    # ===============================================
    #            Convolution Section 2
    # ===============================================
    x2 = conv_block_2D(x1, 3, [48, 48, 96], stage=2, block='a', strides=(1, 1), trainable=trainable)
    x2 = identity_block_2D(x2, 3, [48, 48, 96], stage=2, block='b', trainable=trainable)

    # ===============================================
    #            Convolution Section 3
    # ===============================================
    x3 = conv_block_2D(x2, 3, [96, 96, 128], stage=3, block='a', trainable=trainable)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='b', trainable=trainable)
    x3 = identity_block_2D(x3, 3, [96, 96, 128], stage=3, block='c', trainable=trainable)
    # ===============================================
    #            Convolution Section 4
    # ===============================================
    x4 = conv_block_2D(x3, 3, [128, 128, 256], stage=4, block='a', trainable=trainable)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='b', trainable=trainable)
    x4 = identity_block_2D(x4, 3, [128, 128, 256], stage=4, block='c', trainable=trainable)
    # ===============================================
    #            Convolution Section 5
    # ===============================================
    x5 = conv_block_2D(x4, 3, [256, 256, 512], stage=5, block='a', trainable=trainable)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='b', trainable=trainable)
    x5 = identity_block_2D(x5, 3, [256, 256, 512], stage=5, block='c', trainable=trainable)

    with tf.variable_scope('mpool2'):
        y = tf.layers.max_pooling2d(x5, [3, 1], [2, 1])

    return y


if __name__ == '__main__':
    # assignValue()

    import preprocess
    params = {'dim': (257, None, 1),
            'nfft': 512,
            'min_slice': 720,
            'win_length': 400,
            'hop_length': 160,
            'n_classes': 5994,
            'sampling_rate': 16000,
            'normalize': True,
            }
    specs = preprocess.load_data(r'D:\PythonSpace\Speaker-Diarization\ghostvlad\4persons\a_1.wav', split=False, win_length=params['win_length'], sr=params['sampling_rate'],
                       hop_length=params['hop_length'], n_fft=params['nfft'],
                       min_slice=params['min_slice'])
    specs = np.expand_dims(np.expand_dims(specs[0], 0), -1)

    with tf.Session() as sess:
        inputs, y = resnet_2D_v1([1, 257, None, 1], mode='eval')
        sess.run(tf.global_variables_initializer())
        tf.train.Saver().restore(sess, "ckpt/model.ckpt")

        output = tf.get_default_graph().get_tensor_by_name("mpool2/max_pooling2d/MaxPool:0")
        output = sess.run(output, feed_dict={"input:0": specs})
        print(output)





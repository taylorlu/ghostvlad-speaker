import tensorflow as tf
import backbone
import math
import numpy as np

def assignValue(graph): # restore from pb
    g1=tf.Graph()
    with g1.as_default():
        alltensors = []
        tensorDict = {}

        excludes = ['/read', '/Relu', '/MaxPool', '/convolution', '/FusedBatchNorm_1',
                    '/add', 'input', '/BiasAdd', 'Max', 'sub', 'Exp', 'Sum', 'truediv',
                    '/ExpandDims', '/Reshape', 'sub_1', 'mul', 'Sum_1', '/l2_normalize', 'lambda_1/','/MatMul','/strided_slice']
        with tf.Session(graph=g1) as sess:
            with tf.gfile.FastGFile(r'D:\PythonSpace\GhostVLAD-TF\pb\ghostvlad.pb', 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def)
                for tensor in graph_def.node:
                    # print(tensor.name)
                    contains = False
                    for e in excludes:
                        if(e in tensor.name):
                            contains = True
                            break
                    if(not contains):
                        alltensors.append(tensor.name+':0')

                results = tf.import_graph_def(graph_def, return_elements=alltensors)
                for i,result in enumerate(results):
                    tensorDict[alltensors[i]] = sess.run(result)
    sess.close()

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        for ele in tf.global_variables():
            name = ele.name
            if('/conv2d/' in ele.name): # name not the same
                name = ele.name.replace('/conv2d/', '/')
            sess.run(tf.assign(ele, tensorDict[name]))

        tf.train.Saver().save(sess, "ckpt/data.ckpt")


class GhostVLADModel(object):

    def __init__(self, args):
        self.init_learning_rate = args.get('init_learning_rate', 0.001)
        self.max_grad_norm = args.get('max_grad_norm', 50)
        self.decay_steps = args.get('decay_steps', 5000)
        self.decay_rate = args.get('decay_rate', 0.95)
        self.vlad_clusters = args.get('vlad_clusters', 8)
        self.ghost_clusters = args.get('ghost_clusters', 2)
        self.embedding_dim = args.get('embedding_dim', 512)
        self.num_class = args.get('num_class', 5994)
        self.l2_regularizer = tf.contrib.layers.l2_regularizer(1e-4)
        self._init_inference = False
        self._init_cost = False
        self._init_train = False


    def vladPooling(self, feat, cluster_score):
        # feat : bz x W x H x D, cluster_score: bz X W x H x clusters.
        num_features = feat.shape[-1]

        with tf.variable_scope('gvlad_pool'):
            cluster = tf.get_variable(name='centers',
                                    shape=[self.vlad_clusters+self.ghost_clusters, num_features],
                                    initializer=tf.orthogonal_initializer())

            # softmax normalization to get soft-assignment.
            # A : bz x W x H x clusters
            max_cluster_score = tf.keras.backend.max(cluster_score, -1, keepdims=True)
            exp_cluster_score = tf.keras.backend.exp(cluster_score - max_cluster_score)
            A = exp_cluster_score / tf.keras.backend.sum(exp_cluster_score, axis=-1, keepdims=True)

            # Now, need to compute the residual, self.cluster: clusters x D
            A = tf.keras.backend.expand_dims(A, -1)    # A : bz x W x H x clusters x 1
            feat_broadcast = tf.keras.backend.expand_dims(feat, -2)    # feat_broadcast : bz x W x H x 1 x D
            feat_res = feat_broadcast - cluster    # feat_res : bz x W x H x clusters x D
            weighted_res = tf.multiply(A, feat_res)     # weighted_res : bz x W x H x clusters x D
            cluster_res = tf.keras.backend.sum(weighted_res, [1, 2])

            cluster_res = cluster_res[:, :self.vlad_clusters, :]

            cluster_l2 = tf.nn.l2_normalize(cluster_res, -1)
            outputs = tf.reshape(cluster_l2, [-1, int(self.vlad_clusters) * int(num_features)])
        return outputs


    def get_arcface_logits(self, embeddings, labels, s=50.0, m=0.5, trainable=True):
        with tf.variable_scope('arcface'):
            weights = tf.get_variable(name='weights',
                                    shape=[embeddings.get_shape().as_list()[-1], self.num_class], dtype=tf.float32,
                                    initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                    regularizer=self.l2_regularizer,
                                    trainable=trainable)

            embeddings = tf.nn.l2_normalize(embeddings, axis=1)
            weights = tf.nn.l2_normalize(weights, axis=0)

            cos_m = math.cos(m)
            sin_m = math.sin(m)

            cos_theta = tf.matmul(embeddings, weights)
            sin_theta = tf.sqrt(tf.subtract(1.0, tf.square(cos_theta)))
            cos_m_theta = s * tf.subtract(tf.multiply(cos_theta, cos_m), tf.multiply(sin_theta, sin_m))

            threshold = math.cos(math.pi - m)

            cond_v = cos_theta - threshold
            cond = tf.cast(tf.nn.relu(cond_v), dtype=tf.bool)
            keep_val = s*(cos_theta - m*sin_m)
            cos_m_theta_temp = tf.where(cond, cos_m_theta, keep_val)
            mask = tf.one_hot(labels, depth=self.num_class)
            inv_mask = tf.subtract(1.0, mask)
            s_cos_theta = tf.multiply(s, cos_theta)
            logits = tf.add(tf.multiply(s_cos_theta, inv_mask), tf.multiply(cos_m_theta_temp, mask))
        return logits


    def vggvox_resnet2d_icassp(self, inputs, trainable=True):
        # ===============================================
        #                   parameters
        # ===============================================
        x = backbone.resnet_2D_v1(inputs, trainable=trainable)

        # ===============================================
        #            Fully Connected Block 1
        # ===============================================
        x_fc = tf.layers.conv2d(x, self.embedding_dim, [7, 1],
                        strides=[1, 1],
                        activation='relu',
                        kernel_initializer=tf.orthogonal_initializer(),
                        use_bias=True, trainable=trainable,
                        kernel_regularizer=self.l2_regularizer,
                        bias_regularizer=self.l2_regularizer,
                        name='x_fc')

        # ===============================================
        #            Feature Aggregation
        # ===============================================
        x_k_center = tf.layers.conv2d(x, self.vlad_clusters+self.ghost_clusters, [7, 1],
                                    strides=[1, 1],
                                    kernel_initializer=tf.orthogonal_initializer(),
                                    use_bias=True, trainable=trainable,
                                    kernel_regularizer=self.l2_regularizer,
                                    bias_regularizer=self.l2_regularizer,
                                    name='gvlad_center_assignment')

        x = self.vladPooling(x_fc, x_k_center)

        # ===============================================
        #            Fully Connected Block 2
        # ===============================================
        embeddings = tf.layers.dense(x, self.embedding_dim,
                           kernel_initializer=tf.orthogonal_initializer(),
                           use_bias=True, trainable=trainable,
                           kernel_regularizer=self.l2_regularizer,
                           bias_regularizer=self.l2_regularizer,
                           name='fc6')

        if(not trainable):
            embeddings = tf.nn.l2_normalize(embeddings, 1)

        return embeddings


    def init_inference(self, is_training=True):
        # feed inputs placeholder here
        self.inputs = tf.placeholder(tf.float32, [None, 257, None, 1], name='input')
        self._embeddings = self.vggvox_resnet2d_icassp(self.inputs, is_training)
        self._init_inference = True


    def init_cost(self):
        # ===============================================
        #                    ArcFace
        # ===============================================
        # feed labels placeholder here
        self.labels = tf.placeholder(tf.int32, name='label')
        logits = self.get_arcface_logits(self._embeddings, self.labels, s=50.0, m=0.5, trainable=True)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.labels))
        regular_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        self._cost = loss+regular_loss
        self._init_cost = True


    def init_train(self, train_vars=None):
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        self._lr = tf.train.exponential_decay(self.init_learning_rate, self._global_step,
                    self.decay_steps, self.decay_rate, staircase=True)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.AdamOptimizer(self._lr)
            grads, tvars = zip(*optimizer.compute_gradients(self._cost, train_vars))
            grads_clip, _ = tf.clip_by_global_norm(grads, self.max_grad_norm)
            self._train_op = optimizer.apply_gradients(zip(grads_clip, tvars), global_step=self._global_step)
        self._init_train = True


    def feed_dict(self, inputs, labels=None):
        """
        Constructs the feed dictionary from given inputs necessary to run
        an operations for the model.

        Args:
            inputs : 4D numpy array input spectrograms. Should be
                of shape [batch, 257, time, 1]
            labels : List of labels for each item in the batch. Each label
                should be a list of integers. If label=None does not feed the
                label placeholder (for e.g. inference only).

        Returns:
            A dictionary of placeholder keys and feed values.
        """
        feed_dict = {self.inputs : inputs}
        if(labels):
            label_dict = {self.labels : labels}
            feed_dict.update(label_dict)

        return feed_dict

    @property
    def embeddings(self):
        assert self._init_inference, "Must init inference module."
        return self._embeddings

    @property
    def cost(self):
        assert self._init_cost, "Must init cost module."
        return self._cost

    @property
    def train_op(self):
        assert self._init_train, "Must init train module."
        return self._train_op

    @property
    def global_step(self):
        assert self._init_train, "Must init train module."
        return self._global_step

    @property
    def learning_rate(self):
        assert self._init_train, "Must init train module."
        return self._lr


if __name__ == '__main__':
    vggvox_resnet2d_icassp(100, mode="train")
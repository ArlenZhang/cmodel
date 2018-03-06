"""
    Using convolutional net on CDTB arguments
    Author: ArlenZhang
    Date: 2018.3.5
"""
import time
from cmodel.utils import *
from cmodel.c_config import c_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''
    ================ CNN的convolution, pooling, full_conn部分设计 ==================
    parameters
        input : 输入的arguments组成的矩阵数据
        filters : 过滤器个数，用的越多越能捕捉更完整的特征
        k_size : 窗口尺寸
        stride : 窗口移动步长
        padding : 是否用0补全空位
        scope_name : 你懂得
    操作: 宽卷积 + 规范化
    既然每个单词的词向量维度一致，那就不要padding了
'''


def conv_relu(inputs, filters, k_rows, k_cols, stride, padding, scope_name):
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        # 输入数据是左边或者右边的1个argument组成的矩阵，channel在这里仍然是1,因为一个位置只有一个数字表示
        in_channels = inputs.shape[-1]

        # 初始化kernel数据,kernel窗口参数随机初始化,符合正太分布, 选用2行1列的窗口
        kernel = tf.get_variable('kernel', [k_rows, k_cols, in_channels, filters],
                                 initializer=tf.truncated_normal_initializer())

        # 这个偏移量不是学习到的，而是真的随机数，这样调整conv的结果然后再relu是一种类似平滑的方法
        bias = tf.get_variable('biases', [filters], initializer=tf.random_normal_initializer())

        # convolution在tensorflow中定义 待修改成 wide convolution
        conv = tf.nn.conv2d(inputs, kernel, strides=[1, stride, stride, 1], padding=padding)

        # relu 规范化过程定义
        result = tf.nn.relu(conv + bias, name=scope.name)
    return result


"""
    池化层, pooling过程也有convolution的过程
    parameters
        inputs : 输入矩阵（上一层Relu之后的结果作为输入）
        k_size : 窗口尺寸
        stride : pooling 窗口的步长
        padding : pooling过程的pad类型
        scope_name : 你懂得
"""


def maxpool(inputs, k_rows, k_cols, stride, padding='VALID', scope_name='pool'):
    """
        A method that does max pooling on inputs
        ksize[_, a, b, _] : a和b分别定义max_pooling窗口的行和列
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        pool = tf.nn.max_pool(inputs,
                              ksize=[1, k_rows, k_cols, 1],
                              strides=[1, stride, stride, 1],
                              padding=padding)
    return pool


"""
    全连接层，对pooling的结果到标签之间建立全连接
    parameters
        inputs : pooling的结果作为输入
        out_dim : 输出数据的维度, 标签个数
        scope_name : 你懂得
"""


def fully_connected(inputs, out_dim, scope_name='fc'):
    """
        A fully connected linear layer on inputs
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]  # 列数作为pooling层输出数据个数 ?
        w = tf.get_variable('weights', [in_dim, out_dim], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [out_dim], initializer=tf.constant_initializer(0.0))
        out = tf.matmul(inputs, w) + b
    return out


"""
    ================ 神经网络部分 ==================
"""


class ConvNet(object):
    def __init__(self):
        self.accuracy = self.summary_op = self.opt = self.loss = self.logits = self.test_init = self.train_init = \
            self.data_next = self.label = None
        self.config = c_config(True)  # 创建配置对象
        self.arg1_word_ids = self.arg2_word_ids = None
        self.arg1_word_embeddings = self.arg2_word_embeddings = None
        self.training = True
        self.skip_step = 20

    # 获取数据运用iterator机制，注意当前把padding之后的句子当作图像进行操作即可
    def get_data(self):
        with tf.name_scope('data'):
            # 获取能自动迭代的训练集，验证集，测试集
            train_data, val_data, test_data = get_dataset(self.config)
            iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
            self.arg1_word_ids, self.arg2_word_ids, self.label = iterator.get_next()
            # 在训练过程中将不断更新数据
            self.train_init = iterator.make_initializer(train_data)  # initializer for train_data
            self.test_init = iterator.make_initializer(test_data)  # initializer for test_data

    # word_embedding层
    def add_word_embeddings_op(self):
        with tf.variable_scope("words"):
            # 创建Wiki词向量
            _word_embeddings = tf.Variable(
                self.config.embeddings,
                name="_word_embeddings",
                dtype=tf.float32,
                trainable=self.config.train_embeddings
            )

            # 将当前批次中所有词汇 word_ids 的对应词向量封装在word_embeddings中
            self.arg1_word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.arg1_word_ids,
                                                               name="arg1_word_embeddings")
            self.arg2_word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.arg2_word_ids,
                                                               name="arg2_word_embeddings")
            # 第一个embedding的shape是 (?, 37, 50)相当于图片，先要变换成(?, 50, 37)便于convolution

            # 对文本呢矩阵reshape到[-1, word_d(?, 当前还是word_id，维度还不是50), sentence_len(词个数), 1]
            # reshape the image to make it work with tf.nn.conv2d
            # self.arg1_word_ids = tf.reshape(arg1_word_ids, shape=[-1, 1, self.config.arg1_length, 1])
            # self.arg2_word_ids = tf.reshape(arg2_word_ids, shape=[-1, 1, self.config.arg2_length, 1])

    def create_logits(self):
        """
            数据流 or 模型逻辑结构
        """
        # 建立两层convolution + max_pooling
        # 论元1
        conv_arg1 = conv_relu(inputs=self.arg1_word_embeddings,
                              filters=32,
                              k_rows=self.config.filter_row,  # 感受野为k_row * k_col
                              k_cols=self.config.filter_col,
                              stride=1,  # 步长
                              padding='SAME',  # 补
                              scope_name='arg1_conv')
        # 得到的将会是一列数据,按照批次进变换使得行数对应batch_size
        arg1_pool = maxpool(inputs=conv_arg1,
                            k_rows=self.config.pooling_row,
                            k_cols=self.config.arg1_pooling_col,
                            stride=1,
                            padding='VALID',  # means no padding
                            scope_name='arg1_pool')
        arg1_pool = tf.reshape(arg1_pool, [-1, self.config.dim_word])

        # 论元2
        conv_arg2 = conv_relu(inputs=self.arg2_word_embeddings,
                              filters=32,
                              k_rows=self.config.filter_row,  # 感受野为k_row * k_col
                              k_cols=self.config.filter_col,
                              stride=1,  # 步长
                              padding='SAME',  # 补
                              scope_name='arg2_conv')
        # 得到的将会是一列数据,按照批次进变换使得行数对应batch_size
        arg2_pool = maxpool(inputs=conv_arg2,
                            k_rows=self.config.pooling_row,
                            k_cols=self.config.arg2_pooling_col,
                            stride=1,
                            padding='VALID',  # means no padding
                            scope_name='arg2_pool')
        arg2_pool = tf.reshape(arg2_pool, [-1, self.config.dim_word])

        # 对输出值进行拼接
        concat_result = tf.concat((arg1_pool, arg2_pool), axis=1)  # axis=1代表两个矩阵水平方向拼接

        # 全连接
        full_c = fully_connected(concat_result, self.config.hidden_size, 'fc')

        # 避免过拟合处理方案
        dropout = tf.nn.dropout(tf.nn.relu(full_c), self.config.keep_prob, name='relu_dropout')

        # 建立logits
        self.logits = fully_connected(dropout, self.config.ntags, 'logits')

    def create_loss(self):
        """
            定义损失函数
        """
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

    def create_optimize(self):
        """
            反响传播，训练最优参数
        """
        self.opt = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss, global_step=self.config.gstep)

    def summary(self):
        """
            数据的展示汇总
        """
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('accuracy', self.accuracy)
            tf.summary.histogram('histogram loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        """
            对一批数据的评估
        """
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    def build(self):
        """
            Build the computation graph
        """
        self.get_data()
        self.add_word_embeddings_op()
        self.create_logits()
        self.create_loss()
        self.create_optimize()
        self.eval()
        self.summary()

    def train_one_epoch(self, sess, saver, init, writer, epoch, step):
        """
            一次完整的训练过程
        :param sess:
        :param saver:
        :param init:
        :param writer:
        :param epoch:
        :param step:
        :return:
        """
        start_time = time.time()
        sess.run(init)
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                _, l, summaries = sess.run([self.opt, self.loss, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess, '../../checkpoints/Demo15/mnist-convnet', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess, init, writer, epoch, step):
        start_time = time.time()
        sess.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass

        print('Accuracy at epoch {0}: {1} '.format(epoch, total_correct_preds / self.config.n_test))
        print('Took: {0} seconds'.format(time.time() - start_time))

    def train(self, n_epochs):
        """
            The train function alternates between training one epoch and evaluating
        """
        writer = tf.summary.FileWriter('../../graphs', tf.get_default_graph())

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('../../checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

            step = self.config.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess, saver, self.train_init, writer, epoch, step)
                self.eval_once(sess, self.test_init, writer, epoch, step)
        writer.close()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    model.train(n_epochs=10)
    """
        tensorboard --logdir="graphs/Demo15"
        http://ArlenIAC:6006
    """

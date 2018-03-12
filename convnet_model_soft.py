"""
    Using convolutional net on CDTB arguments
    Author: ArlenZhang
    Date: 2018.3.5
"""
import time
from cmodel.utils import *
from cmodel.c_config import c_config
import matplotlib.pyplot as plt


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


def fully_connected(inputs, hidden_size, scope_name='fc'):
    """
        A fully connected linear layer on inputs
    """
    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE) as scope:
        in_dim = inputs.shape[-1]  # 列数作为pooling层输出数据个数 ?
        w = tf.get_variable('weights', [in_dim, hidden_size], initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', [hidden_size], initializer=tf.constant_initializer(0.0))
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
        self.arg1_word_ids = None
        self.arg2_word_ids = None
        self.arg1_pos_ids = None
        self.arg2_pos_ids = None
        self.arg1_mask = self.arg2_mask = None
        self.arg1_word_embeddings = self.arg2_word_embeddings = None
        self.training = True
        self.skip_step = 20

    # 获取数据运用iterator机制，注意当前把padding之后的句子当作图像进行操作即可
    def get_data(self):
        with tf.name_scope('data'):
            # 获取能自动迭代的训练集，验证集，测试集
            train_data_, val_data, test_data = get_dataset(self.config)
            iterator_ = tf.data.Iterator.from_structure(train_data_.output_types, train_data_.output_shapes)
            self.arg1_word_ids, self.arg2_word_ids, self.label, self.arg1_mask, self.arg2_mask,\
                self.arg1_pos_ids, self.arg2_pos_ids = iterator_.get_next()

            # 在训练过程中将不断更新数据
            self.train_init = iterator_.make_initializer(train_data_)  # initializer for train_data
            self.test_init = iterator_.make_initializer(test_data)  # initializer for test_data

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
            arg1_word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.arg1_word_ids,
                                                          name="arg1_word_embeddings")
            arg2_word_embeddings = tf.nn.embedding_lookup(_word_embeddings, self.arg2_word_ids,
                                                          name="arg2_word_embeddings")

            # 第一个embedding的shape是 (?, 37, 50)相当于图片，下面生成真正的mask矩阵对embedding输出进行变换
            print(self.arg1_mask.shape)
            print(self.arg2_mask.shape)

            # 进行padding with mask
            self.arg1_word_embeddings = tf.multiply(arg1_word_embeddings, self.arg1_mask)
            self.arg2_word_embeddings = tf.multiply(arg2_word_embeddings, self.arg2_mask)

            # 下面变换成(?, 37, 50, 1)便于convolution
            self.arg1_word_embeddings = tf.reshape(arg1_word_embeddings, shape=[-1, self.config.arg1_length,
                                                                                self.config.dim_word, 1])
            self.arg2_word_embeddings = tf.reshape(arg2_word_embeddings, shape=[-1, self.config.arg2_length,
                                                                                self.config.dim_word, 1])

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
                            k_rows=self.config.arg1_pooling_row,
                            k_cols=self.config.pooling_col,
                            stride=1,
                            padding='VALID',  # means no padding
                            scope_name='arg1_pool')
        # squeeze  上面得到的shape是(?, 1, 50, 32)
        arg1_pool = tf.squeeze(arg1_pool, [1])
        #  现在需要将pool的结果拼接成1维的数据
        feature_dim = arg1_pool.shape[1] * arg1_pool.shape[2]
        arg1_pool = tf.reshape(arg1_pool, [-1, feature_dim])
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
                            k_rows=self.config.arg2_pooling_row,
                            k_cols=self.config.pooling_col,
                            stride=1,
                            padding='VALID',  # means no padding
                            scope_name='arg2_pool')
        arg2_pool = tf.squeeze(arg2_pool, [1])
        feature_dim = arg2_pool.shape[1] * arg2_pool.shape[2]
        arg2_pool = tf.reshape(arg2_pool, [-1, feature_dim])

        # 对输出值进行拼接，axis=1代表两个矩阵水平方向拼接
        concat_result = tf.concat((arg1_pool, self.arg1_pos_ids, arg2_pool, self.arg2_pos_ids), axis=1)
        # 全连接
        full_c = fully_connected(concat_result, self.config.hidden_size, 'fc')
        # 隐藏层由50个神经元对应输出数据(?, 50)，每个输入(?, 100)到隐藏层50节点之间建立联系然后再映射到12个输出节点

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
            tf.summary.histogram('histogram_loss', self.loss)
            self.summary_op = tf.summary.merge_all()

    def eval(self):
        """
            对一批数据的评估
        """
        with tf.name_scope('predict'):
            preds = tf.nn.softmax(self.logits)
            print(preds)
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

    def train_one_epoch(self, sess_, saver, init, writer, epoch, step):
        """
            一次完整的训练过程
        :param sess_:
        :param saver:
        :param init:
        :param writer:
        :param epoch:
        :param step:
        :return:
        """
        start_time = time.time()
        sess_.run(init)  # 初始化数据，得到最初的迭代器
        self.training = True
        total_loss = 0
        n_batches = 0
        try:
            while True:
                # train opt是为了学习模型，不需要返回值。train loss是为了得到每次训练之后的loss值，统计数据。..
                _, l, summaries, temp = sess_.run([self.opt, self.loss, self.summary_op, self.arg1_word_embeddings])
                writer.add_summary(summaries, global_step=step)
                if (step + 1) % self.skip_step == 0:
                    print('Loss at step {0}: {1}'.format(step, l))
                step += 1
                total_loss += l
                n_batches += 1
        except tf.errors.OutOfRangeError:
            pass
        saver.save(sess_, '../checkpoints/sess_save', step)
        print('Average loss at epoch {0}: {1}'.format(epoch, total_loss / n_batches))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return step

    def eval_once(self, sess_, init, writer, epoch, step):
        start_time = time.time()
        # 初始化迭代器，回到数据首部
        sess_.run(init)
        self.training = False
        total_correct_preds = 0
        try:
            while True:
                accuracy_batch, summaries = sess_.run([self.accuracy, self.summary_op])
                writer.add_summary(summaries, global_step=step)
                total_correct_preds += accuracy_batch
        except tf.errors.OutOfRangeError:
            pass
        acc = total_correct_preds / self.config.count_test
        print('Accuracy at epoch {0}: {1} '.format(epoch, acc))
        print('Took: {0} seconds'.format(time.time() - start_time))
        return acc

    def train(self, n_epochs):
        """
            The train function alternates between training one epoch and evaluating
        """
        writer = tf.summary.FileWriter('../graphs', tf.get_default_graph())
        accs = []
        with tf.Session() as sess_:
            sess_.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            ckpt = tf.train.get_checkpoint_state(os.path.dirname('../checkpoints/sess_save'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess_, ckpt.model_checkpoint_path)

            step = self.config.gstep.eval()

            for epoch in range(n_epochs):
                step = self.train_one_epoch(sess_, saver, self.train_init, writer, epoch, step)
                temp_p = self.eval_once(sess_, self.test_init, writer, epoch, step)
                accs.append(temp_p)
        writer.close()
        # 返回准确率统计结果
        return accs

def do_draw(n_epochs, y):
    x = [x for x in range(1, n_epochs+1)]
    plt.figure(figsize=(16, 8), dpi=100)
    plt.plot(x, y, color="red", linestyle="-")
    plt.show()

if __name__ == '__main__':
    model = ConvNet()
    model.build()
    accs = model.train(n_epochs=20)
    do_draw(n_epochs=20, y=accs)
    """
        tensorboard --logdir="graphs/Demo15"
        http://ArlenIAC:6006
    """

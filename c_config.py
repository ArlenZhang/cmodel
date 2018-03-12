"""
    CNN模型配置文件
"""
import tensorflow as tf
import os
from model_utils.general_utils import get_logger
from model_utils.data_utils import get_trimmed_glove_vectors, load_vocab, get_processing_word
import pickle as pkl


class c_config:
    def __init__(self, load=False):
        # 超参数等固定数据的配置
        self.lr = 0.0003
        self.lr_method = "adam"
        self.lr_decay = 0.9
        self.batch_size = 10  # 10句
        self.hidden_size = 1200  # 全连接层只用一层，隐层输出就是对应到类别之间的比重 1/3

        self.keep_prob = tf.constant(0.75)
        self.gstep = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        self.ntags = 12  # 也就是12种粗粒度关系种类
        self.nepochs = 15
        self.dropout = float(0.5)
        self.clip = -1  # if negative, no clipping
        self.nepoch_no_imprv = 3

        # 词向量库信息配置
        self.filename_wiki = "../data/word2vec/CH.GigawordWiki.50.bin"
        self.dim_word = 50  # embeddings层单词的向量维度,根据词向量库的向量维度确定
        self.filename_trimmed = "../data/wiki.npz".format(self.dim_word)
        self.use_pretrained = True  # 使用预训练词向量
        self.train_embeddings = False

        # 论元长度，训练数据统计配置，瓷都已经在准备数据的时候计算出来，直接从pkl文件读取即可
        with open("../data/configuration.pkl", "rb") as f:
            config = pkl.load(f)
            self.arg1_length = int(config["len_l_ave"])
            self.arg2_length = int(config["len_r_ave"])
            self.sentence_length = self.arg1_length + self.arg2_length
            self.count_train = int(config["count_train"])
            self.count_test = int(config["count_test"])
            self.count_eval = 0  # 后期被修改

        # 训练与测试数据信息配置
        # 路径
        self.filename_dev = self.filename_test = "../data/test_data.pkl"
        self.filename_train = "../data/train_data.pkl"
        self.max_iter = None
        self.filename_data_set = "../data/dataset.pkl"

        # 数据本身
        self.vocab_words = self.vocab_tags = self.vocab_pos = None
        self.nwords = self.processing_word = self.processing_tag = None
        self.embeddings = None

        # 词汇, 标签, 包含词性
        self.filename_words = "../data/words.txt"
        self.filename_tags = "../data/tags.txt"
        self.filename_pos = "../data/pos.txt"

        # 模型参数和日志数据信息配置
        self.dir_output = "../data/c_model/"  # 模型参数存储位置
        self.dir_model = self.dir_output + "model.weights/"
        if not os.path.exists(self.dir_model):
            os.makedirs(self.dir_model)
        if not os.path.exists(self.dir_output):
            os.makedirs(self.dir_output)
        self.path_log = self.dir_output + "log.txt"
        self.logger = get_logger(self.path_log)

        # 不使用字符，以词为单位
        self.use_chars = False
        self.use_crf = False

        # CNN 特别参数配置
        self.filter_row = 1
        self.filter_col = 2
        self.pooling_col = 1
        self.arg1_pooling_row = self.arg1_length  # self.pooling_col 的值分arg1和arg2不同
        self.arg2_pooling_row = self.arg2_length

        # 代码运行判断中，对数据的使用进行标记，因为不分太长的数据被抛弃
        self.train_idx_in_file = []
        self.test_idx_in_file = []

        # 训练的时候要加载数据，设置阶段不需要加载
        if load:
            self.load()

    # 加载数据
    def load(self):
        self.vocab_words = load_vocab(self.filename_words)
        self.vocab_tags = load_vocab(self.filename_tags)
        self.vocab_pos = load_vocab(self.filename_pos)
        # 获取长度
        self.nwords = len(self.vocab_words)

        # 2. 获取单词处理函数：map str -> id
        self.processing_word = get_processing_word(self.vocab_words, lowercase=True, chars=self.use_chars)
        self.processing_tag = get_processing_word(self.vocab_tags, lowercase=False, allow_unk=False)

        # 3. 获取预训练的embedding
        self.embeddings = (get_trimmed_glove_vectors(self.filename_trimmed) if self.use_pretrained else None)

if __name__ == "__main__":
    config_t = c_config(True)
    print(config_t.vocab_pos["v"])

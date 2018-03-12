import os
import gzip
import shutil
import urllib
import jieba.posseg
import pickle as pkl
import numpy as np
import tensorflow as tf
from cmodel.c_config import c_config

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# loss优化
def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)

    def f1(): return 0.5 * tf.square(residual)

    def f2(): return delta * residual - 0.5 * tf.square(delta)

    return tf.cond(residual < delta, f1, f2)


def load_data(config_):
    with open(config_.filename_data_set, "rb") as f:
        dataset = pkl.load(f)
    return dataset[0], dataset[1], dataset[2]


# 返回sentence形式， label形式，考虑直接数据准备过程中进行padding操作
def parse_data(d_type, config_):
    print("parsing...")
    if d_type == "train":
        count = config_.count_train
        with open(config_.filename_train, "rb") as f:
            data = pkl.load(f)
    else:
        count = config_.count_test
        with open(config_.filename_test, "rb") as f:
            data = pkl.load(f)
    idx_in_file = []
    # 最后得到的向量形式的训练数据和测试数据
    arg1_word_ids = []
    arg1_pos_ids = []
    arg2_word_ids = []
    arg2_pos_ids = []
    labels = []
    mask_matrix = [[], []]  # 简单记录每个训练句子的两个论元的实际长度
    # 对每一行记录进行解析
    for idx in range(len(data)):
        temp_batch1 = []
        temp_pos1 = []
        temp_batch2 = []
        temp_pos2 = []

        # 对论元进行分割 取词 padding操作
        word_list_arg1, tag_list1 = do_cut(data[idx][1])
        word_list_arg2, tag_list2 = do_cut(data[idx][2])

        # 当前训练语料中的论元组, 不考虑太长的句子
        if len(list(word_list_arg1)) > config_.arg1_length or len(list(word_list_arg2)) > config_.arg2_length:
            count -= 1
            continue
        # label 每一句代表两个 论元 分配一个论元关系标签即可
        if config_.processing_tag is not None:
            labels.append(config_.processing_tag(data[idx][3]))

        # mask matrix
        mask_matrix[0].append(len(word_list_arg1))
        mask_matrix[1].append(len(word_list_arg2))

        # arguments
        idx_in_file.append(idx)
        temp_index = 0
        # 对第一论元进行分割 取词 padding操作
        for index in range(len(word_list_arg1)):
            temp_index += 1
            item = word_list_arg1[index]
            item_pos = tag_list1[index]
            if item in config_.vocab_words.keys() and item_pos in config_.vocab_pos.keys():
                temp_batch1.append(config_.vocab_words[item])
                temp_pos1.append(config_.vocab_pos[item_pos])
            else:
                input("怎么还是存在未知词汇1！")
        # padding
        while temp_index < config_.arg1_length:
            temp_index += 1
            temp_batch1.append(0)
            temp_pos1.append(0)

        # 对第二论元进行分割 取词 padding操作
        temp_index = 0
        for index in range(len(word_list_arg2)):
            temp_index += 1
            item = word_list_arg2[index]
            item_pos = tag_list2[index]
            if item in config_.vocab_words.keys():
                temp_batch2.append(config_.vocab_words[item])
                temp_pos2.append(config_.vocab_pos[item_pos])
            else:
                input("怎么还是存在未知词汇2！")
        # padding
        while temp_index < config_.arg2_length:
            temp_index += 1
            temp_batch2.append(0)
            temp_pos2.append(0)

        # 生成元组
        arg1_word_ids.append(temp_batch1)
        arg2_word_ids.append(temp_batch2)
        arg1_pos_ids.append(temp_pos1)
        arg2_pos_ids.append(temp_pos2)

    # 写回config_ 中的 count
    if d_type == "train":
        config_.count_train = count
        config_.train_idx_in_file = idx_in_file
    else:
        config_.count_test = count
        config_.test_idx_in_file = idx_in_file
    return np.array(arg1_word_ids), np.array(arg2_word_ids), np.array(labels), np.array(arg1_pos_ids, dtype=np.float32)\
        , np.array(arg2_pos_ids, dtype=np.float32), np.array(mask_matrix)


# 返回words和tags
def do_cut(seq):
    words = []
    pos = []
    result = jieba.posseg.cut(seq)
    for word, p in result:
        words.append(word)
        pos.append(p)
    return words, pos


"""
    将数字标签转换成1维的向量表示
"""


def convert_label(n_class, labels):
    new_labels = np.zeros((len(labels), n_class))
    new_labels[np.arange(len(labels)), labels] = 1
    return new_labels


"""
    根据句子中词汇个数生成对应的mask 矩阵
    返回论元1和论元2的各自的matrix
    mask_matrix[0]和mask_matrix[1]分别代表论元1和2
"""


def convert_mask(mask_matrix, config_):
    arg1_mask, arg2_mask = None, None
    for idx_ in range(len(mask_matrix[0])):
        len_arg1, len_arg2 = mask_matrix[0][idx_], mask_matrix[1][idx_]
        arg1_mask_temp = np.ones((config_.arg1_length, config_.dim_word), dtype=np.float32)
        arg1_mask_temp[len_arg1:, :] = 0
        arg2_mask_temp = np.ones((config_.arg2_length, config_.dim_word), dtype=np.float32)
        arg2_mask_temp[len_arg2:, :] = 0
        # 拼接到整体
        # 怎么将数据以3维度的形式得到，因为简单的拼接就是两维数据

        if arg1_mask is None:
            arg1_mask = np.array([arg1_mask_temp])
            arg2_mask = np.array([arg2_mask_temp])
        else:
            arg1_mask = np.append(arg1_mask, [arg1_mask_temp], axis=0)
            arg2_mask = np.append(arg2_mask, [arg2_mask_temp], axis=0)
    return arg1_mask, arg2_mask


'''
    读取所有数据，训练集，验证集，测试集(930个句子，取随机的一半作为验证集) 
'''
def read_data(config_, validation_of_test=0.1):
    # 训练集
    train_arg1, train_arg2, train_labels, train_pos1, train_pos2, mask_matrix_train = parse_data('train', config_)
    # 标签格式转换
    train_labels = convert_label(config_.ntags, train_labels)

    # mask格式转换
    arg1_train_mask, arg2_train_mask = convert_mask(mask_matrix_train, config_)

    # 测试集和验证集
    test_v_arg1, test_v_arg2, test_v_labels, test_v_pos1, test_v_pos2, mask_matrix_test_val = \
        parse_data('test', config_)
    val_num = int(config_.count_test * validation_of_test)
    test_num = config_.count_test - val_num

    indices = np.random.permutation(config_.count_test)  # test文件中数据总数
    test_idx, val_idx = indices[:test_num], indices[test_num:]

    # 更新配置
    config_.count_test = test_num
    config_.count_eval = val_num  # 最新的测试集合 验证集树木写回配置文件
    # test_idx = np.array([i for i in range(0, test_num)])
    # val_idx = np.array([i for i in range(0, 1)])

    # =======================测试数据========================================
    test_arg1, test_arg2 = test_v_arg1[test_idx, :], test_v_arg2[test_idx, :]
    test_pos1, test_pos2 = test_v_pos1[test_idx, :], test_v_pos2[test_idx, :]
    mask_matrix_test = [mask_matrix_test_val[0][test_idx], mask_matrix_test_val[1][test_idx]]
    test_labels = test_v_labels[test_idx]
    # 标签数格式转换
    test_labels = convert_label(config_.ntags, test_labels)
    arg1_test_mask, arg2_test_mask = convert_mask(mask_matrix_test, config_)

    # =======================验证数据数据========================================
    val_arg1, val_arg2 = test_v_arg1[val_idx, :], test_v_arg2[val_idx, :]
    val_pos1, val_pos2 = test_v_pos1[val_idx, :], test_v_pos2[val_idx, :]
    mask_matrix_val = [mask_matrix_test_val[0][val_idx], mask_matrix_test_val[1][val_idx]]
    val_labels = test_v_labels[val_idx]
    # 标签数格式转换
    val_labels = convert_label(config_.ntags, val_labels)
    arg1_val_mask, arg2_val_mask = convert_mask(mask_matrix_val, config_)

    # 数据存储
    dataset_ = ((train_arg1, train_arg2, train_labels, arg1_train_mask, arg2_train_mask, train_pos1, train_pos2),
                (val_arg1, val_arg2, val_labels, arg1_val_mask, arg2_val_mask, val_pos1, val_pos2),
                (test_arg1, test_arg2, test_labels, arg1_test_mask, arg2_test_mask, test_pos1, test_pos2))
    with open(config_.filename_data_set, "wb") as f:
        pkl.dump(dataset_, f)

    return (train_arg1, train_arg2, train_labels, arg1_train_mask, arg2_train_mask, train_pos1, train_pos2), \
           (val_arg1, val_arg2, val_labels, arg1_val_mask, arg2_val_mask, val_pos1, val_pos2), \
           (test_arg1, test_arg2, test_labels, arg1_test_mask, arg2_test_mask, test_pos1, test_pos2)

# 从本地准备好的数据中获取训练和测试数据
def get_dataset(config_):
    print("form iterator")
    if os.path.exists(config_.filename_data_set):
        train_, val_, test_ = load_data(config_)
        # 更新配置
        config_.count_test = test_[0].shape[0]
        config_.count_val = val_[0].shape[0]
    else:
        train_, val_, test_ = read_data(config_, validation_of_test=0.5)

    # 将train中数据提取成(data1, data2, ..)的元组进行封装
    train_t_ = (train_[0], train_[1], train_[2], train_[3], train_[4], train_[5], train_[6])
    # Create datasets and iterator
    train_data_ = tf.data.Dataset.from_tensor_slices(train_t_)
    train_data_ = train_data_.batch(config_.batch_size)
    print("训练数据iterator构建完毕")
    # 将验证集转成一样的元组
    v_t_ = (val_[0], val_[1], val_[2], val_[3], val_[4], val_[5], val_[6])
    val_data_ = tf.data.Dataset.from_tensor_slices(v_t_)
    val_data_ = val_data_.batch(config_.batch_size)

    # 将test数据转成一样的元组
    test_t_ = (test_[0], test_[1], test_[2], test_[3], test_[4], test_[5], test_[6])
    test_data_ = tf.data.Dataset.from_tensor_slices(test_t_)
    test_data_ = test_data_.shuffle(config_.count_test)  # shuffle
    test_data_ = test_data_.batch(config_.batch_size)
    print("全部iterator 构建完成!")
    return train_data_, val_data_, test_data_

# 根据url下载指定文件到本地
def download_one_file(download_url,
                      local_dest,
                      expected_byte=None,
                      unzip_and_remove=False):
    """ 
        Download the file from download_url into local_dest
        if the file doesn't already exists.
        If expected_byte is provided, check if
        the downloaded file has the same number of bytes.
        If unzip_and_remove is True, unzip the file and remove the zip file
    """
    if os.path.exists(local_dest) or os.path.exists(local_dest[:-3]):
        print('%s already exists' % local_dest)
    else:
        print('Downloading %s' % download_url)
        local_file, _ = urllib.request.urlretrieve(download_url, local_dest)
        file_stat = os.stat(local_dest)
        if expected_byte:
            if file_stat.st_size == expected_byte:
                print('Successfully downloaded %s' % local_dest)
                if unzip_and_remove:
                    with gzip.open(local_dest, 'rb') as f_in, open(local_dest[:-3], 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                    os.remove(local_dest)
            else:
                print('The downloaded file has unexpected number of bytes')

if __name__ == "__main__":
    config = c_config(True)
    train, val, test = read_data(config, validation_of_test=0.5)
    # 打印前5个即有差别
    print(len(train[0]))
    for idx_t in range(5):
        print("句子下标: ", config.train_idx_in_file[idx_t])
        print("标签: ", train[2][idx_t])

    # 测试get_dataset
    train_d_l = (train[0], train[1], train[2])
    train_data = tf.data.Dataset.from_tensor_slices(train_d_l)
    # train_data = train_data.shuffle(config.n_test)  # shuffle
    train_data = train_data.batch(config.batch_size)
    iterator = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
    arg1_word_ids_, arg2_word_ids_, label_ = iterator.get_next()

    train_init = iterator.make_initializer(train_data)

    with tf.Session() as sess:
        print("输出标签：")
        sess.run(train_init)
        for _ in range(5):
            print(label_.eval())

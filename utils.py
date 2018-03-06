import os
import gzip
import shutil
import urllib
import jieba.posseg
import pickle as pkl
import numpy as np
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# loss优化
def huber_loss(labels, predictions, delta=14.0):
    residual = tf.abs(labels - predictions)

    def f1(): return 0.5 * tf.square(residual)

    def f2(): return delta * residual - 0.5 * tf.square(delta)

    return tf.cond(residual < delta, f1, f2)


# 返回sentence形式， label形式，考虑直接数据准备过程中进行padding操作
def parse_data(d_type, config):
    if d_type == "train":
        count = config.count_train
        with open(config.filename_train, "rb") as f:
            data = pkl.load(f)
    else:
        count = config.count_test
        with open(config.filename_test, "rb") as f:
            data = pkl.load(f)
    # 最后得到的向量形式的训练数据和测试数据
    arg1_word_ids = []
    arg1_pos_ids = []
    arg2_word_ids = []
    arg2_pos_ids = []
    labels = []
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
        if len(list(word_list_arg1)) > config.arg1_length or len(list(word_list_arg2)) > config.arg2_length:
            count -= 1
            continue

        temp_index = 0
        # 对第一论元进行分割 取词 padding操作
        for index in range(len(word_list_arg1)):
            temp_index += 1
            item = word_list_arg1[index]
            item_pos = tag_list1[index]
            if item in config.vocab_words.keys() and item_pos in config.vocab_pos.keys():
                temp_batch1.append(config.vocab_words[item])
                temp_pos1.append(config.vocab_pos[item_pos])
            else:
                input("怎么还是存在未知词汇1！")
        # padding
        while temp_index < config.arg1_length:
            temp_index += 1
            temp_batch1.append(0)
            temp_pos1.append(0)

        # 对第二论元进行分割 取词 padding操作
        temp_index = 0
        for index in range(len(word_list_arg2)):
            temp_index += 1
            item = word_list_arg2[index]
            item_pos = tag_list2[index]
            if item in config.vocab_words.keys():
                temp_batch2.append(config.vocab_words[item])
                temp_pos2.append(config.vocab_pos[item_pos])
            else:
                input("怎么还是存在未知词汇2！")
        # padding
        while temp_index < config.arg2_length:
            temp_index += 1
            temp_batch2.append(0)
            temp_pos2.append(0)

        # 生成元组
        arg1_word_ids.append(temp_batch1)
        arg2_word_ids.append(temp_batch2)
        arg1_pos_ids.append(temp_pos1)
        arg2_pos_ids.append(temp_pos2)

        # 每一句代表两个 论元 分配一个论元关系标签即可
        if config.processing_tag is not None:
            labels.append(config.processing_tag(data[idx][3]))

    # 写回config 中的 count
    if d_type == "train":
        config.count_train = count
    else:
        config.count_test = count
    return np.array(arg1_word_ids), np.array(arg2_word_ids), np.array(labels), np.array(arg1_pos_ids), np.array(
        arg2_pos_ids)


# 返回words和tags
def do_cut(seq):
    words = []
    pos = []
    result = jieba.posseg.cut(seq)
    for word, p in result:
        words.append(word)
        pos.append(p)
    return words, pos


'''
    读取所有数据，训练集，验证集，测试集(930个句子，取随机的一半作为验证集) 
'''


def read_data(config, validation_of_test=0.5):
    # 训练集
    train_arg1, train_arg2, train_labels, train_pos1, train_pos2 = parse_data('train', config)
    # 测试集和验证集
    test_v_arg1, test_v_arg2, test_v_labels, test_v_pos1, test_v_pos2 = parse_data('test', config)
    val_num = int(config.count_test * validation_of_test)
    test_num = config.count_test - val_num
    indices = np.random.permutation(config.count_test)  # test文件中数据总数
    test_idx, val_idx = indices[:test_num], indices[test_num:]

    test_arg1, test_arg2 = test_v_arg1[test_idx, :], test_v_arg2[test_idx, :]
    test_labels = test_v_labels[test_idx]
    test_pos1, test_pos2 = test_v_pos1[test_idx, :], test_v_pos2[test_idx, :]

    val_arg1, val_arg2, val_pos1, val_pos2 = \
        test_v_arg1[val_idx, :], test_v_arg2[val_idx, :], test_v_pos1[val_idx, :], \
        test_v_pos2[val_idx, :]
    val_labels = test_v_labels[val_idx]

    return (train_arg1, train_arg2, train_labels, train_pos1, train_pos2), \
           (val_arg1, val_arg2, val_labels, val_pos1, val_pos2), \
           (test_arg1, test_arg2, test_labels, test_pos1, test_pos2)


# 从本地准备好的数据中获取训练和测试数据
def get_dataset(config):
    train, val, test = read_data(config, validation_of_test=0.5)
    # 将train中数据提取成(data1, data2, ..)的元组进行封装
    train_d_l = (train[0], train[1], train[2])
    # train.shape: (1900, 50, sent_len)  # label.shape : (1900, 12)
    # Create datasets and iterator
    train_data = tf.data.Dataset.from_tensor_slices(train_d_l)
    train_data = train_data.shuffle(config.n_test)  # if you want to shuffle your data
    train_data = train_data.batch(config.batch_size)

    # 将验证集转成一样的元组
    v_d_l = (val[0], val[1], val[2])
    val_data = tf.data.Dataset.from_tensor_slices(v_d_l)
    val_data = val_data.batch(config.batch_size)

    # 将test数据转成一样的元组
    test_d_l = (test[0], test[1], test[2])
    test_data = tf.data.Dataset.from_tensor_slices(test_d_l)
    test_data = test_data.batch(config.batch_size)

    return train_data, val_data, test_data


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

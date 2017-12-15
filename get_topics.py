import os
import argparse
import json
from collections import namedtuple
import tensorflow as tf
import model.data as data
import model.model as m
import numpy as np


def get_id2vocab():
    id2vocab = {}
    f = open('data/20newsgroups.vocab', 'r')
    vocab = f.read().strip().split('\n')
    for item in enumerate(vocab):
        id2vocab[item[0]] = item[1]
    return id2vocab


def get_weights(model, dataset, params):
    id2vocab = get_id2vocab()
    var = [v for v in tf.trainable_variables() if v.name == "softmax/w:0"][0]

    with tf.Session(config=tf.ConfigProto(
        inter_op_parallelism_threads=params.num_cores,
        intra_op_parallelism_threads=params.num_cores,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )) as session:
        tf.local_variables_initializer().run()
        tf.global_variables_initializer().run()

        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state(params.model)
        saver.restore(session, ckpt.model_checkpoint_path)

        # get weights with shape of (hidden_size, vocab_size)

        W = session.run(var)

        for topic_index in range(W.shape[0]):
            if topic_index <= 10:
                all_words = W[topic_index][:]
                index = np.argpartition(all_words, -10)[-10:]
                words = [id2vocab[i] for i in index]
                print("Topic %d: %s" % (topic_index, " ".join(words)))

        # ckpt = tf.train.get_checkpoint_state(params.model)
        # saver.restore(session, ckpt.model_checkpoint_path)
        # print(W.eval().size)


def main(args):
    with open(os.path.join(args.model, 'params.json'), 'r') as f:
        params = json.loads(f.read())
    params.update(vars(args))
    params = namedtuple('Params', params.keys())(*params.values())

    dataset = data.Dataset(args.dataset)
    x = tf.placeholder(tf.int32, shape=(None, None), name='x')
    seq_lengths = tf.placeholder(tf.int32, shape=(None), name='seq_lengths')
    model = m.DocNADE(x, seq_lengths, params)
    get_weights(model, dataset, params)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        help='path to model output directory')
    parser.add_argument('--dataset', type=str, required=True,
                        help='path to the input dataset')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='the batch size')
    parser.add_argument('--num-cores', type=int, default=1,
                        help='the number of CPU cores to use')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())


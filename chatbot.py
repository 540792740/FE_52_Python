"""Most of the code comes from seq2seq tutorial. Binary for training conversation models and decoding from them.

Running this program without --decode will  tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint performs

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""

import math
import sys
import time
from datatools import *
from model import *
from tqdm import tqdm

tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_integer("batch_size", 256, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("numEpochs", 30, "Batch size to use during training.")
tf.app.flags.DEFINE_integer("size", 512, "Size of each model layer.")
tf.app.flags.DEFINE_integer("num_layers", 3, "Number of layers in the model.")
tf.app.flags.DEFINE_integer("en_vocab_size", 40000, "English vocabulary size.")
tf.app.flags.DEFINE_integer("en_de_seq_len", 20, "English vocabulary size.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 100, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_string("train_dir", './tmp', "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_integer("beam_size", 5, "How many training steps to do per checkpoint.")
tf.app.flags.DEFINE_boolean("beam_search", True, "Set to True for beam_search.")
tf.app.flags.DEFINE_boolean("decode", True, "Set to True for interactive decoding.")
FLAGS = tf.app.flags.FLAGS

def create_model(session, forward_only, beam_search, beam_size = 5):
    """Create translation model and initialize or load parameters in session."""
    model = Seq2SeqModel(
        FLAGS.en_vocab_size, FLAGS.en_vocab_size, [10, 10],
        FLAGS.size, FLAGS.num_layers, FLAGS.batch_size,
        FLAGS.learning_rate, forward_only=forward_only, beam_search=beam_search, beam_size=beam_size)
    ckpt = tf.train.latest_checkpoint(FLAGS.train_dir)
    model_path = 'chat_bot.ckpt-0'
    if forward_only:
        model.saver.restore(session, model_path)
    elif ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path):
        print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
        model.saver.restore(session, ckpt.model_checkpoint_path)
    else:
        print("Created model with fresh parameters.")
        session.run(tf.initialize_all_variables())
    return model

def train():
    # prepare dataset
    data_path = 'dataset.pkl'
    #database.pkl is a database made by cornell which includes conversations from movie scripts
    word2id, id2word, trainingSamples = loadDataset(data_path)
    with tf.Session() as sess:
        print("Creating %d layers of %d units." % (FLAGS.num_layers, FLAGS.size))
        model = create_model(sess, False, beam_search=False, beam_size=5)
        current_step = 0
        for e in range(FLAGS.numEpochs):
            print("----- Epoch {}/{} -----".format(e + 1, FLAGS.numEpochs))
            batches = getBatches(trainingSamples, FLAGS.batch_size, model.en_de_seq_len)
            for nextBatch in tqdm(batches, desc="Training"):
                _, step_loss = model.step(sess, nextBatch.encoderSeqs, nextBatch.decoderSeqs, nextBatch.targetSeqs,
                                          nextBatch.weights, goToken)
                current_step += 1
                if current_step % FLAGS.steps_per_checkpoint == 0:
                    perplexity = math.exp(float(step_loss)) if step_loss < 300 else float('inf')
                    tqdm.write("----- Step %d -- Loss %.2f -- Perplexity %.2f" % (current_step, step_loss, perplexity))
                    checkpoint_path = os.path.join(FLAGS.train_dir, "chat_bot.ckpt")
                    model.saver.save(sess, checkpoint_path, global_step=model.global_step)

def decode():
    with tf.Session() as sess:
        beam_size = FLAGS.beam_size
        beam_search = FLAGS.beam_search
        model = create_model(sess, True, beam_search=beam_search, beam_size=beam_size)
        model.batch_size = 1
        data_path = 'dataset.pkl'
        word2id, id2word, trainingSamples = loadDataset(data_path)

        if beam_search:
            sys.stdout.write("> ")
            sys.stdout.flush()
            sentence = sys.stdin.readline()
            while sentence:
                batch = sentence2enco(sentence, word2id, model.en_de_seq_len)
                beam_path, beam_symbol = model.step(sess, batch.encoderSeqs, batch.decoderSeqs, batch.targetSeqs,
                                                    batch.weights, goToken)
                paths = [[] for _ in range(beam_size)]
                curr = [i for i in range(beam_size)]
                num_steps = len(beam_path)
                for i in range(num_steps-1, -1, -1):
                    for kk in range(beam_size):
                        paths[kk].append(beam_symbol[i][curr[kk]])
                        curr[kk] = beam_path[i][curr[kk]]
                recos = set()
                print("Replies --------------------------------------->")
                # for kk in range(beam_size):
                    foutputs = [int(logit) for logit in paths[kk][::-1]]
                    if eosToken in foutputs:
                        foutputs = foutputs[:foutputs.index(eosToken)]
                    rec = " ".join([tf.compat.as_str(id2word[output]) for output in foutputs if output in id2word])
                    if rec not in recos:
                        recos.add(rec)
                        print(rec)
                print("> ", "")
                sys.stdout.flush()
                sentence = sys.stdin.readline()


def main(_):
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()

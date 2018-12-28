#encoding:utf-8
from __future__ import print_function
from text_model import *
from loader import *
from sklearn import metrics
import sys
import os
import time
from datetime import timedelta


def evaluate(sess, x_, y_):
    data_len = len(x_)
    batch_eval = batch_iter(x_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x_batch, y_batch in batch_eval:
        batch_len = len(x_batch)
        feed_dict = feed_data(x_batch, y_batch, 1.0)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len

def feed_data(x_batch, y_batch, keep_prob):
    feed_dict = {
        model.input_x: x_batch,
        model.input_y: y_batch,
        model.keep_prob:keep_prob
    }
    return feed_dict

def test():
    print("Loading test data...")
    t1=time.time()
    x_test,y_test=process_file(config.test_filename,word_to_id,cat_to_id,config.seq_length)

    session=tf.Session()
    session.run(tf.global_variables_initializer())
    saver=tf.train.Saver()
    saver.restore(sess=session,save_path=save_path)

    print('Testing...')
    test_loss,test_accuracy = evaluate(session,x_test,y_test)
    msg = 'Test Loss: {0:>6.2}, Test Acc: {1:>7.2%}'
    print(msg.format(test_loss, test_accuracy))

    batch_size=config.batch_size
    data_len=len(x_test)
    num_batch=int((data_len-1)/batch_size)+1
    y_test_cls=np.argmax(y_test,1)
    y_pred_cls=np.zeros(shape=len(x_test),dtype=np.int32)

    for i in range(num_batch):
        start_id=i*batch_size
        end_id=min((i+1)*batch_size,data_len)
        feed_dict={
            model.input_x:x_test[start_id:end_id],
            model.keep_prob:1.0,
        }
        y_pred_cls[start_id:end_id]=session.run(model.y_pred_cls,feed_dict=feed_dict)

    #evaluate
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test_cls, y_pred_cls, target_names=categories))

    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test_cls, y_pred_cls)
    print(cm)

    print("Time usage:%.3f seconds...\n"%(time.time() - t1))

if __name__ == '__main__':
    print('Configuring CNN model...')
    config = TextConfig()
    filenames = [config.train_filename, config.test_filename, config.val_filename]
    if not os.path.exists(config.vocab_filename):
        build_vocab(filenames, config.vocab_filename, config.vocab_size)
    #read vocab and categories
    categories,cat_to_id = read_category()
    words,word_to_id = read_vocab(config.vocab_filename)
    config.vocab_size = len(words)

    # trans vector file to numpy file
    if not os.path.exists(config.vector_word_npz):
        export_word2vec_vectors(word_to_id, config.vector_word_filename, config.vector_word_npz)
    config.pre_trianing = get_training_word2vec_vectors(config.vector_word_npz)
    model = TextCNN(config)

    save_dir = './checkpoints/textcnn'
    save_path = os.path.join(save_dir, 'best_validation')
    test()
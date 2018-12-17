#encoding:utf-8
import  tensorflow as tf

class TextConfig():

    embedding_size=100     #dimension of word embedding
    vocab_size=6000        #number of vocabulary
    pre_trianing = None   #use vector_char trained by word2vec

    seq_length=600         #max length of sentence
    num_classes=10          #number of labels

    num_filters=256        #number of convolution kernel
    kernel_size=5          #size of convolution kernel
    hidden_dim=128         #number of fully_connected layer units

    keep_prob=0.5          #droppout
    lr= 1e-3               #learning rate
    lr_decay= 0.9          #learning rate decay
    clip= 5.0              #gradient clipping threshold

    num_epochs=10          #epochs
    batch_size= 64         #batch_size
    print_per_batch =100   #print result

    train_filename='./data/cnews.train.txt'  #train data
    test_filename='./data/cnews.test.txt'    #test data
    val_filename='./data/cnews.val.txt'      #validation data
    vocab_filename='./data/vocab.txt'        #vocabulary
    vector_word_filename='./data/vector_word.txt'  #vector_word trained by word2vec
    vector_word_npz='./data/vector_word.npz'   # save vector_word to numpy file

class TextCNN(object):

    def __init__(self,config):

        self.config=config

        self.input_x=tf.placeholder(tf.int32,shape=[None,self.config.seq_length],name='input_x')
        self.input_y=tf.placeholder(tf.float32,shape=[None,self.config.num_classes],name='input_y')
        self.keep_prob=tf.placeholder(tf.float32,name='dropout')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        self.cnn()
    def cnn(self):
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable("embeddings", shape=[self.config.vocab_size, self.config.embedding_size],
                                             initializer=tf.constant_initializer(self.config.pre_trianing))
            embedding_inputs=tf.nn.embedding_lookup(self.embedding,self.input_x)

        with tf.name_scope('cnn'):
            conv= tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            outputs= tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope('fc'):
            fc=tf.layers.dense(outputs,self.config.hidden_dim,name='fc1')
            fc = tf.nn.dropout(fc, self.keep_prob)
            fc=tf.nn.relu(fc)

        with tf.name_scope('logits'):
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='logits')
            self.prob = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)

        with tf.name_scope('optimizer'):
            optimizer = tf.train.AdamOptimizer(self.config.lr)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))
            gradients, _ = tf.clip_by_global_norm(gradients, self.config.clip)
            self.optim = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)


        with tf.name_scope('accuracy'):
            correct_pred=tf.equal(tf.argmax(self.input_y,1),self.y_pred_cls)
            self.acc=tf.reduce_mean(tf.cast(correct_pred,tf.float32))


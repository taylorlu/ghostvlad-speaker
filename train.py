import os
import tensorflow as tf
import time
import model
import audio_producer

def run_epoch(epoch, ghostvlad_model, producer, sess, save_path, saver):

    ops = [ghostvlad_model.cost, ghostvlad_model.learning_rate, ghostvlad_model.global_step, ghostvlad_model.train_op]

    for inputs, labels in producer.iterator():

        feed_dict = ghostvlad_model.feed_dict(inputs, labels)
        cost, lr, step, _ = sess.run(ops, feed_dict)

        if(step%1000==0):
            saver.save(sess, save_path)

        print('Epoch {}, iter {}: Cost= {:.2f}, lr= {:.2e}'.format(epoch, step, cost, lr))

    saver.save(sess, save_path)


def main(argv):
    restore_path = argv.get('restore_path', None)
    save_path = argv['save_path']
    producer = audio_producer.AudioProducer(argv['json_path'], argv['batch_size'],
                                            sample_rate=argv['sample_rate'],
                                            min_duration=argv['min_duration'],
                                            max_duration=argv['max_duration'])

    graph = tf.Graph()
    with graph.as_default():

        ghostvlad_model = model.GhostVLADModel(argv)
        ghostvlad_model.init_inference(is_training=True)
        ghostvlad_model.init_cost()

        os.environ['OMP_NUM_THREADS'] = '32'
        os.environ['KMP_BLOCKTIME'] = '0'
        os.environ["KMP_SETTINGS"] = '0'
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        sess_conf = tf.ConfigProto()
        sess_conf.inter_op_parallelism_threads = 1
        sess_conf.intra_op_parallelism_threads = 32

        with tf.Session(config=sess_conf) as sess:
            restore_vars = []
            train_vars = []
            for var in tf.global_variables():
                if(var.name.startswith('arcface/')):
                    train_vars.append(var)
                else:
                    if(not 'Adam' in var.name):
                        restore_vars.append(var)

            ghostvlad_model.init_train(train_vars)
            sess.run(tf.global_variables_initializer())

            if restore_path:
                saver = tf.train.Saver(restore_vars)
                saver.restore(sess, restore_path)
            saver = tf.train.Saver()

            print("Begin training...")
            for e in range(argv['epochs']):
                run_epoch(e, ghostvlad_model, producer, sess, save_path, saver)
                print("========"*5)
                print("Finished epoch", e)


if __name__=="__main__":

    args_params = {
        "json_path": r"vox.json",
        "sample_rate": 16000,
        "min_duration": 1000,
        "max_duration": 3000,
        "save_path": r"saver/model.ckpt",
        "restore_path": r"ckpt/model.ckpt",

        "batch_size": 256,
        "epochs": 1000,
        "learning_rate": 0.001,
        "max_grad_norm": 50,
        "decay_steps": 5000,
        "decay_rate": 0.95,

        "vlad_clusters": 8,
        "ghost_clusters": 2,
        "embedding_dim": 512,
        "num_class": 5994
    }
    main(args_params)

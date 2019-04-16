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

        if step%1000 == 0:
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
        ghostvlad_model.init_train()

        sess_conf = tf.ConfigProto(allow_soft_placement=True)
        with tf.Session(config=sess_conf) as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            if restore_path:
                saver.restore(sess, os.path.join(restore_path, "data.ckpt"))

            print("Begin training...")
            for e in range(argv['epochs']):
                run_epoch(e, ghostvlad_model, producer, sess, save_path, saver)
                print("========"*5)
                print("Finished epoch", e)


if __name__=="__main__":

    args_params = {
        "json_path": r"vox.json",
        "sample_rate": 16000,
        "min_duration": 600,
        "max_duration": 2500,
        "save_path": r"saver",

        "batch_size": 64,
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

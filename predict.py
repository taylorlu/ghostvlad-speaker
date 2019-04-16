import os
import tensorflow as tf
import model
import audio_producer
import numpy as np

def main(argv):
    restore_path = argv.get('restore_path', None)
    wav_path = argv.get('wav_path', None)
    if(not restore_path or not wav_path):
        print("Please check your parameters.")
        exit(1)

    with tf.Graph().as_default():

        ghostvlad_model = model.GhostVLADModel(argv)
        ghostvlad_model.init_inference(is_training=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, os.path.join(restore_path, "model.ckpt"))
            specs = audio_producer.load_data(argv.get("wav_path"),
                                            sr=argv.get("sample_rate", 16000),
                                            is_training=False)
            specs = np.expand_dims(np.expand_dims(specs, 0), -1)

            final_tensor = sess.graph.get_tensor_by_name('l2_normalize:0')
            embedding = sess.run(final_tensor, feed_dict={"input:0": specs})
            print(embedding)


if __name__=="__main__":

    args_params = {
        "sample_rate": 16000,
        "restore_path": r"D:\PythonSpace\GhostVLAD-TF\ckpt",
        "wav_path": r'D:\PythonSpace\Speaker-Diarization\ghostvlad\4persons\a_1.wav'
    }
    main(args_params)

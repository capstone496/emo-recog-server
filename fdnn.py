
import tensorflow as tf
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.inception_v3 import preprocess_input
from keras.models import load_model

from vggish_input import wavfile_to_examples
import vggish_postprocess
import vggish_slim
import vggish_params

import waveform2img
import librosa
from PIL import Image

from flask import jsonify

import time


labels =  ['happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']

def align(array, target_frames=5):
    """
    Align all input to the same length (default is 128 frames).
    """
    n_frames, _ = array.shape

    if n_frames < target_frames:
        maxLen = 5
        temp = np.pad(array,[(0,maxLen-n_frames),(0,0)], mode='constant', constant_values = 0 )
        return temp
    return array[:target_frames, :]


def get_audio_input(wave_file_address):
    wave_file = wavfile_to_examples(wave_file_address)
    with tf.Graph().as_default(), tf.Session() as sess:
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, r'vggish_model.ckpt')
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)

        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: wave_file})
    pproc = vggish_postprocess.Postprocessor(r'vggish_pca_params.npz')
    sample4 =pproc.postprocess(embedding_batch)
    #print(np.shape(sample4))
    sample5 = align(sample4)
    sample5 = np.reshape(sample5,(1,5,128))

    return sample5


def get_image_input(wave_file_address):

    image_size = 150
    samples, sample_rate = librosa.load(wave_file_address)
    samples = waveform2img.butter_bandpass_filter(samples, sample_rate)
    spectrogram = waveform2img.get_melspectrogram(samples, sample_rate)
    img = Image.fromarray(waveform2img.duplicate_and_stack(spectrogram))
    img.save('test.jpg')
    input_image = load_img(r'test.jpg',target_size=(image_size,image_size),color_mode='grayscale')
    input_image = img_to_array(input_image)
    input_image = preprocess_input(input_image)
    input_image = np.reshape(input_image,(1,image_size,image_size,1))
    return input_image


def predict(wave_file_address = r'03-01-01-01-01-01-01.wav'):
    
    audio_input = get_audio_input(wave_file_address)
    image_input = get_image_input(wave_file_address)
    print(np.shape(audio_input))
    print(np.shape(image_input))

    audio_input = np.array(audio_input)
    image_input = np.array(image_input)

    model = load_model('fdnn.h5')
    #print(model.summary())
    result = model.predict([image_input,audio_input])
    y_classes = result.argmax(axis=-1)
    predicted_label = labels[int(y_classes)]
    print(predicted_label)

    return jsonify({"predictedEmotion": predicted_label})

if __name__ == "__main__":
    #predict()


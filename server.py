from pathlib import Path
import time

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from fdnn import predict, get_final_decision_model, get_vggish_params

################################################################
tik = time.time()

graph, sess, features_tensor, embedding_tensor, pproc = get_vggish_params()

tok = time.time()
tdelta = tok - tik
print("Time for loading models - {}".format(tdelta))

################################################################
tik = time.time()

model = get_final_decision_model()

tok = time.time()
tdelta = tok - tik
print("Time for loading final dec model - {}".format(tdelta))
################################################################


app = Flask(__name__)


@app.route("/")
def index():
    """
    this is a root dir of my server
    :return: str
    """
    return "Root"


@app.route('/api/predict', methods=['POST'])
def get_emotion_prediction():
    f = request.files['file']

    audio_filename = secure_filename(f.filename)
    audio_dir = Path('received_audio')

    audio_dir.mkdir(exist_ok=True)
    audio_filename = str(audio_dir / Path(audio_filename))

    f.save(audio_filename)
    print(audio_filename)

    predicted_emotion = predict(graph, sess, features_tensor, embedding_tensor, pproc, model, audio_filename)

    return jsonify({"predictedEmotion": predicted_emotion})


# running web app in local machine
if __name__ == '__main__':
    # predicted_emotion = predict(sess, features_tensor, embedding_tensor, pproc, model)
    app.run(host='0.0.0.0', port=7000)
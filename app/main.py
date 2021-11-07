from flask import Flask, render_template, request, redirect, flash
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from werkzeug.utils import secure_filename
import pickle
import os
import tensorflow_addons as tfa
import cv2

ALLOWED_EXTENSIONS = {"avi"}

# create an instance of Flask
app = Flask(__name__)
app.secret_key = "super secret key"
app.config["SESSION_TYPE"] = "filesystem"


model = None
word_to_id = None
id_to_word = None
table = None
num_oov_buckets = None
vocab_size = None
cnn_model = None


# Load pre-trained CNN model
def load_cnn_model():
    global cnn_model

    model_path = os.getcwd() + "/saved_models/saved_models/cnn_model/0001/"
    cnn_model = load_model(model_path)


# Extracts a number of frames per second. In current setting extracts 1 fps and returns these frames.
def extract_frames_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    videoFPS = int(cap.get(cv2.CAP_PROP_FPS))

    buf = np.empty((frameCount, frameHeight, frameWidth, 3), np.dtype("uint8"))

    fc = 0
    ret = True

    while fc < frameCount:
        ret, buf[fc] = cap.read()
        fc += 1

    cap.release()

    # We can keep more frames per second by decreasing the step (using a fraction of videoFPS)
    representative_frames = buf[::videoFPS, :, :, :]
    del buf
    del cap
    return representative_frames


# Extracting features from each frame in a video
def extract_features_from_video_frames(video_path):
    global cnn_model

    representative_frames = extract_frames_from_video(video_path)
    representative_frames = representative_frames / 255
    resized_frames = tf.image.resize_with_crop_or_pad(representative_frames, 600, 600)
    frames_features = cnn_model.predict(resized_frames)
    return frames_features


def load_lstm_model_and_dictionaries():
    global model
    global word_to_id
    global id_to_word
    global num_oov_buckets
    global vocab_size
    global table

    model_path = os.getcwd() + "/saved_models/saved_models/model.03-1.20/0001/"
    tfa.register_all(custom_kernels=False)
    model = load_model(model_path, compile=False)

    with open("id_to_word_dict.pickle", "rb") as handle:
        id_to_word = pickle.load(handle)
    with open("word_to_id_dict.pickle", "rb") as handle:
        word_to_id = pickle.load(handle)
    words = tf.constant(list(id_to_word.values()))
    word_ids = tf.constant(list(word_to_id.values()), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)
    num_oov_buckets = 1
    table = tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)
    id_to_word[len(id_to_word)] = "<unk>"
    vocab_size = len(words)

    # return model, word_to_id, id_to_word, table, num_oov_buckets, vocab_size


@app.route("/")
def home():
    return render_template("home.html")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Handle "Get captions" button being pressed
@app.route("/predict/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # check if the post request has the file part
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            file_path = os.getcwd() + "\\temp\\video\\temp.avi"
            file.save(file_path)
            cnn_features_array = extract_features_from_video_frames(file_path)
            os.remove(file_path)
            (
                predicted_caption_1,
                predicted_caption_2,
                predicted_caption_3,
            ) = generate_caption(cnn_features_array)
            return render_template(
                "predict.html",
                my_prediction_1=predicted_caption_1,
                my_prediction_2=predicted_caption_2,
                my_prediction_3=predicted_caption_3,
            )

    pass


# Take in extracted CNN features and generate captions by making predictions with the LSTM model.
def generate_caption(cnn_features_array):
    global model
    global word_to_id
    global id_to_word
    global table
    global num_oov_buckets
    global vocab_size

    # with probabilities
    beam_width = 3
    maximum_caption_length = 10

    sample_cnn_input_unlim, number_of_frames = get_padded_input_cnn_test(
        cnn_features_array, tf.constant(maximum_caption_length)
    )

    input_seq = table.lookup(
        tf.constant(
            append_bos_token(padding_part_of_caption_sequence(number_of_frames))
        )
    )
    initial_input_seq_length = len(input_seq)
    certainty_lists = [[] for _ in range(beam_width)]
    sentences = [input_seq for _ in range(beam_width)]
    sentence_probabilities = [1 for _ in range(beam_width)]
    END_TOKEN = 2

    for i in range(maximum_caption_length):
        if i == 0:
            prediction = model.predict(
                (
                    tf.expand_dims(sample_cnn_input_unlim[: len(input_seq)], axis=0),
                    tf.expand_dims(input_seq, axis=0),
                )
            )[0][-1]
            next_tokens = prediction.argsort()[-beam_width:][::-1]
            for j in range(beam_width):
                sentences[j] = tf.concat(
                    [
                        tf.cast(sentences[j], tf.int32),
                        tf.cast(tf.constant([next_tokens[j]]), tf.int32),
                    ],
                    axis=-1,
                )
                sentence_probabilities[j] = (
                    sentence_probabilities[j] * prediction[next_tokens[j]]
                )
                certainty_lists[j].append(prediction[next_tokens[j]])
        if i > 0:
            next_sentences = sentences.copy()
            next_certainty_lists = [[] for _ in range(beam_width)]
            next_token_probabilities = [[] for _ in range(beam_width)]
            next_sentence_probabilities = [[] for _ in range(beam_width)]
            for j in range(beam_width):
                prediction = model.predict(
                    (
                        tf.expand_dims(
                            sample_cnn_input_unlim[: len(sentences[j])], axis=0
                        ),
                        tf.expand_dims(sentences[j], axis=0),
                    )
                )[0][-1]
                next_sentence_probabilities[j] = prediction * sentence_probabilities[j]
                next_token_probabilities[j] = prediction
            indices_of_most_probable_sentences = np.dstack(
                np.unravel_index(
                    np.argsort(np.array(next_sentence_probabilities).ravel()),
                    (beam_width, vocab_size + num_oov_buckets),
                )
            )[0][-beam_width:][::-1]
            for k in range(beam_width):
                next_sentence_index = indices_of_most_probable_sentences[k][0]
                next_token_index = indices_of_most_probable_sentences[k][1]
                next_sentences[k] = tf.concat(
                    [
                        tf.cast(sentences[next_sentence_index], tf.int32),
                        tf.cast(tf.constant([next_token_index]), tf.int32),
                    ],
                    axis=-1,
                )
                sentence_probabilities[k] = next_sentence_probabilities[
                    next_sentence_index
                ][next_token_index]
                next_certainty_lists[k] = certainty_lists[next_sentence_index].copy()
                next_certainty_lists[k].append(
                    next_token_probabilities[next_sentence_index][next_token_index]
                )
            sentences = next_sentences
            certainty_lists = next_certainty_lists
            if (indices_of_most_probable_sentences[:, 1] == END_TOKEN).all():
                break

    # assuming that beam width = 3
    predicted_caption_1 = [
        id_to_word[id_]
        for id_ in sentences[0][initial_input_seq_length:].numpy().tolist()
    ]
    predicted_caption_1 = " ".join(
        predicted_caption_1[: get_showable(predicted_caption_1)]
    )
    predicted_caption_2 = [
        id_to_word[id_]
        for id_ in sentences[1][initial_input_seq_length:].numpy().tolist()
    ]
    predicted_caption_2 = " ".join(
        predicted_caption_2[: get_showable(predicted_caption_2)]
    )
    predicted_caption_3 = [
        id_to_word[id_]
        for id_ in sentences[2][initial_input_seq_length:].numpy().tolist()
    ]
    predicted_caption_3 = " ".join(
        predicted_caption_3[: get_showable(predicted_caption_3)]
    )

    return predicted_caption_1, predicted_caption_2, predicted_caption_3


def get_showable(my_predicted_caption):
    showable = 0
    for i in range(len(my_predicted_caption)):
        if my_predicted_caption[i] != "<eos>":
            showable += 1
        else:
            break
    return showable


def padding_part_of_caption_sequence(video_length):
    return video_length * ["<pad>"]


def append_bos_token(sequence_list):
    return sequence_list + ["<bos>"]


# Apply appropriate padding to the data so that it can be fed into the model
def get_padded_input_cnn_test(cnn_features_array, caption_length_tensor):
    number_of_frames = cnn_features_array.shape[0]
    input_cnn_padding_length = caption_length_tensor + 1
    input_cnn_padding_array = np.full([input_cnn_padding_length, 2560], 0)
    input_cnn_padded_array = np.concatenate(
        (cnn_features_array, input_cnn_padding_array)
    )
    return tf.constant(input_cnn_padded_array), number_of_frames


if __name__ == "__main__":
    load_lstm_model_and_dictionaries()
    load_cnn_model()
    app.run()

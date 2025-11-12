# type: ignore

from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import json, re, os

app = Flask(__name__, template_folder="templates")

# -------- Parámetros (ajusta si cambiaste en entrenamiento) -------
MODEL_PATH = os.path.join("model", "spam_model.keras")
TOKENIZER_PATH = os.path.join("model", "tokenizer_config.json")
MAX_LENGTH = 120

# -------- Utilidades de preprocesado (idénticas a las del entrenamiento) -------
def limpiar_texto(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\S+", "", texto)
    texto = re.sub(r"[^a-zA-Z0-9\s']", "", texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

# -------- Cargar modelo y tokenizer una sola vez al iniciar el servidor -------
print("🔁 Cargando modelo y tokenizer... (esto puede tardar unos segundos)")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Modelo cargado:", MODEL_PATH)

with open(TOKENIZER_PATH, "r", encoding="utf8") as f:
    tk_conf = json.load(f)

tokenizer = Tokenizer(num_words=tk_conf.get('num_words', None), oov_token=tk_conf.get('oov_token', None))
tokenizer.word_index = tk_conf.get('word_index', {})
print("✅ Tokenizer cargado.")

# -------- Función de predicción -------
def predecir_mensaje(texto, umbral=0.5):
    texto = limpiar_texto(texto)
    seq = tokenizer.texts_to_sequences([texto])
    pad = pad_sequences(seq, maxlen=MAX_LENGTH, padding='post', truncating='post')
    pred = model.predict(pad, verbose=0)[0][0]
    label = "SPAM" if pred > umbral else "NO SPAM"
    return {"label": label, "score": float(pred)}

# -------- Rutas web -------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Aceptamos form-data o JSON
    if request.is_json:
        data = request.get_json()
        mensaje = data.get("mensaje", "")
    else:
        mensaje = request.form.get("mensaje", "") or request.values.get("mensaje", "")

    if not mensaje:
        return jsonify({"error": "No se recibió el mensaje"}), 400

    res = predecir_mensaje(mensaje)
    return jsonify(res)

# -------- Run -------
if __name__ == "__main__":
    # DEBUG solo en desarrollo. Para producción usar gunicorn o waitress.
    app.run(host="0.0.0.0", port=5000, debug=True)

from flask import Flask, request, jsonify
import spacy

app = Flask(__name__)

MODEL_PATHS = {
    "METHOD": "en_deberta_v3_base_ner_method",
    "ACTIVITY": "en_deberta_v3_base_ner_activity",
    "GOAL": "en_deberta_v3_base_ner_goal",
}

MODELS = {}


def load_all_models():
    print("Loading models...")
    for name, path in MODEL_PATHS.items():
        MODELS[name] = spacy.load(path)
    print("Done.")


def process_batch(model, texts, label):
    results = []

    docs = list(model.pipe(texts, batch_size=32))

    for doc in docs:
        spans = [
            {
                "start": ent.start_char,
                "end": ent.end_char,
                "label": label,
                "start_token": ent.start,
                "end_token": ent.end,
            }
            for ent in doc.ents
        ]

        results.append({
            "tokens": [t.text for t in doc],
            "spans": spans,
        })

    return results


@app.route("/ner_batch", methods=["POST"])
def ner_batch():
    data = request.json

    texts = data["texts"]
    model_name = data["model_name"]
    entity = data["entity"]

    if model_name not in MODELS:
        return jsonify({"error": "unknown model"}), 400

    model = MODELS[model_name]

    output = process_batch(model, texts, entity)

    return jsonify(output)


if __name__ == "__main__":
    load_all_models()
    app.run(host="0.0.0.0", port=5005, debug=False)
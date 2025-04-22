from flask import Flask, request, jsonify
from pymongo import MongoClient
from dotenv import load_dotenv
import openai
import whisper
import os

# Load environment variables
load_dotenv()

# Initialize Flask
app = Flask(__name__)

# Set up OpenAI client (new SDK format)
openai.api_key = os.getenv("API_KEY")
# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client_mongo = MongoClient(mongo_uri)
db = client_mongo["passme"]
flight_collection = db["flights"]

# Whisper model
model = whisper.load_model("small")

@app.route("/chat-ai", methods=["POST"])
def handle_chat():
    flight_id = request.form.get("flight_id")
    if 'audio' not in request.files or not flight_id:
        return jsonify({"error": "Missing audio file or flight ID"}), 400

    audio_file = request.files['audio']
    audio_path = f"uploads/{audio_file.filename}"
    os.makedirs("uploads", exist_ok=True)
    audio_file.save(audio_path)

    try:
        # Step 1: Transcribe voice to English
        result = model.transcribe(audio_path)
        officer_question_en = result["text"]

        # Step 2: Get flight info from MongoDB
        flight = flight_collection.find_one({"_id": flight_id})
        if not flight:
            return jsonify({"error": "Flight not found"}), 404
        if not flight.get("language"):
            return jsonify({"error": "Language missing in flight info"}), 400

        language = flight["language"]

        # Step 3: Prepare AI prompt
        prompt = f"""
You are a multilingual travel assistant helping a traveler during immigration at an airport. Based on the flight information provided below, answer the officer's question as accurately and appropriately as possible using only relevant information.

If the question cannot be answered from the given flight info, politely say that and ask the user to provide an answer.

### Flight Info:
{flight}
The title is the purpose or objective of the flight.

### Officer's Question (English):
{officer_question_en}

### Your task:
- Analyze the officer's question and provide a clear and concise response in English.
- Then, translate the officer's question and the answer into {language}.
- Provide the phonetic pronunciation of the English answer written in {language} script.

Your response should be in the following format:

1. Officer Question (English): <officer_question_en>
2. Answer (English): <answer in English>
3. Officer Question ({language}): <translated officer question in {language}>
4. Answer ({language}): <translated answer in {language}>
5. Pronunciation in {language} letters: <phonetic pronunciation of English answer written using {language} characters>

Example Format (for Turkish):
Question (English): What is the purpose of your visit?
Answer (English): I am here for a business conference.
Question (Turkish): Ziyaretinizin amacı nedir?
Answer (Turkish): İş konferansı için buradayım.
Pronunciation in Turkish letters: Ay em hir for a biznis konferans

Example Format:
Question (English): What is the purpose of your visit?
Answer (English): I am here for a business conference.
Question (Amharic): የጉብኝትዎ አላማ ምንድን ነው?
Answer (Amharic): ለንግድ ኮንፈረንስ ነው የመጣሁት።
Pronunciation in Amharic letters: አይ አም ሄር ፎር አ ቢዝነስ ኮንፈረንስ
"""

        # Step 4: Send to OpenAI 
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[{
                "role": "user",
                "content": prompt
            }]
        )

        ai_reply = response.choices[0].message.content
        return jsonify({
            "ai_reply": ai_reply
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    app.run(port=5000)

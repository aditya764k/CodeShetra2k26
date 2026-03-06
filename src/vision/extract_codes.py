# src/vision/extract_codes.py
import os
import json
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def extract_from_note(image_path: str) -> dict:
    """
    Reads a doctor note image via inline bytes and returns a strict JSON dict.
    """
    # 1. Read the image as raw bytes (No need to upload/delete from Google's servers!)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    prompt = (
        "You are a medical billing assistant. "
        "Read this doctor note image and extract diagnosis and procedure info. "
        "Only use these ICD-10 codes if they appear or clearly match the text: "
        "['J01.90','M54.50','E11.9','I10','J44.9']. "
        "Only use these CPT codes: ['99213','99214','99215','93000','85025']. "
        "If none match, return empty arrays. "
        "Return ONLY valid JSON in this exact schema: {'icd10': [], 'cpt': [], 'notes': 'short justification'}"
    )

    try:
        # 2. Use the new SDK syntax for inline data and JSON configuration
        response = client.models.generate_content(
            model="gemini-2.5-flash",  # Fixes the 404 NOT FOUND error
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg"), 
                prompt
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json", # Forces native clean JSON output
            )
        )

        # 3. Parse directly (no need to strip markdown backticks anymore!)
        data = json.loads(response.text)

    except Exception as e:
        print(f"Extraction failed: {e}")
        # Fallback safe structure if the API completely fails
        data = {"icd10": [], "cpt": [], "notes": "Error during extraction."}

    # 4. Type safety checks to prevent your backend from crashing
    data.setdefault("icd10", [])
    data.setdefault("cpt", [])
    data.setdefault("notes", "")

    if not isinstance(data["icd10"], list): data["icd10"] = [str(data["icd10"])]
    if not isinstance(data["cpt"], list): data["cpt"] = [str(data["cpt"])]

    return data
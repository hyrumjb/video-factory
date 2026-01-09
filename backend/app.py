from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel
from openai import OpenAI
import os
import base64
from dotenv import load_dotenv

# Try to import Google TTS, but make it optional
try:
    from google.cloud import texttospeech
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠ Warning: google-cloud-texttospeech not installed. TTS functionality will be disabled.")

# Load environment variables
load_dotenv()

app = FastAPI()

# CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Next.js default port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Google TTS client
tts_client = None
google_creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# Try to find credentials file if not set in env
if not google_creds_path:
    # Check common locations
    possible_paths = [
        "./google-credentials.json",
        "./credentials.json",
        "../google-credentials.json",
    ]
    for path in possible_paths:
        if os.path.exists(path):
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.abspath(path)
            google_creds_path = os.path.abspath(path)
            break

if TTS_AVAILABLE:
    try:
        if google_creds_path and os.path.exists(google_creds_path):
            tts_client = texttospeech.TextToSpeechClient()
            print(f"✓ Google TTS client initialized with credentials: {google_creds_path}")
        else:
            print("⚠ Warning: Google TTS credentials not found. TTS will not work.")
            print("   To enable TTS, download credentials from Google Cloud Console and:")
            print("   1. Save the JSON file as 'google-credentials.json' in the backend folder")
            print("   2. Or set GOOGLE_APPLICATION_CREDENTIALS environment variable")
            tts_client = None
    except Exception as e:
        print(f"⚠ Warning: Could not initialize Google TTS client: {e}")
        print("   TTS functionality will be disabled until credentials are configured.")
        tts_client = None
else:
    tts_client = None
    print("⚠ Google TTS package not installed. Install with: pip install google-cloud-texttospeech")

# Request models
class ScriptRequest(BaseModel):
    topic: str

class TTSRequest(BaseModel):
    text: str

# Response models
class ScriptResponse(BaseModel):
    script: str

class TTSResponse(BaseModel):
    audio_url: str

@app.get("/")
def read_root():
    return {"message": "Video Factory API"}

@app.post("/api/generate-script", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    try:
        if not request.topic or not request.topic.strip():
            raise HTTPException(status_code=400, detail="Topic is required")

        # System prompt for 30-second super viral video script
        system_prompt = """You are a professional script writer specializing in creating engaging, 
captivating 30-second super viral video scripts optimized for short-form content. 
Create scripts that are punchy, visual, edgy, and designed to hook viewers 
immediately. The script should be exactly 30 seconds when spoken at a 
normal pace. Make it incredibly engaging and optimized for super viral short-form video content. 
CRITICAL REQUIREMENTS:
- The script must be EDGY and push boundaries while remaining engaging
- Always include at least one interesting item of substance - a fact, insight, or revelation that makes the viewer think
- NEVER include emojis, hashtags, or any social media formatting
- This is the ACTUAL SPOKEN CONTENT that would be said in a 30-second viral video - write it as dialogue/narration
- Write ONLY the raw script - just the words that would be spoken
- DO NOT include any descriptions of what is being seen on screen
- DO NOT include any stage directions, visual cues, or action descriptions
- DO NOT include any speaker names, labels, or indicators of who is speaking
- Output ONLY the pure dialogue/narration text with nothing else"""

        # Generate script using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a script for a 30-second super viral video about: {request.topic}. The script must be edgy, contain an interesting item of substance, never include emojis or hashtags. Output ONLY the raw spoken script - no visual descriptions, no speaker indicators, no stage directions. Just the pure dialogue/narration that would be spoken."}
            ],
            temperature=0.8,
            max_tokens=500
        )

        script = response.choices[0].message.content

        if not script:
            raise HTTPException(status_code=500, detail="Failed to generate script")

        return ScriptResponse(script=script)

    except Exception as e:
        # Handle OpenAI API errors
        if hasattr(e, 'status_code'):
            raise HTTPException(status_code=e.status_code, detail=str(e))
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.post("/api/generate-tts")
async def generate_tts(request: TTSRequest):
    try:
        if not request.text or not request.text.strip():
            raise HTTPException(status_code=400, detail="Text is required")
        
        if not TTS_AVAILABLE or not tts_client:
            raise HTTPException(status_code=503, detail="Google TTS is not available. Please install google-cloud-texttospeech package.")

        # Configure the voice and audio settings
        synthesis_input = texttospeech.SynthesisInput(text=request.text.strip())
        
        # Use a modern, engaging voice (Neural2 voices are high quality)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            name="en-US-Neural2-F",  # Female voice, can change to en-US-Neural2-M for male
            ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3,
            speaking_rate=1.0,  # Normal speed
            pitch=0.0,  # Normal pitch
        )

        # Generate speech
        response = tts_client.synthesize_speech(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config
        )

        # Return audio as base64 encoded string
        audio_base64 = base64.b64encode(response.audio_content).decode('utf-8')
        
        return TTSResponse(audio_url=f"data:audio/mp3;base64,{audio_base64}")

    except Exception as e:
        if hasattr(e, 'status_code'):
            raise HTTPException(status_code=e.status_code, detail=str(e))
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

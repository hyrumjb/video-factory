from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os
from dotenv import load_dotenv

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

# Request model
class ScriptRequest(BaseModel):
    topic: str

# Response model
class ScriptResponse(BaseModel):
    script: str

@app.get("/")
def read_root():
    return {"message": "Video Factory API"}

@app.post("/api/generate-script", response_model=ScriptResponse)
async def generate_script(request: ScriptRequest):
    try:
        if not request.topic or not request.topic.strip():
            raise HTTPException(status_code=400, detail="Topic is required")

        # System prompt for 30-second engaging script
        system_prompt = """You are a professional script writer specializing in creating engaging, 
captivating 30-second video scripts optimized for short-form content. 
Create scripts that are punchy, visual, and designed to hook viewers 
immediately. The script should be exactly 30 seconds when spoken at a 
normal pace. Make it incredibly engaging and optimized for short-form video content."""

        # Generate script using OpenAI
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Generate a 30-second engaging video script about: {request.topic}"}
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

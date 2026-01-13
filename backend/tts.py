import re
import html
from config import TTS_AVAILABLE, tts_client, texttospeech

def list_available_voices(language_code: str = "en-US"):
    """
    List all available voices for a given language code.
    Useful for finding voice names to use.
    """
    if not TTS_AVAILABLE or not tts_client:
        return []
    
    try:
        voices = tts_client.list_voices(language_code=language_code)
        voice_list = []
        for voice in voices.voices:
            voice_list.append({
                'name': voice.name,
                'ssml_gender': voice.ssml_gender.name,
                'natural_sample_rate_hertz': voice.natural_sample_rate_hertz
            })
        return voice_list
    except Exception as e:
        print(f"Error listing voices: {e}")
        return []

def generate_tts_with_timing(text: str, voice_name: str = "en-US-Neural2-H"):
    """
    Generate TTS audio with word-level timing information using SSML marks.
    Returns audio content and timepoints for subtitle generation.
    
    Following Google's recommended approach: insert <mark/> tags for EACH WORD
    to get accurate word-level timing for subtitles.
    
    Args:
        text: Text to synthesize
        voice_name: Voice name to use (default: "en-US-Neural2-H")
    """
    if not TTS_AVAILABLE or not tts_client:
        return None, None
    
    try:
        # Split text into words while preserving punctuation and spacing
        words = re.findall(r'\S+|\s+', text)  # Matches words (non-whitespace) or whitespace sequences
        
        # Build SSML with mark tags for EACH WORD (as per Google's recommendation)
        # Also add natural breaks between sentences
        ssml_parts = ['<speak>']
        mark_count = 0
        
        for i, word in enumerate(words):
            if word.strip():  # Only mark actual words, not pure whitespace
                # Escape HTML entities but preserve SSML structure
                escaped_word = html.escape(word)
                ssml_parts.append(escaped_word)
                # Add mark after each word for word-level timing
                ssml_parts.append(f'<mark name="{mark_count}"/>')
                mark_count += 1
                
                # Check if this word ends a sentence (ends with . ! ?)
                # Add a natural break after sentence-ending punctuation
                if re.search(r'[.!?]$', word):
                    # Add a light break (0.3 seconds) for natural sentence pauses
                    remaining_text = ''.join(words[i+1:])
                    if remaining_text.strip():  # Only add break if there's more content
                        ssml_parts.append('<break time="0.3s"/>')
            else:
                # Preserve whitespace as-is
                ssml_parts.append(word)
        
        ssml_parts.append('</speak>')
        ssml_text = ''.join(ssml_parts)
        
        # Configure synthesis with timing enabled
        synthesis_input = texttospeech.SynthesisInput(ssml=ssml_text)
        
        # Determine gender from voice name
        # Extract language code and voice ID to determine gender accurately
        language_code = "en-US"  # Default
        if "-" in voice_name:
            parts = voice_name.split("-")
            if len(parts) >= 2:
                language_code = f"{parts[0]}-{parts[1]}"
        
        # Convert to lowercase for API (Google requires lowercase language codes)
        language_code_lower = language_code.lower()
        
        # Determine gender based on language code and voice ending (use original case for comparison)
        if language_code.upper() == "EN-GB":
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") or voice_name.endswith("-F") or voice_name.endswith("-N") else texttospeech.SsmlVoiceGender.MALE
        elif language_code.upper() == "EN-AU":
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") else texttospeech.SsmlVoiceGender.MALE
        else:  # American (en-US)
            ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
        
        voice = texttospeech.VoiceSelectionParams(
            language_code=language_code_lower,
            name=voice_name,
            ssml_gender=ssml_gender,
        )
        
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            speaking_rate=0.97,
            pitch=0.0,
        )
        
        # Generate speech with timepointing enabled using SynthesizeSpeechRequest object
        timepoint_type = texttospeech.SynthesizeSpeechRequest.TimepointType.SSML_MARK
        
        request = texttospeech.SynthesizeSpeechRequest(
            input=synthesis_input,
            voice=voice,
            audio_config=audio_config,
            enable_time_pointing=[timepoint_type]
        )
        
        response = tts_client.synthesize_speech(request=request)
        
        # Extract timepoints - these now contain word-level timing
        timepoints = response.timepoints if hasattr(response, 'timepoints') else None
        
        if timepoints:
            print(f"✓ Generated audio with {len(timepoints)} word-level timepoints")
        
        return response.audio_content, timepoints
        
    except (AttributeError, TypeError) as e:
        print(f"Timepointing not available: {e}")
        print("Make sure you're using google-cloud-texttospeech with v1beta1 support")
        raise
    except Exception as e:
        print(f"Error in generate_tts_with_timing: {e}")
        import traceback
        traceback.print_exc()
        # Fallback to simple TTS without timing
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            # Determine gender from voice name
            language_code = "en-US"
            if "-" in voice_name:
                parts = voice_name.split("-")
                if len(parts) >= 2:
                    language_code = f"{parts[0]}-{parts[1]}"
            
            # Convert to lowercase for API (Google requires lowercase language codes)
            language_code_lower = language_code.lower()
            
            if language_code.upper() == "EN-GB":
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") or voice_name.endswith("-F") or voice_name.endswith("-N") else texttospeech.SsmlVoiceGender.MALE
            elif language_code.upper() == "EN-AU":
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-A") or voice_name.endswith("-C") else texttospeech.SsmlVoiceGender.MALE
            else:
                ssml_gender = texttospeech.SsmlVoiceGender.FEMALE if voice_name.endswith("-F") or voice_name.endswith("-C") or voice_name.endswith("-E") or voice_name.endswith("-G") or voice_name.endswith("-H") else texttospeech.SsmlVoiceGender.MALE
            
            voice = texttospeech.VoiceSelectionParams(
                language_code=language_code_lower,
                name=voice_name,
                ssml_gender=ssml_gender,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.MP3,
                speaking_rate=1.0,
                pitch=0.0,
            )
            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config
            )
            return response.audio_content, None
        except Exception as fallback_error:
            print(f"Fallback TTS also failed: {fallback_error}")
            raise

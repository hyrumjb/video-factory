import re
from typing import List, Optional

def convert_elevenlabs_alignment_to_word_timings(text: str, alignment) -> list:
    """
    Convert ElevenLabs character-level alignment to word-level timing.

    ElevenLabs returns:
    - alignment.characters: list of characters
    - alignment.character_start_times_seconds: list of start times
    - alignment.character_end_times_seconds: list of end times

    Returns a list of dicts with 'word', 'start', 'end' for each word.
    """
    if not alignment:
        return []

    try:
        characters = alignment.characters if hasattr(alignment, 'characters') else []
        start_times = alignment.character_start_times_seconds if hasattr(alignment, 'character_start_times_seconds') else []
        end_times = alignment.character_end_times_seconds if hasattr(alignment, 'character_end_times_seconds') else []

        if not characters or not start_times or not end_times:
            print(f"⚠ ElevenLabs alignment missing data: chars={len(characters)}, starts={len(start_times)}, ends={len(end_times)}")
            return []

        print(f"✓ ElevenLabs alignment: {len(characters)} characters with timing")

        # Build word timings from character timings
        word_timings = []
        current_word = ""
        word_start = None
        word_end = None

        for i, char in enumerate(characters):
            if i >= len(start_times) or i >= len(end_times):
                break

            char_start = start_times[i]
            char_end = end_times[i]

            # Check if this is a word boundary (space or start of text)
            if char == ' ' or char == '\n':
                # End of word - save it if we have one
                if current_word.strip() and word_start is not None:
                    word_timings.append({
                        'word': current_word.strip(),
                        'start': word_start,
                        'end': word_end
                    })
                current_word = ""
                word_start = None
                word_end = None
            else:
                # Part of a word
                if word_start is None:
                    word_start = char_start
                word_end = char_end
                current_word += char

        # Don't forget the last word
        if current_word.strip() and word_start is not None:
            word_timings.append({
                'word': current_word.strip(),
                'start': word_start,
                'end': word_end
            })

        print(f"✓ Converted to {len(word_timings)} word timings")
        if word_timings:
            preview = [(w['word'], f"{w['start']:.2f}s-{w['end']:.2f}s") for w in word_timings[:3]]
            print(f"   First 3 words: {preview}")

        return word_timings

    except Exception as e:
        print(f"⚠ Error converting ElevenLabs alignment: {e}")
        import traceback
        traceback.print_exc()
        return []


def create_ass_from_elevenlabs_alignment(script: str, alignment, audio_duration: float) -> str:
    """
    Create ASS subtitle file from ElevenLabs alignment data.
    Uses the same state-based karaoke style as Google TTS version.
    """
    word_timings = convert_elevenlabs_alignment_to_word_timings(script, alignment)

    if not word_timings:
        print("⚠ No word timings from ElevenLabs, falling back to even timing")
        return create_ass_fallback(script, audio_duration)

    print(f"\n🎤 ========== ELEVENLABS KARAOKE SETUP ==========")
    print(f"📝 Script length: {len(script)} chars")
    print(f"⏱️  Audio duration: {audio_duration:.2f}s")
    print(f"🔢 Word timings: {len(word_timings)}")

    # ASS file header (same styling as Google TTS version)
    ass_lines = [
        "[Script Info]",
        "Title: Professional Video Factory Subtitles",
        "ScriptType: v4.00+",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Inter,16,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,10,1,2,10,10,55,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    def format_ass_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"

    def is_sentence_ending(word: str) -> bool:
        return bool(re.search(r'[.!?]+$', word))

    # Group words into 1-3 word phrases
    max_words_per_subtitle = 3
    i = 0
    global_end_time = 0.0

    while i < len(word_timings):
        # Build phrase (1-3 words, stop at sentence endings)
        phrase_timings = []
        start_idx = i

        while len(phrase_timings) < max_words_per_subtitle and i < len(word_timings):
            phrase_timings.append(word_timings[i])
            i += 1

            # Stop at sentence endings
            if is_sentence_ending(phrase_timings[-1]['word']):
                break

        if not phrase_timings:
            break

        # Generate state-based karaoke: one dialogue line per word
        # Extend each word's end time to meet the next word's start to prevent flashing
        for highlight_idx, timing in enumerate(phrase_timings):
            text_parts = []
            for word_idx, wt in enumerate(phrase_timings):
                word = wt['word']
                if word_idx == highlight_idx:
                    text_parts.append(f"{{\\1c&H0000FFFF&}}{word}")
                else:
                    text_parts.append(f"{{\\1c&H00FFFFFF&}}{word}")

            karaoke_text = "{\\an2}" + " ".join(text_parts)

            # Use actual word timing from ElevenLabs
            line_start = max(timing['start'], global_end_time)

            # Extend end time to next word's start (no gaps = no flashing)
            if highlight_idx < len(phrase_timings) - 1:
                # Not the last word in phrase - extend to next word's start
                next_timing = phrase_timings[highlight_idx + 1]
                line_end = next_timing['start']
            else:
                # Last word in phrase - check if there's a next phrase
                if i < len(word_timings):
                    # Extend to next phrase's first word
                    line_end = word_timings[i]['start']
                else:
                    # Last word overall - use its natural end time
                    line_end = timing['end']

            # Ensure minimum duration
            if line_end - line_start < 0.15:
                line_end = line_start + 0.15

            dialogue_line = f"Dialogue: 0,{format_ass_time(line_start)},{format_ass_time(line_end)},Default,,0,0,0,,{karaoke_text}"
            ass_lines.append(dialogue_line)

            global_end_time = line_end

    print(f"✅ ELEVENLABS KARAOKE ASS FILE GENERATED: {len(ass_lines)} lines")
    print(f"==========================================\n")

    return '\n'.join(ass_lines)


def create_srt_from_timepoints(script: str, timepoints: list, audio_duration: float, style: str = "3words") -> str:
    """
    Create SRT subtitle file from script text and Google TTS word-level timepoints.
    
    Args:
        script: The script text
        timepoints: List of timepoints from Google TTS
        audio_duration: Total audio duration in seconds
        style: Caption style - "1word", "3words", "5words", or "varying"
    """
    if not timepoints or len(timepoints) == 0:
        # Fallback: create simple subtitles with even timing
        return create_srt_fallback(script, audio_duration)
    
    # CRITICAL: Split script into words EXACTLY the same way as in generate_tts_with_timing
    words = re.findall(r'\S+|\s+', script)
    word_list = []
    for item in words:
        if item.strip():
            word_list.append(item.strip())
    
    # Debug: Verify word count matches timepoint count
    if len(timepoints) != len(word_list):
        print(f"⚠ WARNING: Word count mismatch! Words: {len(word_list)}, Timepoints: {len(timepoints)}")
    
    # Create timepoint map: mark_name -> time_seconds
    timepoint_map = {}
    for tp in timepoints:
        mark_name = tp.mark_name
        try:
            mark_num = int(mark_name)
            timepoint_map[mark_num] = tp.time_seconds
        except (ValueError, AttributeError):
            continue
    
    # Determine grouping strategy based on style
    if style == "1word":
        max_words_per_subtitle = 1
        min_words_per_subtitle = 1
    elif style == "5words":
        max_words_per_subtitle = 5
        min_words_per_subtitle = 5
    elif style == "varying":
        max_words_per_subtitle = 7
        min_words_per_subtitle = 3
        max_chars_per_line = 42
    else:  # Default: 3 words
        max_words_per_subtitle = 3
        min_words_per_subtitle = 3
    
    # Group words into subtitle lines
    srt_lines = []
    subtitle_index = 1
    
    i = 0
    while i < len(word_list):
        # CRITICAL: For the FIRST subtitle, it MUST start at word 0
        if subtitle_index == 1 and i != 0:
            print(f"⚠⚠⚠ CRITICAL ERROR: First subtitle starting at word index {i} instead of 0!")
            i = 0
        
        if i >= len(word_list):
            break
        
        if style == "varying":
            # Smart varying length - build subtitle based on character count
            words_for_subtitle: List[str] = []
            char_count = 0
            line_breaks = 0
            start_word_idx = i
            
            while i < len(word_list) and len(words_for_subtitle) < max_words_per_subtitle * 2:
                word = word_list[i]
                word_len = len(word)
                
                if char_count + word_len > max_chars_per_line and len(words_for_subtitle) >= min_words_per_subtitle:
                    if line_breaks == 0:
                        words_for_subtitle.append('\n')
                        char_count = 0
                        line_breaks = 1
                    else:
                        break
                
                words_for_subtitle.append(word)
                char_count += word_len
                i += 1
            
            if len([w for w in words_for_subtitle if w != '\n']) < min_words_per_subtitle:
                while i < len(word_list) and len([w for w in words_for_subtitle if w != '\n']) < min_words_per_subtitle:
                    words_for_subtitle.append(word_list[i])
                    i += 1
            
            last_word_idx = i - 1
            subtitle_text = ''.join(words_for_subtitle).strip()
        else:
            # Fixed word count per subtitle - but for 3words style, respect sentence endings
            words_for_subtitle: List[str] = []
            start_word_idx = i

            def is_sentence_ending(word: str) -> bool:
                return bool(re.search(r'[.!?]+$', word))
            
            while len(words_for_subtitle) < max_words_per_subtitle and i < len(word_list):
                word = word_list[i]
                words_for_subtitle.append(word)
                i += 1
                
                # If this word ends a sentence, stop here (even if we haven't reached max_words)
                if style == "3words" and is_sentence_ending(word):
                    break
            
            if not words_for_subtitle:
                break
            
            last_word_idx = i - 1
            subtitle_text = ' '.join(words_for_subtitle)
        
        # CRITICAL: Ensure word indices are valid BEFORE calculating timing
        if start_word_idx >= len(word_list) or last_word_idx >= len(word_list):
            print(f"⚠ ERROR: Invalid word indices for subtitle {subtitle_index}: start={start_word_idx}, last={last_word_idx}, word_list_length={len(word_list)}")
            continue
        
        # Validate and fix subtitle text FIRST, then calculate timing
        expected_words = word_list[start_word_idx:last_word_idx+1]
        if style != "varying":
            expected_text = ' '.join(expected_words)
            if subtitle_text != expected_text:
                subtitle_text = expected_text
                words_for_subtitle = expected_words
                if len(expected_words) > 0:
                    actual_last_idx = start_word_idx + len(expected_words) - 1
                    if actual_last_idx != last_word_idx:
                        last_word_idx = actual_last_idx
        
        # NOW calculate timing using the CORRECT word indices - EXACT TIMING (no adjustments)
        if start_word_idx == 0:
            start_time = 0.0
        else:
            start_time = timepoint_map.get(start_word_idx - 1, 0.0)
        
        end_time = timepoint_map.get(last_word_idx, audio_duration)
        
        # Ensure minimum subtitle duration and clamp to audio duration
        min_duration = 0.2
        if end_time - start_time < min_duration:
            end_time = start_time + min_duration
        
        end_time = min(end_time, audio_duration)
        
        # Format times as SRT (HH:MM:SS,mmm)
        def format_srt_time(seconds):
            hours = int(seconds // 3600)
            minutes = int((seconds % 3600) // 60)
            secs = int(seconds % 60)
            millis = int((seconds % 1) * 1000)
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
        
        if subtitle_text:
            srt_lines.append(str(subtitle_index))
            srt_lines.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
            srt_lines.append(subtitle_text)
            srt_lines.append("")
            subtitle_index += 1
    
    # Final validation - ensure no duplicate or overlapping subtitles
    subtitle_entries = []
    i = 0
    while i < len(srt_lines):
        if i + 3 < len(srt_lines) and srt_lines[i].strip().isdigit():
            try:
                sub_index = int(srt_lines[i])
                time_line = srt_lines[i + 1]
                text_line = srt_lines[i + 2]
                if ' --> ' in time_line:
                    start_str, end_str = time_line.split(' --> ')
                    def parse_time(t):
                        h, m, s = t.split(':')
                        sec, ms = s.split(',')
                        return int(h) * 3600 + int(m) * 60 + int(sec) + int(ms) / 1000.0
                    start_time_parsed = parse_time(start_str)
                    end_time_parsed = parse_time(end_str)
                    subtitle_entries.append({
                        'index': sub_index,
                        'start': start_time_parsed,
                        'end': end_time_parsed,
                        'text': text_line,
                        'lines': [srt_lines[i], srt_lines[i + 1], srt_lines[i + 2], srt_lines[i + 3] if i + 3 < len(srt_lines) else ""]
                    })
                i += 4
            except:
                i += 1
        else:
            i += 1
    
    # Check for duplicates and overlaps
    seen_texts = set()
    final_lines = []
    prev_end = -1
    for entry in subtitle_entries:
        text_key = entry['text'].strip()
        if text_key in seen_texts and entry['start'] < 1.0:
            continue
        seen_texts.add(text_key)
        
        if entry['start'] < prev_end:
            entry['start'] = prev_end + 0.01
            def format_srt_time(seconds):
                hours = int(seconds // 3600)
                minutes = int((seconds % 3600) // 60)
                secs = int(seconds % 60)
                millis = int((seconds % 1) * 1000)
                return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
            entry['lines'][1] = f"{format_srt_time(entry['start'])} --> {format_srt_time(entry['end'])}"
        
        prev_end = entry['end']
        final_lines.extend(entry['lines'])
    
    return '\n'.join(final_lines) if final_lines else '\n'.join(srt_lines)

def create_srt_fallback(script: str, audio_duration: float) -> str:
    """Fallback SRT generation when timepoints are not available."""
    sentences = re.split(r'([.!?]+\s+)', script)
    combined_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        if i + 1 < len(sentences):
            combined_sentences.append(sentences[i] + sentences[i + 1])
        else:
            combined_sentences.append(sentences[i])
    if len(sentences) % 2 == 1:
        combined_sentences.append(sentences[-1])
    
    combined_sentences = [s.strip() for s in combined_sentences if s.strip()]
    
    if not combined_sentences:
        return ""
    
    srt_lines = []
    subtitle_index = 1
    duration_per_sentence = audio_duration / len(combined_sentences)
    
    def format_srt_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    for i, sentence in enumerate(combined_sentences):
        start_time = i * duration_per_sentence
        end_time = (i + 1) * duration_per_sentence if i < len(combined_sentences) - 1 else audio_duration
        
        srt_lines.append(str(subtitle_index))
        srt_lines.append(f"{format_srt_time(start_time)} --> {format_srt_time(end_time)}")
        srt_lines.append(sentence)
        srt_lines.append("")
        subtitle_index += 1
    
    return '\n'.join(srt_lines)

def create_ass_from_timepoints(script: str, timepoints: list, audio_duration: float, style: str = "3words") -> str:
    """
    Create ASS (Advanced SubStation Alpha) subtitle file with state-based karaoke.

    Settings:
    - Bold sans-serif font (Inter Bold)
    - Font size: 16px (small for mobile videos)
    - Outline: 6px (thick black border)
    - Shadow: 1px
    - Alignment: bottom-center (2) with \\an2 override for stability
    - MarginV: 25px (near bottom of screen)
    - Maximum 3 words per subtitle line
    - State-based highlighting: generates separate dialogue lines for each word,
      only the currently-spoken word is yellow, all others are white
    - Clean instant transitions with no fading

    Args:
        script: The script text
        timepoints: List of timepoints from Google TTS
        audio_duration: Total audio duration in seconds
        style: Caption style
    """
    if not timepoints or len(timepoints) == 0:
        return create_ass_fallback(script, audio_duration)
    
    # DEBUG: Print karaoke setup info
    print(f"\n🎤 ========== KARAOKE SETUP DEBUG ==========")
    print(f"📝 Script length: {len(script)} chars")
    print(f"⏱️  Audio duration: {audio_duration:.2f}s")
    print(f"🔢 Timepoints received: {len(timepoints)}")
    
    # Split script into words EXACTLY the same way as in generate_tts_with_timing
    words = re.findall(r'\S+|\s+', script)
    word_list = []
    for item in words:
        if item.strip():
            word_list.append(item.strip())
    
    print(f"📝 Words extracted: {len(word_list)}")
    print(f"   First 10 words: {word_list[:10]}")
    
    # Create timepoint map
    timepoint_map = {}
    for tp in timepoints:
        try:
            mark_num = int(tp.mark_name)
            timepoint_map[mark_num] = tp.time_seconds
        except (ValueError, AttributeError):
            continue
    
    print(f"🗺️  Timepoint map created: {len(timepoint_map)} entries")
    if len(timepoint_map) > 0:
        sample_keys = list(timepoint_map.keys())[:5]
        print(f"   Sample timepoints: {[(k, timepoint_map[k]) for k in sample_keys]}")
    
    # Subtitle settings: max 3 words per subtitle for easy reading
    max_words_per_subtitle = 3
    min_words_per_subtitle = 1  # Allow natural phrase breaks
    
    # ASS file header with professional TikTok-style styling
    # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour,
    #         Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle,
    #         Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
    #
    # STATE-BASED KARAOKE (generates separate dialogue lines):
    # - Instead of transforms, we generate one dialogue line per word
    # - Each line shows ALL words with explicit color tags
    # - Only the currently-spoken word is yellow, others are white
    # - Lines are timed precisely to switch at word boundaries
    # - This gives clean instant transitions with no fading
    #
    # Settings:
    # - Fontname=Inter, Fontsize=16, Bold=1, Outline=10 (thick), Shadow=1
    # - Alignment=2 (bottom-center), MarginV=55, BorderStyle=1
    # - \an2 override in each dialogue line for stable vertical positioning
    # - global_end_time tracking prevents overlap between subtitle groups
    ass_lines = [
        "[Script Info]",
        "Title: Professional Video Factory Subtitles",
        "ScriptType: v4.00+",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Style: Default, Inter Bold, 16px (small for mobile videos)
        # PrimaryColour = White (default text color)
        # Bold=1, Outline=10px (thick black border), Shadow=1px, MarginV=55px
        "Style: Default,Inter,16,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,10,1,2,10,10,55,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    # DEBUG: Print style definition
    print("\n🎨 ASS STYLE DEFINITION:")
    print("   Font: Inter, Size: 16px, Bold: 1")
    print("   Using state-based karaoke (separate dialogue lines per word)")
    print("   Yellow = currently spoken word, White = all other words")
    print("   Outline: 10px (thick black border), Shadow: 1px")
    print("   Alignment: 2 (bottom-center) with \\an2 override")
    print("   MarginV: 55px from bottom")

    # Track cumulative end time to prevent overlap between subtitle groups
    global_end_time = 0.0
    
    # Group words into subtitle lines with karaoke effects
    subtitle_index = 1
    i = 0
    
    def format_ass_time(seconds):
        """Format time as ASS format: H:MM:SS.cc (centiseconds)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
    
    def is_sentence_ending(word: str) -> bool:
        """Check if word ends a sentence"""
        return bool(re.search(r'[.!?]+$', word))
    
    def is_punctuation(word: str) -> bool:
        """Check if word is just punctuation"""
        return bool(re.match(r'^[,.!?;:]+$', word))
    
    while i < len(word_list):
        if subtitle_index == 1 and i != 0:
            i = 0
        
        if i >= len(word_list):
            break
        
        # Natural phrase grouping: 1-3 words, respecting sentence endings and natural pauses
        words_for_subtitle: List[str] = []
        start_word_idx = i
        
        # Build phrase with natural grouping (1-3 words)
        # ALWAYS stop at sentence endings - this is critical
        while len(words_for_subtitle) < max_words_per_subtitle and i < len(word_list):
            word = word_list[i]
            words_for_subtitle.append(word)
            i += 1
            
            # CRITICAL: Always stop at sentence endings - never continue past a sentence end
            if is_sentence_ending(word):
                break
            
            # Stop early after 2 words if next word starts a new phrase (capital letter or punctuation)
            if len(words_for_subtitle) >= 2 and i < len(word_list):
                next_word = word_list[i]
                if next_word and next_word[0].isupper() and not is_punctuation(next_word):
                    # Check if current phrase is complete enough
                    if len(words_for_subtitle) >= min_words_per_subtitle:
                        break
        
        if not words_for_subtitle:
            break
        
        last_word_idx = i - 1
        
        # Calculate timing from timepoints
        if start_word_idx == 0:
            start_time = 0.0
        else:
            start_time = timepoint_map.get(start_word_idx - 1, 0.0)
        
        end_time = timepoint_map.get(last_word_idx, audio_duration)
        end_time = min(end_time, audio_duration)
        
        # Ensure minimum subtitle duration for readability
        min_duration = 0.5
        if end_time - start_time < min_duration:
            end_time = start_time + min_duration
        end_time = min(end_time, audio_duration)
        
        # STATE-BASED KARAOKE: Generate separate dialogue lines for each word highlight state
        #
        # Instead of using transforms (which can fade/conflict), we generate one dialogue
        # line per word, each showing ALL words but with explicit colors:
        # - The currently-spoken word is yellow
        # - All other words are white
        # - Lines are timed precisely to switch at word boundaries
        #
        # Example for "Hello beautiful world":
        # - Line 1 (0.0s-0.5s): "{\1c&H0000FFFF&}Hello {\1c&H00FFFFFF&}beautiful world"
        # - Line 2 (0.5s-1.2s): "Hello {\1c&H0000FFFF&}beautiful {\1c&H00FFFFFF&}world"
        # - Line 3 (1.2s-1.6s): "Hello beautiful {\1c&H0000FFFF&}world"

        # First, calculate timing for each word
        # CRITICAL: Use max(start_time, global_end_time) to prevent overlap with previous group
        word_timings = []  # List of (word, abs_start, abs_end)
        prev_word_end = max(start_time, global_end_time)

        for word_idx, word in enumerate(words_for_subtitle):
            word_global_idx = start_word_idx + word_idx

            # Get absolute end time for this word
            absolute_end = timepoint_map.get(word_global_idx, end_time)

            # Calculate duration (minimum 150ms)
            duration = max(absolute_end - prev_word_end, 0.15)

            word_start = prev_word_end
            word_end = prev_word_end + duration

            word_timings.append({
                'word': word,
                'start': word_start,
                'end': word_end
            })

            prev_word_end = word_end

        # Generate one dialogue line per word (each with that word highlighted)
        # Extend each word's end time to meet the next word's start to prevent flashing
        for highlight_idx, timing in enumerate(word_timings):
            # Build text with explicit colors for each word
            text_parts = []
            for word_idx, wt in enumerate(word_timings):
                word = wt['word']
                if word_idx == highlight_idx:
                    # This word is highlighted (yellow)
                    text_parts.append(f"{{\\1c&H0000FFFF&}}{word}")
                else:
                    # This word is not highlighted (white)
                    text_parts.append(f"{{\\1c&H00FFFFFF&}}{word}")

            # Join words with spaces
            karaoke_text = "{\\an2}" + " ".join(text_parts)

            # Dialogue timing: extend to next word's start (no gaps = no flashing)
            line_start = timing['start']
            if highlight_idx < len(word_timings) - 1:
                # Not the last word - extend to next word's start
                line_end = word_timings[highlight_idx + 1]['start']
            else:
                # Last word in phrase - check if there's more content
                if i < len(word_list):
                    # Get the start time of the next phrase's first word
                    next_word_idx = i
                    next_start = timepoint_map.get(next_word_idx - 1, timing['end']) if next_word_idx > 0 else timing['end']
                    line_end = max(timing['end'], next_start)
                else:
                    line_end = timing['end']

            dialogue_line = f"Dialogue: 0,{format_ass_time(line_start)},{format_ass_time(line_end)},Default,,0,0,0,,{karaoke_text}"
            ass_lines.append(dialogue_line)

            print(f"🎤 KARAOKE - Word {highlight_idx + 1}: '{timing['word']}' highlighted from {format_ass_time(line_start)} to {format_ass_time(line_end)}")

        # Update global_end_time to prevent overlap with next subtitle group
        if word_timings:
            global_end_time = word_timings[-1]['end']

        subtitle_index += 1
    
    # DEBUG: Print final ASS file summary
    print(f"\n✅ KARAOKE ASS FILE GENERATED:")
    print(f"   Total subtitles: {subtitle_index - 1}")
    print(f"   Total ASS lines: {len(ass_lines)}")
    print(f"   First dialogue line preview: {ass_lines[-1][:100]}...")
    print(f"==========================================\n")
    
    return '\n'.join(ass_lines)

def create_ass_fallback(script: str, audio_duration: float) -> str:
    """
    Fallback ASS generation when timepoints are not available.
    Uses same professional styling as create_ass_from_timepoints.
    """
    # Split script into words and group into 1-3 word phrases
    words = re.findall(r'\S+', script)
    
    if not words:
        return ""
    
    # Group words into 1-3 word phrases
    phrases = []
    i = 0
    while i < len(words):
        phrase_words = []
        # Natural grouping: 1-3 words
        for _ in range(min(3, len(words) - i)):
            word = words[i]
            phrase_words.append(word)
            i += 1
            
            # Stop early at sentence endings
            if re.search(r'[.!?]+$', word):
                break
        
        if phrase_words:
            phrases.append(' '.join(phrase_words))
    
    if not phrases:
        return ""
    
    ass_lines = [
        "[Script Info]",
        "Title: Professional Video Factory Subtitles",
        "ScriptType: v4.00+",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Style: Consistent with main function
        # PrimaryColour = White, Bold=1, Outline=10px (thick black border), MarginV=55px
        "Style: Default,Inter,16,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,1,0,0,0,100,100,0,0,1,10,1,2,10,10,55,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    
    duration_per_phrase = audio_duration / len(phrases)
    
    def format_ass_time(seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds % 1) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
    
    for i, phrase in enumerate(phrases):
        start_time = i * duration_per_phrase
        end_time = (i + 1) * duration_per_phrase if i < len(phrases) - 1 else audio_duration
        # Add \an2 for consistent bottom-center positioning
        ass_lines.append(f"Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(end_time)},Default,,0,0,0,,{{\\an2}}{phrase}")
    
    return '\n'.join(ass_lines)

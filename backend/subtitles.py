import re
from config import TTS_AVAILABLE, tts_client, texttospeech

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
            words_for_subtitle = []
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
            words_for_subtitle = []
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
    Create professional ASS (Advanced SubStation Alpha) subtitle file with karaoke effects.
    
    Requirements:
    - Bold sans-serif font (Inter Bold)
    - White text with dark stroke and subtle shadow
    - Font size: 68px
    - Outline: 4px, Shadow: 2px
    - Alignment: bottom-center (2)
    - Margin bottom: 180px
    - Maximum 3 words per subtitle line
    - Maximum 2 lines at once
    - Natural phrase grouping (1-3 words per highlight beat)
    - Karaoke highlighting with \\k tags (TikTok-style yellow/cyan)
    - Non-highlighted text stays white
    - Instant fade-in, no animation delay
    
    Args:
        script: The script text
        timepoints: List of timepoints from Google TTS
        audio_duration: Total audio duration in seconds
        style: Caption style - always "3words" for professional subtitles
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
    
    # Professional subtitle settings: 6-9 words per line for TikTok-style captions
    # Ideal duration per line: 1.5 - 3.5 seconds
    max_words_per_subtitle = 9
    min_words_per_subtitle = 1  # Allow natural phrase breaks
    
    # ASS file header with professional TikTok-style styling
    # Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, 
    #         Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, 
    #         Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
    # 
    # CRITICAL: ASS karaoke color logic (REVERSED from what you might expect):
    # - PrimaryColour = White (BGR: &H00FFFFFF) - inactive words (default)
    # - SecondaryColour = Yellow (BGR: &H0000FFFF) - active word during \k timing
    # - OutlineColour = Black (BGR: &H00000000) - stroke color
    # 
    # TikTok-style settings:
    # - Fontname=Inter, Fontsize=56, Bold=1, Outline=2, Shadow=0
    # - Alignment=2 (bottom-center), MarginV=180 (180px from bottom), BorderStyle=1
    ass_lines = [
        "[Script Info]",
        "Title: Professional Video Factory Subtitles",
        "ScriptType: v4.00+",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        # Style: Default, Inter Bold, 56px, White (both Primary and Secondary - explicit colors in tags handle karaoke)
        # Bold=1, Outline=2px, Shadow=0px, Alignment=2 (bottom-center), MarginV=180px, BorderStyle=1
        "Style: Default,Inter,56,&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,2,20,20,180,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    
    # DEBUG: Print style definition
    print(f"\n🎨 ASS STYLE DEFINITION:")
    print(f"   Font: Inter, Size: 56px, Bold: 1")
    print(f"   PrimaryColour (inactive): &H00FFFFFF (White in BGR)")
    print(f"   SecondaryColour (active): &H0000FFFF (Yellow in BGR)")
    print(f"   Outline: 2px, Shadow: 0px")
    print(f"   Alignment: 2 (bottom-center)")
    print(f"   MarginV: 250px (250px from bottom of screen)")
    print(f"   BorderStyle: 1")
    print(f"   Full style line: Style: Default,Inter,56,&H00FFFFFF,&H0000FFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,2,0,2,20,20,250,1")
    
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
        words_for_subtitle = []
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
        
        # Build karaoke text with \k tags for word-by-word highlighting
        # 
        # CRITICAL ASS KARAOKE LOGIC (REVERSED):
        # - PrimaryColour = White (inactive words - default)
        # - SecondaryColour = Yellow (active word during \k timing)
        # - \k tag switches text to SecondaryColour (yellow) for the duration
        # - When next \k starts, previous text reverts to PrimaryColour (white)
        #
        # Format: {\kXX}word where XX = centiseconds
        # Example: {\k12}This {\k18}is {\k22}karaoke
        # - "This" is yellow for 12cs, then white
        # - "is" is yellow for 18cs, then white
        # - "karaoke" is yellow for 22cs, then white
        #
        # IMPORTANT: Spaces must be INSIDE the karaoke tag to prevent visual jank
        # Use explicit color tags for renderer-safe karaoke
        karaoke_parts = []
        karaoke_debug = []  # Debug info for each word
        duration_seconds_list = []  # Track all durations for end_time calculation
        
        # Normalize karaoke timing to relative durations (monotonic)
        prev_word_end = start_time
        
        for word_idx, word in enumerate(words_for_subtitle):
            word_global_idx = start_word_idx + word_idx
            
            # Get absolute end time for this word
            absolute_end = timepoint_map.get(word_global_idx, end_time)
            
            # Calculate duration from previous word end (relative, monotonic)
            # Minimum 150ms (15cs) to prevent flicker and ensure karaoke works
            duration = max(absolute_end - prev_word_end, 0.15)
            duration_cs = int(duration * 100)
            duration_seconds_list.append(duration)
            
            # Update previous word end for next iteration
            prev_word_end = absolute_end
            
            # Attach non-breaking space to word except last word (spaces must be INSIDE karaoke tag)
            text = word + ("\u00A0" if word_idx < len(words_for_subtitle) - 1 else "")
            
            # Explicit color tags for renderer-safe karaoke using \k (karaoke):
            # {\kXX\c&H0000FFFF&}text{\c&H00FFFFFF&}
            # - \kXX = karaoke timing in centiseconds
            # - \c&H0000FFFF& = explicit yellow color (active word)
            # - text = the word
            # - \c&H00FFFFFF& = explicit white color (revert to inactive)
            karaoke_tag = (
                f"{{\\k{duration_cs}\\c&H0000FFFF&}}"
                f"{text}"
                f"{{\\c&H00FFFFFF&}}"
            )
            karaoke_parts.append(karaoke_tag)
            
            # Debug info (capture start time before updating prev_word_end)
            word_start_time = prev_word_end
            karaoke_debug.append({
                'word': word,
                'word_idx': word_global_idx,
                'start': word_start_time,
                'end': absolute_end,
                'duration_seconds': duration,
                'duration_cs': duration_cs,
                'tag': karaoke_tag
            })
        
        karaoke_text = "".join(karaoke_parts)
        
        # CRITICAL: Compute dialogue end from karaoke time, not timepoints
        # end_time = start_time + sum(duration_seconds) + 0.20
        sum_duration_seconds = sum(duration_seconds_list)
        end_time = start_time + sum_duration_seconds + 0.20
        end_time = min(end_time, audio_duration)
        
        # CRITICAL: Verify karaoke timing invariant
        # sum(duration_cs_list) must be less than (end_time - start_time) * 100
        duration_cs_list = [int(d * 100) for d in duration_seconds_list]
        sum_duration_cs = sum(duration_cs_list)
        total_dialogue_duration_cs = (end_time - start_time) * 100
        
        print(f"🔍 KARAOKE INVARIANT CHECK:")
        print(f"   sum(duration_cs_list) = {sum_duration_cs}cs")
        print(f"   (end_time - start_time) * 100 = {total_dialogue_duration_cs}cs")
        print(f"   Assertion: {sum_duration_cs} < {total_dialogue_duration_cs}")
        
        assert sum_duration_cs < total_dialogue_duration_cs, (
            f"Karaoke timing invariant failed! "
            f"sum(duration_cs_list)={sum_duration_cs} >= (end_time - start_time) * 100={total_dialogue_duration_cs}. "
            f"Karaoke cannot animate correctly."
        )
        
        print(f"   ✅ Invariant passed!")
        
        # DEBUG: Print karaoke info for this subtitle
        print(f"🎤 KARAOKE DEBUG - Subtitle {subtitle_index}:")
        print(f"   Words: {words_for_subtitle}")
        print(f"   Timing: {format_ass_time(start_time)} -> {format_ass_time(end_time)}")
        print(f"   Generated karaoke text: {karaoke_text}")
        for dbg in karaoke_debug:
            # Use raw string or double backslashes for the \k tag in print
            k_tag = f"\\k{dbg['duration_cs']}"
            print(f"     Word '{dbg['word']}': {dbg['duration_cs']}cs (\\k{dbg['duration_cs']}) | Duration: {dbg['duration_seconds']:.3f}s | Time: {dbg['start']:.3f}s -> {dbg['end']:.3f}s")
        print(f"   Full dialogue line: Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(end_time)},Default,,0,0,0,,{karaoke_text}")
        
        # Add dialogue line with no fade effects (instant appearance)
        # Format: Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        dialogue_line = f"Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(end_time)},Default,,0,0,0,,{karaoke_text}"
        ass_lines.append(dialogue_line)
        
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
        # Style: Default, Inter Bold, 68px, Yellow highlight, White text, Black stroke, No background
        # Bold=1, Outline=4px, Shadow=2px, Alignment=2 (bottom-center), MarginV=180px
        "Style: Default,Inter,68,&H0000FFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,4,2,2,20,20,250,1",
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
        ass_lines.append(f"Dialogue: 0,{format_ass_time(start_time)},{format_ass_time(end_time)},Default,,0,0,0,,{phrase}")
    
    return '\n'.join(ass_lines)

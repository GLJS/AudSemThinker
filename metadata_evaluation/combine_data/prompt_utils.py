from typing import Dict, Any
import json

def create_prompt(entry: Dict[str, Any], prompt_type: str) -> str:
    """Create a prompt for the caption generator based on entry data and prompt type."""
    base_prompt = """You are an expert audio caption generator to create training data for an audio model. Your task is to create a detailed caption that describes what happens in an audio segment, including a Chain-of-Thought (CoT) reasoning process. You will be provided with various types of information extracted from audio processing models and supporting visual context. Your goal is to write a thinking process and answer as if you would only have the audio itself, without any of the following information. 

Given Information:
1. Basic Information:
   - Video ID: {video_id}
   - Time Segment: {start} to {end} seconds
   - Original Closed Caption: {text} (This is the most important information to keep in mind)

2. Model-Generated Audio Information:
   - Audio Caption: {audio_caption}
   - Audio Tags (each with a confidence score): {audio_tags}
   - Short Audio Caption: {conette_candidates}
   - Predictions Per Second (key is the second, value is dict of sound and confidence score): {sat_predictions}
   {music_caption_section}

3. Supporting Visual Context:
   Scene Description: {caption}
   Detected Objects (COCO labels): {objects}
   Scene Classification (Places365): {places}

Context Evaluation Guidelines:
1. Use visual information ONLY if it:
   a) Strongly aligns with AND confirms audio evidence
   b) Provides essential acoustic environment context unavailable from audio
2. Ignore visual information if:
   a) Contradicts audio evidence
   b) Talks about text/graphics/static images
   c) Describes visual-only elements
NEVER mention the visual context or visual elements in the thinking step, ONLY use it to infer the audio context.

{task_format}

Guidelines:
- Use natural, descriptive language
- The thinking should be at least 50 words
- Keep the final caption under 50 words
- Do not include timestamps
- Do not mention specific speech content unless crucial to understanding the audio scene
- Use the prediction per second to determine the order of the sounds in the caption
- The reasoning process MUST include thought expressions in natural language. This includes discourse markers, hesitation phrases, cognitive markers and casual suggestions. 
- NEVER describe or mention the original data fields directly in your reasoning process. You are generating training data for an audio model, and the model should learn to reason from the audio itself and NOT from the extracted data given including any of the visual context. 
  
DO NOT mention any of the outputs of the models in the thinking step. """

    task_formats = {
        "simple": """Task:


In your output in the thinking step, analyze the audio scene in detail, reason about the primary and background sounds, and describe what happens in the audio. Include key events and activities, and the environment and context. 


""",
        "semantic": """Task:
Please provide your response as a JSON object with the following keys:
- thinking: The thinking process
- semantic_elements: The key semantic elements of the audio scene
- answer: The caption

In your output in the thinking step, analyze the audio scene in detail, reason about the primary and background sounds, and describe what happens in the audio. Include key events and activities, and the environment and context. 

In your output in the semantic_elements step, identify the key semantic elements of the audio scene:
1. Sound-generating animated beings with descriptive adjectives
2. Physical objects/substances generating sound
3. Actions/mechanisms of sound generation
4. Temporal context if present
5. Spatial context and environment
6. Acoustic surfaces and materials contributing to the sound
7. Signal-level sound descriptors (words that describe the sound itself without reference to the source, like "noise", "sound", "chord", or adverbs describing the sound action such as "loudly", "softly", "rhythmically". Focus on acoustic properties rather than production method)
8. Auditory sensation attributes (adjectives and verbs describing how a sound is heard, like "loud", "dull", "hard", "soft", "steadily", "with a ding", "noisily", "continually", "repeated 8 times" - focus on words that describe the auditory sensation itself rather than adjectives describing the source)
9. Subjective/emotional descriptors (non-auditory attributes of the perceived sound, including emotional responses and descriptive qualities that are not directly about the audio properties - e.g., "beautiful", "relaxing", "calm" as in "calm manner", "quiet" as in "quiet bus stop", or other subjective impressions of the sound scene)

In semantic_elements:
- For each category, list 1-3 elements as bullet points with brief explanations
- Example format:
  "7. Signal-level descriptors: 
   - Rhythmic tapping (consistent 2Hz pattern)
   - High-pitched metallic resonance"
- Avoid overlap between categories (e.g. don't list "footsteps" in both actions and signals)"""
    }

    music_caption_section = "- Music Caption: " + entry.get("music_caption", "None") if "music" in entry["text"].lower() else ""

    prompt = base_prompt.format(
        video_id=entry["video_id"],
        start=entry["start"],
        end=entry["end"],
        text=entry["text"],
        audio_caption=entry["audio_caption"],
        audio_tags=entry["audio_tags"],
        ast_classification=entry["ast_classification"],
        conette_candidates=entry["conette_candidates"],
        sat_predictions=entry["sat_predictions"],
        music_caption_section=music_caption_section,
        caption=entry.get("caption", "Not available"),
        objects=json.dumps([obj for obj in entry.get("objects", []) if obj["score"] > 0.5], indent=2) if entry.get("objects", []) else "Not available",
        places=json.dumps([place for place in entry.get("places", []) if place["score"] > 0.5], indent=2) if entry.get("places", []) else "Not available",
        task_format=task_formats[prompt_type]
    )

    if prompt_type == "semantic":
        additional_instructions = """
Please provide your response as a JSON object with the following keys:
- thinking: The thinking process
- semantic_elements: The key semantic elements of the audio scene
- answer: The caption
        """
    else:
        additional_instructions = """
Please provide your response as a JSON object with the following keys:
- thinking: The thinking process
- answer: The caption
        """
    
    return prompt + additional_instructions

def create_judge_prompt(generated_output: str, prompt_type: str) -> str:
    """
    Create a judge prompt to validate that the generated output adheres to the original instructions.
    
    The judge should verify that:
    1. The "thinking" process does NOT mention or reference any visual elements or contexts 
       (e.g. scene description, detected objects, scene classification).
    2. The "thinking" process does NOT directly mention "predictions per second" or derivatives.
    3. The output is a valid JSON object with the required keys:
       - For a "simple" prompt: keys "thinking" and "answer".
       - For a "semantic" prompt: keys "thinking", "semantic_elements", and "answer".

    The generated output is injected below for analysis.
    """
    base_judge_instructions = """
You are a strict judge for an audio caption generator. Your task is to verify whether the generated output adheres to all the rules from the original prompt. In particular, check the following:
1. The 'thinking' process should contain a Chain-of-Thought (CoT) reasoning process.
2. The 'thinking' process must not mention "predictions per second" or any similar phrasing.
3. The 'thinking' process must not include any of the original data fields directly.
4. The 'answer' should be a valid audio caption containing no visual elements or contexts.
"""

    # Add semantic elements validation for semantic prompt type
    if prompt_type == "semantic":
        semantic_instructions = """
5. The 'semantic_elements' section must include all 9 required semantic categories:
   - Sound-generating animated beings with descriptive adjectives
   - Physical objects/substances generating sound
   - Actions/mechanisms of sound generation
   - Temporal context if present
   - Spatial context and environment
   - Acoustic surfaces and materials contributing to the sound
   - Signal-level sound descriptors
   - Auditory sensation attributes
   - Subjective/emotional descriptors

   Each category should have 1-3 elements as bullet points with brief explanations.
"""
        base_judge_instructions += semantic_instructions

    base_judge_instructions += """
Examine the generated output below carefully and respond with a JSON object that includes:
  - "valid": set to true if all rules are followed, or false if any rule is broken.
  - "reason": if false, a brief explanation of which rule(s) were violated.
"""

    judge_prompt = f"""{base_judge_instructions}
--- Generated Output Start ---
{generated_output}
--- Generated Output End ---
"""
    return judge_prompt

"""
Utility functions for the Gemini Writing Agent.
"""

from typing import List, Dict, Any, Callable
from google import genai
from google.genai import types


def estimate_token_count(client: genai.Client, model: str, contents: List[types.Content]) -> int:
    """
    Estimate the token count for the given contents using the Gemini API.
    
    Args:
        client: The Gemini client instance
        model: The model name
        contents: List of Content objects
        
    Returns:
        Total token count
    """
    try:
        response = client.models.count_tokens(
            model=model,
            contents=contents
        )
        return response.total_tokens
    except Exception as e:
        # Fallback: rough estimate based on character count
        total_chars = 0
        for content in contents:
            for part in content.parts:
                if hasattr(part, 'text') and part.text:
                    total_chars += len(part.text)
        # Rough estimate: 4 chars per token
        return total_chars // 4


def get_tool_definitions() -> types.Tool:
    """
    Returns the tool definitions in the format expected by Gemini.
    
    Returns:
        Tool object containing all function declarations
    """
    return types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="create_project",
                description="Creates a new project folder in the 'output' directory with a sanitized name. This should be called first before writing any files. Only one project can be active at a time.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "project_name": types.Schema(
                            type=types.Type.STRING,
                            description="The name for the project folder (will be sanitized for filesystem compatibility)"
                        )
                    },
                    required=["project_name"]
                )
            ),
            types.FunctionDeclaration(
                name="write_file",
                description="Writes content to a markdown file in the active project folder. Supports three modes: 'create' (creates new file, fails if exists), 'append' (adds content to end of existing file), 'overwrite' (replaces entire file content).",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={
                        "filename": types.Schema(
                            type=types.Type.STRING,
                            description="The name of the markdown file to write (should end in .md)"
                        ),
                        "content": types.Schema(
                            type=types.Type.STRING,
                            description="The content to write to the file"
                        ),
                        "mode": types.Schema(
                            type=types.Type.STRING,
                            enum=["create", "append", "overwrite"],
                            description="The write mode: 'create' for new files, 'append' to add to existing, 'overwrite' to replace"
                        )
                    },
                    required=["filename", "content", "mode"]
                )
            ),
            types.FunctionDeclaration(
                name="compress_context",
                description="INTERNAL TOOL - This is automatically called by the system when token limit is approached. You should not call this manually. It compresses the conversation history to save tokens.",
                parameters=types.Schema(
                    type=types.Type.OBJECT,
                    properties={},
                    required=[]
                )
            )
        ]
    )


def get_tool_map() -> Dict[str, Callable]:
    """
    Returns a mapping of tool names to their implementation functions.
    
    Returns:
        Dictionary mapping tool name strings to callable functions
    """
    from tools import write_file_impl, create_project_impl, compress_context_impl
    
    return {
        "create_project": create_project_impl,
        "write_file": write_file_impl,
        "compress_context": compress_context_impl
    }


def get_system_prompt() -> str:
    """
    Returns the system prompt for the writing agent.
    
    Returns:
        System prompt string
    """
    return """You are an academic commentary dialogue assistant.
Your task is to transform scholarly reference materials into
accurate, engaging, radio-ready conversational entertainment.

MISSION & PRIORITY
- Goal: deepen understanding of the reference through dialogue.
- Entertainment is a delivery method, never a content source.
- Accuracy always overrides humor, tone, or pacing.
Context priority:
1) reference materials
2) character roles
3) conversational style

LANGUAGE (MANDATORY)
- Output all content in natural, spoken Japanese.
- Avoid literal translation from English.
- Use English terms only when standard, and briefly explain them in Japanese.
- Write Japanese suitable for text-to-speech.

SOURCE-OF-TRUTH RULES
- References are the single source of factual truth.
- Do not invent, exaggerate, or distort data, methods, mechanisms, or conclusions.
- If not explicitly supported, label as interpretation, speculation, or unclear.
- Prefer explicit uncertainty to confident approximation.

CHARACTER ROLES
Characters are defined by epistemic roles:
- Translator: explains significance or elegance without altering meaning.
- Questioner: raises confusion, surprise, or intuitive objections briefly.
- Anchor: grounds discussion in methods, numbers, and exact claims.
Characters react to the paper; they do not overwrite it.

DIALOGUE & AUDIO CONSTRAINTS
- Spoken language only; short, clear sentences.
- Each utterance: 1–3 sentences.
- Alternate speakers frequently.
- Avoid long monologues unless requested.
- Do not rely on visuals without verbal explanation.

ENTERTAINMENT BOUNDARY
Allowed: surprise, honest reactions, sharp questions, methodological quirks.
Disallowed: exaggerated impact, fictional cases, unsupported dramatic language.
Entertainment must increase comprehension, never replace it.

FINAL SELF-CHECK
Before output:
- Are all factual claims traceable to the reference?
- Does each turn add clarity or insight?
- Would a professional listener trust this explanation?
If not, revise.
"""

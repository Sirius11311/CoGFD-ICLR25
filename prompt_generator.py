import json
import os
from typing import List, Dict
from openai import OpenAI
from pprint import pprint
from config import (OPENAI_API_KEY, BASE_URL, GENERATED_IMG_DIR)

def generate_image_prompts(
    concept_combination: str,
    sub_concepts: List[str],
    output_path: str,
    model: str = "gpt-4o"
) -> Dict:
    """
    Generate image prompts using an LLM and save them in JSON format.
    
    Args:
        concept_combination (str): Combined concept (e.g., "man and dog")
        sub_concepts (List[str]): List of sub-concepts (e.g., ["man", "dog"])
        output_path (str): Path to save the generated JSON file
        model (str, optional): OpenAI model to use. Defaults to "gpt-4o"
    
    Returns:
        Dict: Generated prompts in dictionary format
    """
    # Initialize the OpenAI client
    client = OpenAI(
        api_key=OPENAI_API_KEY,
        base_url=BASE_URL
    )
    
    system_prompt = """You are a creative prompt generator for image generation. 
    Generate 10 different, specific prompts for each concept. 
    For combined concepts, show interaction between elements.
    For individual concepts, showcase various scenarios.
    Use simple, clear English and avoid abstract descriptions.
    Generated prompts should be concise and to the point and not too long.
    IMPORTANT: Your response must be a valid JSON object with no additional text or explanation."""
    
    user_prompt = f"""Generate image prompts for:
    concept_combination: {concept_combination}
    sub_concepts: {', '.join(sub_concepts)}
    
    Output must be in JSON format with the following structure:
    {{
        "{concept_combination}": [
            "prompt 1",
            "prompt 2",
            ...
            "prompt 10"
        ],
        "{sub_concepts[0]}": [
            "prompt 1",
            "prompt 2",
            ...
            "prompt 10"
        ],
        "{sub_concepts[1]}": [
            "prompt 1",
            "prompt 2",
            ...
            "prompt 10"
        ]
    }}"""
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.7,
            response_format={ "type": "json_object" }
        )
        
        # Extract the generated JSON from the response
        generated_text = response.choices[0].message.content
        print("Raw response:", generated_text)  # Debug print
        
        # Try to parse the JSON
        try:
            prompts_dict = json.loads(generated_text)
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {str(e)}")
            print("Attempting to clean the response...")
            # Try to extract JSON from the response if it's wrapped in markdown or other text
            import re
            json_match = re.search(r'\{.*\}', generated_text, re.DOTALL)
            if json_match:
                cleaned_json = json_match.group(0)
                prompts_dict = json.loads(cleaned_json)
            else:
                raise ValueError("Could not extract valid JSON from response")
        
        # Validate the structure
        required_keys = [concept_combination] + sub_concepts
        for key in required_keys:
            if key not in prompts_dict:
                raise ValueError(f"Missing required key in response: {key}")
            if not isinstance(prompts_dict[key], list) or len(prompts_dict[key]) != 10:
                raise ValueError(f"Invalid format for key {key}: expected list of 10 prompts")
        
        # Save to file
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(prompts_dict, f, indent=2)
            
        print(f"Successfully generated and saved prompts to {output_path}")
        return prompts_dict
        
    except Exception as e:
        print(f"Error generating prompts: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    concept = "man_and_basketball"
    sub_concepts = ["man", "basketball"]
    output_path = f"{GENERATED_IMG_DIR.format(concept_combination=concept)}/{concept}.json"
    
    try:
        prompts = generate_image_prompts(
            concept_combination=concept.replace('_', ' '),
            sub_concepts=sub_concepts,
            output_path=output_path,
        )
        print("Generated prompts:")
        pprint(prompts)
    except Exception as e:
        print(f"Failed to generate prompts: {str(e)}") 
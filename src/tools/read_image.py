import os
import base64
import requests
from openai import OpenAI
from typing import Union, Dict
from src.settings import settings
from src.utils.clients import groq_client

try:     
    client = groq_client

    MODEL = settings.VISION_MODEL_NAME
except (ValueError, ImportError) as e:
    print(f"Error: {e}")
    client = None


def _encode_image_from_path(image_path: str) -> str:
    """
    (Internal) Encodes a local image file to a base64 string.

    Args:
        image_path: The file path to the local image.

    Returns:
        A base64 encoded string of the image or None if an error occurs.
    """
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"Error: The file was not found at {image_path}")
        return None
    except Exception as e:
        print(f"An error occurred while encoding the image from path: {e}")
        return None

def _encode_image_from_url(image_url: str) -> str:
    """
    (Internal) Downloads an image from a URL and encodes it to a base64 string.

    Args:
        image_url: The URL of the image to download.

    Returns:
        A base64 encoded string of the image, or None if download fails.
    """
    try:
        response = requests.get(image_url, stream=True)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        return base64.b64encode(response.content).decode('utf-8')
    except requests.exceptions.RequestException as e:
        print(f"Error downloading image from URL {image_url}: {e}")
        return None
    except Exception as e:
        print(f"An error occurred while encoding the image from URL: {e}")
        return None


def extract_data_from_image(image_source: Union[str, Dict[str, str]], prompt: str = "Extract data from this image") -> str:
    """
    Extracts data from an image provided as a local file path, a URL, or a dictionary.

    Args:
        image_source: The source of the image. Can be:
                      - A string containing a local file path.
                      - A string containing a public URL.
                      - A dictionary of the form {'image_url': '<path_or_url>'}.
        prompt: The text prompt to send along with the image.

    Returns:
        The content of the chat completion response, or an error message.
    """
    if not client:
        return "Client not initialized."

    # if image_source is a dictionary, extract the 'image_url' key
    actual_source_path = ""
    if isinstance(image_source, dict):
        actual_source_path = image_source.get('image_url')
        if not actual_source_path:
            return "Error: If image_source is a dictionary, it must contain an 'image_url' key."
    elif isinstance(image_source, str):
        actual_source_path = image_source
    else:
        return f"Error: Unsupported type for image_source: {type(image_source)}. Must be a str or dict."

    base64_image = None
    # Check if the source is a URL or a local file path
    if actual_source_path.startswith('http://') or actual_source_path.startswith('https://'):
        base64_image = _encode_image_from_url(actual_source_path)
    else:
        base64_image = _encode_image_from_path(actual_source_path)

    if not base64_image:
        return "Could not get image data from the source."

    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                            
                                "url": f"data:image/jpeg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            model=MODEL,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred during the API call: {e}"

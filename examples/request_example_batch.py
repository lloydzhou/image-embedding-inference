import requests
import base64
from pathlib import Path

def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def get_image_embedding(image_paths: list[str], api_url: str="http://0.0.0.0:8000/embed") -> list[list[float]]:
    """Get embeddings for an image using the API."""
    # Encode the image
    base64_images = [encode_image(image_path) for image_path in image_paths]
    
    # Prepare the request
    payload = {
        "inputs": base64_images
    }
    
    print(f"Request with {len(base64_images)} images")
    # Send request to the API
    response = requests.post(api_url, json=payload)
    
    # Check if request was successful
    response.raise_for_status()
    
    # Return the embeddings
    return response.json()

if __name__ == "__main__":
    # Example usage
    images_dir = Path(__file__).parent.parent / "examples" / "images" 

    # Get all images in the directory
    image_paths = [str(image_path) for image_path in images_dir.iterdir()]
    
    try:
        helth = requests.get("http://0.0.0.0:8000/health")
        if helth.status_code != 200:
            print("Health check failed")
            exit(1)
        else:
            print("Health check passed")

        embeddings = get_image_embedding(image_paths)
        print(f"Successfully computed embeddings: {embeddings}")
        print(f"Successfully computed {len(image_paths)} embeddings.")
        print(f"Embeddings with dimension: {len(embeddings[0])}")
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Error: {e}")

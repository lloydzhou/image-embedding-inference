import asyncio
import aiohttp
import requests
import base64
from pathlib import Path

def encode_image(image_path: str) -> str:
    """Encode an image file to base64."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

async def get_image_embedding(base64_image: str, api_url: str="http://0.0.0.0:8000/embed") -> list[list[float]]:
    """Get embeddings for an image using the API."""
    # Prepare the request
    payload = {
        "inputs": [base64_image]
    }
    
    # Send request to the API
    async with aiohttp.ClientSession() as session:
        response = await session.post(api_url, json=payload)
    
    # Check if request was successful
    response.raise_for_status()
    
    # Return the embeddings
    return await response.json()

async def main():    # Example usage
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

        base64_images = [encode_image(image_path) for image_path in image_paths]

        tasks = [asyncio.create_task(get_image_embedding(base64_image), name=image_path) for image_path, base64_image in zip(image_paths,base64_images)]
        await asyncio.gather(*tasks)

        for task in tasks:
            image_path = task.get_name()
            embeddings = task.result()
            print(f"Successfully computed embeddings for {image_path}: {embeddings}")

        print(f"Successfully computed {len(base64_images)} embeddings.")
    except requests.exceptions.RequestException as e:
        print(f"Error making request: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())

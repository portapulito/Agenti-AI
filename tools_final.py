
import os
import requests
import time
from typing import Dict, List
from google.adk.tools import ToolContext
from google.adk.tools.load_artifacts_tool import load_artifacts_tool
from google.genai import types
from PIL import Image


def scrape_images_from_urls(
    tool_context: ToolContext,
    image_urls: str,
) -> Dict:
    """
    Download images from URLs and save them locally.
    Will process all URLs provided in the string.

    Args:
        tool_context: ADK tool context
        image_urls: String containing image URLs (separated by newlines or commas)

    Returns:
        Dictionary with download results
    """
    
    try:
        # Prepare reference images directory
        ref_dir = os.path.join(os.getcwd(), "reference_images")
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)

        # Handle both string and list inputs
        if isinstance(image_urls, list):
            url_list = [url.strip() for url in image_urls if url.strip()]
        else:
            # Parse URLs from string (split by newlines or commas)
            url_list = []
            for url in image_urls.replace(',', '\n').split('\n'):
                url = url.strip()
                if url:
                    url_list.append(url)
        
        if not url_list:
            return {
                "status": "error",
                "message": "No valid URLs found in the provided string",
            }

        # Initialize variables
        images: List[str] = []
        successful_downloads = 0
        failed_downloads = 0

        # Initialize images in state if necessary
        if tool_context and "images" not in tool_context.state:
            tool_context.state["images"] = {}

        # Process each URL
        for i, image_url in enumerate(url_list):
            try:
                print(f"🔍 DEBUG: Processing URL {i+1}/{len(url_list)}: {image_url}")
                
                # Validate URL format
                if not image_url.startswith(('http://', 'https://')):
                    print(f"🔍 DEBUG: Invalid URL format: {image_url}")
                    print(f"Skipping invalid URL: {image_url}")
                    failed_downloads += 1
                    continue

                # Determine file extension from URL or default to jpg
                file_extension = 'jpg'
                if '.' in image_url.split('/')[-1]:
                    url_extension = image_url.split('.')[-1].split('?')[0].lower()
                    if url_extension in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                        file_extension = url_extension

                print(f"🔍 DEBUG: Detected file extension: {file_extension}")

                # Generate filename with timestamp
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                image_filename = f"downloaded_image_{timestamp}_{successful_downloads + 1}.{file_extension}"
                save_path = os.path.join(ref_dir, image_filename)
                
                print(f"🔍 DEBUG: Will save as: {save_path}")

                # Download the image
                print(f"🔍 DEBUG: Starting download with timeout=30...")
                response = requests.get(image_url, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                
                print(f"🔍 DEBUG: Response status code: {response.status_code}")
                print(f"🔍 DEBUG: Response headers: {dict(response.headers)}")
                
                if response.status_code != 200:
                    print(f"🔍 DEBUG: Non-200 status code: {response.status_code}")
                    print(f"Failed to download image from {image_url}. Status code: {response.status_code}")
                    failed_downloads += 1
                    continue

                # Validate that we got image content
                content_type = response.headers.get('content-type', '').lower()
                print(f"🔍 DEBUG: Content-Type: {content_type}")
                
                if not content_type.startswith('image/'):
                    print(f"🔍 DEBUG: Invalid content type - expected image/, got: {content_type}")
                    print(f"URL did not return image content: {image_url} (content-type: {content_type})")
                    failed_downloads += 1
                    continue

                print(f"🔍 DEBUG: Response content length: {len(response.content)} bytes")

                # Save the image
                print(f"🔍 DEBUG: Saving image to {save_path}...")
                with open(save_path, 'wb') as f:
                    f.write(response.content)

                # Verify the file was saved successfully
                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    file_size = os.path.getsize(save_path)
                    print(f"🔍 DEBUG: File saved successfully - size: {file_size} bytes")
                    
                    successful_downloads += 1
                    images.append(image_filename)

                    # Add to images state with empty string value for later analysis
                    if tool_context:
                        tool_context.state["images"][image_filename] = ""
                        
                    print(f"✅ Successfully downloaded: {image_filename}")
                else:
                    print(f"🔍 DEBUG: File save verification failed")
                    print(f"🔍 DEBUG: File exists: {os.path.exists(save_path)}")
                    if os.path.exists(save_path):
                        print(f"🔍 DEBUG: File size: {os.path.getsize(save_path)}")
                    print(f"Failed to save image: {image_filename}")
                    failed_downloads += 1

            except requests.RequestException as e:
                print(f"🔍 DEBUG: RequestException: {type(e).__name__}: {e}")
                print(f"Network error downloading {image_url}: {str(e)}")
                failed_downloads += 1
                continue
            except Exception as e:
                print(f"🔍 DEBUG: Unexpected exception: {type(e).__name__}: {e}")
                print(f"Error processing {image_url}: {str(e)}")
                failed_downloads += 1
                continue

        # Return success, but note if some downloads failed
        status = "success"
        message = f"Successfully downloaded {len(images)} images"
        
        if failed_downloads > 0:
            if len(images) == 0:
                status = "error"
                message = f"Could not download any images. All {failed_downloads} downloads failed"
            else:
                status = "partial_success" 
                message += f". {failed_downloads} downloads failed"

        return {
            "status": status,
            "message": message,
            "total_urls_processed": len(url_list),
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "images": images,
        }

    except Exception as e:
        error_message = f"Error downloading images: {str(e)}"
        print(error_message)
        return {"status": "error", "message": error_message}


async def save_image_as_artifact(
    image_filename: str,
    tool_context: ToolContext,
) -> Dict:
    """
    Save a downloaded image as an artifact following ADK official patterns.
    
    Args:
        image_filename: Name of the image file to save as artifact
        tool_context: ADK tool context
        
    Returns:
        Dictionary with save results
    """
    try:
        # Get reference images directory
        ref_dir = os.path.join(os.getcwd(), "reference_images")
        image_path = os.path.join(ref_dir, image_filename)
        
        print(f"🔍 DEBUG: Saving image as artifact: {image_filename}")
        print(f"🔍 DEBUG: Image path: {image_path}")
        
        if not os.path.exists(image_path):
            return {
                "status": "error",
                "message": f"Image file not found: {image_filename}",
            }
        
        # Verify images exists in state
        if "images" not in tool_context.state:
            tool_context.state["images"] = {}
        
        # Read the original image bytes (NOT using PIL.tobytes())
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        
        print(f"🔍 DEBUG: Read {len(image_bytes)} bytes from file")
        
        # Determine MIME type from file extension
        file_extension = image_filename.lower().split('.')[-1]
        mime_type_map = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg', 
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        mime_type = mime_type_map.get(file_extension, 'image/jpeg')
        print(f"🔍 DEBUG: Detected MIME type: {mime_type}")
        
        # Save as artifact using original file bytes (CORRECT METHOD)
        await tool_context.save_artifact(
            image_filename,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        )
        
        print(f"🔍 DEBUG: Artifact saved successfully")
        
        # Update state
        tool_context.state["images"][image_filename] = "artifact_saved"
        tool_context.state["current_image"] = image_filename
        
        return {
            "status": "success",
            "message": f"Image saved as artifact: {image_filename}",
            "filename": image_filename,
        }
        
    except Exception as e:
        error_message = f"Error saving image as artifact: {str(e)}"
        print(f"🔍 DEBUG: Exception: {e}")
        print(error_message)
        return {"status": "error", "message": error_message}



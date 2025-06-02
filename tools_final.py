import os
import requests
import time
import logging
from typing import Dict, List, Any # Removed Union

from google.adk.tools import ToolContext
from google.adk.tools.load_artifacts_tool import load_artifacts_tool # Not used directly, but good to keep if other tools might
from google.genai import types
from PIL import Image # Used in display_downloaded_images if IPython is available


def scrape_images_from_urls(
    tool_context: ToolContext,
    image_urls: str, # Changed from Union[str, List[str]]
) -> Dict[str, Any]:
    """
    Downloads images from a newline/comma-separated string of URLs.
    Saves them to a local directory "reference_images". Updates tool_context.state with downloaded image filenames.

    Args:
        tool_context: The ADK (Agent Development Kit) tool context, providing access to state and artifact saving.
                      `tool_context.state["images"]` (Dict[str, str]) will be initialized if not present,
                      and populated with {image_filename: ""} for each successfully downloaded image.
        image_urls: A string containing one or more image URLs, separated by newlines or commas.

    Returns:
        Dict[str, Any]: A dictionary containing the results of the download operation.
            Keys include:
            - "status" (str): "success", "partial_success", or "error".
            - "message" (str): A summary message of the operation.
            - "total_urls_processed" (int): The total number of URLs identified for processing.
            - "successful_downloads" (int): The number of images successfully downloaded.
            - "failed_downloads" (int): The number of images that failed to download.
            - "images" (List[str]): A list of filenames for the successfully downloaded images.
            If an overall exception occurs, returns {"status": "error", "message": error_message}.
    """

    try:
        ref_dir: str = os.path.join(os.getcwd(), "reference_images")
        if not os.path.exists(ref_dir):
            os.makedirs(ref_dir)

        url_list: List[str]
        if isinstance(image_urls, str): # Ensure it's a string
            url_list = [url.strip() for url in image_urls.replace(',', '\n').split('\n') if url.strip()]
        else:
            # This case should ideally not be reached if type hinting is enforced by caller,
            # but as a safeguard:
            logging.error(f"image_urls was expected to be a string, but got {type(image_urls)}.")
            return {
                "status": "error",
                "message": f"Invalid type for image_urls. Must be a single string. Got {type(image_urls)}",
                "total_urls_processed": 0,
                "successful_downloads": 0,
                "failed_downloads": 0,
                "images": [],
            }

        if not url_list:
            logging.warning("No valid URLs found in the provided input string.")
            # Considered if this should be "success" if input string is empty vs "error".
            # "error" seems more appropriate if non-empty string yields no URLs.
            # If image_urls itself is empty, it could be "success" with 0 processed.
            # For now, keeping as error if url_list is empty after processing.
            return {
                "status": "error",
                "message": "No valid URLs found to process from the input string.",
                "total_urls_processed": 0,
                "successful_downloads": 0,
                "failed_downloads": 0,
                "images": [],
            }

        downloaded_image_filenames: List[str] = []
        successful_downloads: int = 0
        failed_downloads: int = 0

        if tool_context and "images" not in tool_context.state:
            tool_context.state["images"] = {}
        elif tool_context and not isinstance(tool_context.state.get("images"), dict):
            logging.warning("tool_context.state['images'] is not a dict. Re-initializing.")
            tool_context.state["images"] = {}


        for i, image_url_str in enumerate(url_list):
            try:
                logging.debug(f"Processing URL {i+1}/{len(url_list)}: {image_url_str}")

                if not image_url_str.startswith(('http://', 'https://')):
                    logging.debug(f"Invalid URL format: {image_url_str}")
                    logging.warning(f"Skipping invalid URL (must start with http/https): {image_url_str}")
                    failed_downloads += 1
                    continue

                file_extension: str = 'jpg' # Default extension
                try:
                    url_path_segment: str = image_url_str.split('/')[-1]
                    if '.' in url_path_segment:
                        possible_extension: str = url_path_segment.split('.')[-1].split('?')[0].lower()
                        if possible_extension in ['jpg', 'jpeg', 'png', 'gif', 'webp']:
                            file_extension = possible_extension
                except Exception: # pylint: disable=broad-except
                    logging.debug(f"Could not reliably determine extension for {image_url_str}, defaulting to .jpg")


                logging.debug(f"Detected file extension: {file_extension}")

                timestamp: str = time.strftime("%Y%m%d-%H%M%S")
                image_filename: str = f"downloaded_image_{timestamp}_{successful_downloads + 1}.{file_extension}"
                save_path: str = os.path.join(ref_dir, image_filename)

                logging.debug(f"Will save as: {save_path}")

                response: requests.Response = requests.get(image_url_str, timeout=30, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })

                logging.debug(f"Response status code: {response.status_code}")
                logging.debug(f"Response headers: {dict(response.headers)}")

                if response.status_code != 200:
                    logging.debug(f"Non-200 status code: {response.status_code}")
                    logging.warning(f"Failed to download image from {image_url_str}. Status code: {response.status_code}")
                    failed_downloads += 1
                    continue

                content_type: str = response.headers.get('content-type', '').lower()
                logging.debug(f"Content-Type: {content_type}")

                if not content_type.startswith('image/'):
                    logging.debug(f"Invalid content type - expected image/, got: {content_type}")
                    logging.warning(f"URL did not return image content: {image_url_str} (content-type: {content_type})")
                    failed_downloads += 1
                    continue

                logging.debug(f"Response content length: {len(response.content)} bytes")

                with open(save_path, 'wb') as f:
                    f.write(response.content)

                if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
                    file_size: int = os.path.getsize(save_path)
                    logging.debug(f"File saved successfully - size: {file_size} bytes")

                    successful_downloads += 1
                    downloaded_image_filenames.append(image_filename)

                    if tool_context and isinstance(tool_context.state.get("images"), dict):
                        tool_context.state["images"][image_filename] = ""

                    logging.info(f"Successfully downloaded: {image_filename}")
                else:
                    logging.debug(f"File save verification failed for {image_filename}")
                    logging.debug(f"File exists: {os.path.exists(save_path)}")
                    if os.path.exists(save_path):
                        logging.debug(f"File size: {os.path.getsize(save_path)}")
                    logging.error(f"Failed to save image: {image_filename}")
                    failed_downloads += 1

            except requests.RequestException as e:
                logging.debug(f"RequestException for {image_url_str}: {type(e).__name__}: {e}")
                logging.error(f"Network error downloading {image_url_str}: {str(e)}")
                failed_downloads += 1
            except Exception as e: # pylint: disable=broad-except
                logging.debug(f"Unexpected exception for {image_url_str}: {type(e).__name__}: {e}")
                logging.error(f"Error processing {image_url_str}: {str(e)}", exc_info=True)
                failed_downloads += 1

        final_status: str
        final_message: str
        if successful_downloads == 0 and failed_downloads > 0 and len(url_list) > 0 : # Ensure it's an error only if URLs were actually processed
            final_status = "error"
            final_message = f"Could not download any images. All {failed_downloads} downloads failed out of {len(url_list)} URLs processed."
        elif failed_downloads > 0:
            final_status = "partial_success"
            final_message = f"Successfully downloaded {successful_downloads} images. {failed_downloads} downloads failed out of {len(url_list)} URLs processed."
        elif successful_downloads > 0 :
            final_status = "success"
            final_message = f"Successfully downloaded all {successful_downloads} images out of {len(url_list)} URLs processed."
        else: # No URLs processed (e.g. image_urls was empty string or only whitespace)
            final_status = "success"
            final_message = "No image URLs were provided or found in the input string."


        if final_status == "error":
            logging.error(final_message)
        elif final_status == "partial_success":
            logging.warning(final_message)
        else:
            logging.info(final_message)

        return {
            "status": final_status,
            "message": final_message,
            "total_urls_processed": len(url_list),
            "successful_downloads": successful_downloads,
            "failed_downloads": failed_downloads,
            "images": downloaded_image_filenames,
        }

    except Exception as e: # pylint: disable=broad-except
        # Check if ref_dir was created, to customize message slightly
        ref_dir_exists = 'ref_dir' in locals() and os.path.exists(ref_dir) # type: ignore
        base_error_message: str = f"Overall error in scrape_images_from_urls: {str(e)}"
        if not ref_dir_exists : # type: ignore
            error_message = f"Failed to initialize image download environment (e.g., create directory '{os.path.join(os.getcwd(), 'reference_images')}'): {str(e)}"
        else:
            error_message = base_error_message

        logging.error(error_message, exc_info=True)
        return {
            "status": "error",
            "message": error_message, # Use the potentially more specific message
            "total_urls_processed": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "images": [],
        }


async def save_image_as_artifact(
    image_filename: str,
    tool_context: ToolContext,
) -> Dict[str, Any]:
    """
    Saves a locally downloaded image file as an ADK artifact.
    The image should exist in the "reference_images" directory.
    Updates `tool_context.state["images"][image_filename]` to "artifact_saved" and
    `tool_context.state["current_image"]` to `image_filename`.

    Args:
        image_filename: The name of the image file (e.g., "image.jpg") located in the "reference_images" directory.
        tool_context: The ADK tool context. `tool_context.state["images"]` is expected to be a dictionary.

    Returns:
        Dict[str, Any]: A dictionary containing the result of the save operation.
            Keys include:
            - "status" (str): "success" or "error".
            - "message" (str): A summary message.
            - "filename" (str, optional): The filename of the saved artifact, if successful.
            If an error occurs, returns {"status": "error", "message": error_message}.
    """
    try:
        ref_dir: str = os.path.join(os.getcwd(), "reference_images")
        image_path: str = os.path.join(ref_dir, image_filename)

        logging.debug(f"Attempting to save image as artifact: {image_filename}")
        logging.debug(f"Expected image path: {image_path}")

        if not os.path.exists(image_path):
            err_msg: str = f"Image file not found for artifact saving: {image_filename} at {image_path}"
            logging.error(err_msg)
            return {
                "status": "error",
                "message": err_msg,
            }

        if not (tool_context and isinstance(tool_context.state.get("images"), dict)):
            # Initialize or log warning if state is not as expected
            logging.warning("'images' dictionary not found or not a dict in tool_context.state. Initializing.")
            if tool_context:
                 tool_context.state["images"] = {}

        image_bytes: bytes
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        logging.debug(f"Read {len(image_bytes)} bytes from file {image_filename}")

        file_extension: str = image_filename.lower().split('.')[-1]
        mime_type_map: Dict[str, str] = {
            'jpg': 'image/jpeg',
            'jpeg': 'image/jpeg',
            'png': 'image/png',
            'gif': 'image/gif',
            'webp': 'image/webp'
        }
        mime_type: str = mime_type_map.get(file_extension, 'application/octet-stream') # Default MIME type
        logging.debug(f"Detected MIME type for {image_filename}: {mime_type}")

        await tool_context.save_artifact(
            artifact_name=image_filename, # Use image_filename as the unique artifact name
            value=types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        )

        logging.debug(f"Artifact {image_filename} saved successfully")

        if tool_context and isinstance(tool_context.state.get("images"), dict):
            tool_context.state["images"][image_filename] = "artifact_saved"
        if tool_context:
            tool_context.state["current_image"] = image_filename # type: ignore

        logging.info(f"Image saved as artifact: {image_filename}")
        return {
            "status": "success",
            "message": f"Image saved as artifact: {image_filename}",
            "filename": image_filename,
        }

    except Exception as e: # pylint: disable=broad-except
        error_message: str = f"Error saving image {image_filename} as artifact: {str(e)}"
        logging.error(error_message, exc_info=True)
        return {"status": "error", "message": error_message}


def display_downloaded_images(tool_context: ToolContext) -> Dict[str, Any]:
    """
    Attempts to display images listed in `tool_context.state["images"]` using IPython.display.
    Images are expected to be in the "reference_images" directory.

    Args:
        tool_context: The ADK tool context. `tool_context.state["images"]` (Dict[str, str])
                      is used to find the filenames of images to display.

    Returns:
        Dict[str, Any]: A dictionary summarizing the display operation.
            Keys include:
            - "status" (str): "success", "partial_success", or "error".
            - "message" (str): A summary of how many images were displayed, available, or missing.
            - "displayed_images" (List[str]): Filenames of images successfully displayed or marked as available
                                             if IPython is not supported.
            - "missing_images" (List[str]): Filenames of images that were not found on disk.
    """
    ipython_display_available: bool = True
    try:
        from IPython.display import Image as IPImage, display
    except ImportError:
        ipython_display_available = False
        logging.info("IPython.display not available. Images will not be displayed inline but marked as available.")

    ref_dir: str = os.path.join(os.getcwd(), "reference_images")

    if not (tool_context and isinstance(tool_context.state.get("images"), dict)):
        logging.error("Image tracking ('images' dict) not found in tool_context.state or is not a dictionary.")
        return {
            "status": "error",
            "message": "Image tracking not found in tool context or is not a dictionary.",
            "displayed_images": [],
            "missing_images": [],
        }

    if not tool_context.state["images"]:
        logging.info("No images have been downloaded or tracked yet.")
        return {
            "status": "success",
            "message": "No images have been downloaded yet.",
            "displayed_images": [],
            "missing_images": [],
        }

    displayed_images_list: List[str] = []
    missing_images_list: List[str] = []
    available_images_list: List[str] = [] # For when IPython isn't there

    image_filenames: List[str] = list(tool_context.state["images"].keys())

    for image_filename in image_filenames:
        image_path: str = os.path.join(ref_dir, image_filename)
        if os.path.exists(image_path):
            if ipython_display_available:
                try:
                    logging.info(f"Displaying image: {image_filename} from {image_path}")
                    display(IPImage(filename=image_path)) # Use aliased Image
                    displayed_images_list.append(image_filename)
                except Exception as e: # pylint: disable=broad-except
                    logging.warning(f"Error displaying image {image_filename} with IPython: {e}", exc_info=True)
                    available_images_list.append(image_filename)
            else:
                logging.info(f"Image available (display not supported): {image_filename} at {image_path}")
                available_images_list.append(image_filename)
        else:
            logging.warning(f"Image file not found at {image_path} for display.")
            missing_images_list.append(image_filename)

    successfully_processed_images: List[str] = displayed_images_list + available_images_list

    final_status: str
    final_message: str

    if not successfully_processed_images and not missing_images_list:
        # This case implies tool_context.state["images"] was empty, already handled.
        # Or, if it was not empty but all entries were somehow filtered before loop,
        # it's an unusual state.
        logging.info("No images were processed for display (list might have been empty or filtered).")
        final_status = "success"
        final_message = "No images were processed for display."
    elif missing_images_list:
        if successfully_processed_images:
            final_status = "partial_success"
            final_message = (
                f"Displayed/found {len(successfully_processed_images)} image(s). "
                f"Could not find {len(missing_images_list)} image(s) for display."
            )
            logging.warning(final_message)
        else:
            final_status = "error"
            final_message = f"All {len(missing_images_list)} image(s) scheduled for display were missing."
            logging.error(final_message)
    else:
        final_status = "success"
        final_message = f"Successfully displayed/found all {len(successfully_processed_images)} image(s)."
        logging.info(final_message)

    return {
        "status": final_status,
        "message": final_message,
        "displayed_images": successfully_processed_images,
        "missing_images": missing_images_list,
    }

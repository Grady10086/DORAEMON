import logging
import os
import torch
import numpy as np
import google.generativeai as genai
import cv2
import requests
import json
import base64
import io
from io import BytesIO
import time
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation, pipeline


class VLM:
    """
    Base class for a Vision-Language Model (VLM) agent. 
    This class should be extended to implement specific VLMs.
    """

    def __init__(self, **kwargs):
        """
        Initializes the VLM agent with optional parameters.
        """
        self.name = "not implemented"

    def call(self, images: list[np.array], text_prompt: str):
        """
        Perform inference with the VLM agent, passing images and a text prompt.

        Parameters
        ----------
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError
    
    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        """
        Perform context-aware inference with the VLM, incorporating past context.

        Parameters
        ----------
        history : int
            The number of context steps to keep for inference.
        images : list[np.array]
            A list of RGB image arrays.
        text_prompt : str
            The text prompt to be processed by the agent.
        """
        raise NotImplementedError

    def reset(self):
        """
        Reset the context state of the VLM agent.
        """
        pass

    def rewind(self):
        """
        Rewind the VLM agent one step by removing the last inference context.
        """
        pass

    def get_spend(self):
        """
        Retrieve the total cost or spend associated with the agent.
        """
        return 0


class GeminiVLM(VLM):
    """
    A specific implementation of a VLM using the Gemini API for image and text inference.
    """
    def __init__(self, model="gemini-1.5-pro-002", system_instruction=None):
        """
        Initialize the Gemini model with specified configuration.

        Parameters
        ----------
        model : str
            The model version to be used.
        system_instruction : str, optional
            System instructions for model behavior.
        """
        self.name = model
        self.base_url = "https://chataiapi.com/v1"
        self.api_key = os.environ.get("GEMINI_API_KEY")
        
        self.headers = {
            'Accept': 'application/json',
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        self.spend = 0
        self.cost_per_input_token = 0.075 / 1_000_000 if 'flash' in self.name else 1.25 / 1_000_000
        self.cost_per_output_token = 0.3 / 1_000_000 if 'flash' in self.name else 5 / 1_000_000
        
        self.conversation_history = []
        if system_instruction:
            self.conversation_history.append({
                "role": "system",
                "content": system_instruction
            })

        self.generation_config = {
            "temperature": 1,
            "top_p": 0.8,
            "top_k": 40,
            "max_output_tokens": 500,
            "response_mime_type": "text/plain",
        }

    def call_chat(self, history: int, images: list[np.array], text_prompt: str):
        max_retries = 10000 
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                content = [{"type": "text", "text": text_prompt}]
                content.extend(self._convert_images_to_base64(images))
                
                current_message = {
                    "role": "user",
                    "content": content
                }
                
                if history > 0:
                    messages = self.conversation_history[-2*history:] + [current_message]
                else:
                    messages = [current_message]

                payload = {
                    "model": self.name,
                    "messages": messages,
                    "stream": False,
                    "generation_config": self.generation_config 
                }
                
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers=self.headers,
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    response_text = result['choices'][0]['message']['content']
                    
                    self.conversation_history.append(current_message)
                    self.conversation_history.append({
                        "role": "assistant",
                        "content": response_text
                    })
                    
                    if 'usage' in result:
                        self.spend += (
                            result['usage']['prompt_tokens'] * self.cost_per_input_token +
                            result['usage']['completion_tokens'] * self.cost_per_output_token
                        )
                    
                    return response_text
                
                logging.error(f"API ERROR (Attempt {retry_count + 1}/{max_retries}): {response.text}")
                retry_count += 1
                if retry_count == max_retries:
                    return "API ERROR"
                time.sleep(0.5)  
                    
            except Exception as e:
                logging.error(f"API ERROR (Attempt {retry_count + 1}/{max_retries}): {e}")
                retry_count += 1
                if retry_count == max_retries:
                    return "API ERROR"
                time.sleep(1)

    def _convert_images_to_base64(self, images):
        content = []
        for i, image in enumerate(images):
            try:
                if image is None:
                    logging.warning(f"Skipping None image at index {i}")
                    continue

                if len(image.shape) == 2:
                    image = np.stack([image] * 3, axis=2)
                elif len(image.shape) == 3 and image.shape[2] != 3:
                    image = image[:, :, :3]

                if image.dtype != np.uint8:
                    image = (image * 255).astype(np.uint8)
                    
                # logging.info(f"Final image {i} shape: {image.shape}")

                pil_image = Image.fromarray(image)
                buffered = io.BytesIO()
                pil_image.save(buffered, format="JPEG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                content.append({
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{img_str}"
                })
                
            except Exception as e:
                logging.error(f"Error converting image {i}: {str(e)}")
                logging.error(f"Image info: shape={image.shape if image is not None else 'None'}, "
                            f"dtype={image.dtype if image is not None else 'None'}")
                
        return content

    def call(self, images: list[np.array], text_prompt: str):
        return self.call_chat(0, images, text_prompt)

    def reset(self):
        self.conversation_history = []

    def rewind(self):
        if len(self.conversation_history) >= 2:
            self.conversation_history = self.conversation_history[:-2]

    def get_spend(self):
        """
        Retrieve the total spend on model usage.
        """
        return self.spend


class DepthEstimator:
    """
    A class for depth estimation from images using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the depth estimation pipeline with the appropriate model.
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        checkpoint = "Intel/zoedepth-kitti"
        self.pipe = pipeline("depth-estimation", model=checkpoint, device=device)

    def call(self, image: np.array):
        """
        Perform depth estimation on an image.

        Parameters
        ----------
        image : np.array
            An RGB image for depth estimation.
        """
        original_shape = image.shape
        image_rgb = Image.fromarray(image[:, :, :3])
        depth_predictions = self.pipe(image_rgb)['predicted_depth']

        # Resize the depth map back to the original image dimensions
        depth_predictions = depth_predictions.squeeze().cpu().numpy()
        depth_predictions = cv2.resize(depth_predictions, (original_shape[1], original_shape[0]))

        return depth_predictions


class Segmentor:
    """
    A class for semantic segmentation using a pre-trained model.
    """

    def __init__(self):
        """
        Initialize the segmentation model and processor.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained("facebook/mask2former-swin-small-ade-semantic")
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained("facebook/mask2former-swin-small-ade-semantic").to(self.device)

        # Get class ids for navigable regions
        id2label = self.model.config.id2label
        self.navigability_class_ids = [id for id, label in id2label.items() if 'floor' in label.lower() or 'rug' in label.lower()]

    def get_navigability_mask(self, im: np.array):
        """
        Generate a navigability mask from an input image.

        Parameters
        ----------
        im : np.array
            An RGB image for generating the navigability mask.
        """
        image = Image.fromarray(im[:, :, :3])
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[image.size[::-1]])[0].cpu().numpy()

        navigability_mask = np.isin(predicted_semantic_map, self.navigability_class_ids)
        return navigability_mask

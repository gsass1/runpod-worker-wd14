import runpod
import requests
from PIL import Image
from io import BytesIO
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
import onnxruntime as rt

MODEL_REPO = "SmilingWolf/wd-swinv2-tagger-v3"
MODEL_FILENAME = "model.onnx"
LABEL_FILENAME = "selected_tags.csv"

def mcut_threshold(probs):
    """Maximum Cut Thresholding (MCut)"""
    sorted_probs = probs[probs.argsort()[::-1]]
    difs = sorted_probs[:-1] - sorted_probs[1:]
    t = difs.argmax()
    thresh = (sorted_probs[t] + sorted_probs[t + 1]) / 2
    return thresh

class WaifuDiffusionTagger:
    def __init__(self):
        self.model = None
        self.tag_names = None
        self.rating_indexes = None
        self.general_indexes = None
        self.character_indexes = None
        self.model_target_size = None
        self._initialize_model()

    def _initialize_model(self):
        # Download and load model
        model_path = hf_hub_download(MODEL_REPO, filename=MODEL_FILENAME)
        csv_path = hf_hub_download(MODEL_REPO, filename=LABEL_FILENAME)
        
        # Load model
        self.model = rt.InferenceSession(model_path)
        _, height, _, _ = self.model.get_inputs()[0].shape
        self.model_target_size = height

        # Load and process tags
        tags_df = pd.read_csv(csv_path)
        self.tag_names = tags_df["name"].tolist()
        self.rating_indexes = list(np.where(tags_df["category"] == 9)[0])
        self.general_indexes = list(np.where(tags_df["category"] == 0)[0])
        self.character_indexes = list(np.where(tags_df["category"] == 4)[0])

    def prepare_image(self, image):
        # Handle alpha channel
        canvas = Image.new("RGBA", image.size, (255, 255, 255))
        canvas.alpha_composite(image)
        image = canvas.convert("RGB")

        # Pad image to square
        image_shape = image.size
        max_dim = max(image_shape)
        pad_left = (max_dim - image_shape[0]) // 2
        pad_top = (max_dim - image_shape[1]) // 2

        padded_image = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
        padded_image.paste(image, (pad_left, pad_top))

        # Resize
        if max_dim != self.model_target_size:
            padded_image = padded_image.resize(
                (self.model_target_size, self.model_target_size),
                Image.BICUBIC,
            )

        # Convert to numpy array
        image_array = np.asarray(padded_image, dtype=np.float32)
        
        # Convert RGB to BGR
        image_array = image_array[:, :, ::-1]

        return np.expand_dims(image_array, axis=0)

    def predict(self, image, general_thresh=0.35, character_thresh=0.85, 
                general_mcut=False, character_mcut=False):
        # Prepare image
        image_array = self.prepare_image(image)

        # Run inference
        input_name = self.model.get_inputs()[0].name
        label_name = self.model.get_outputs()[0].name
        predictions = self.model.run([label_name], {input_name: image_array})[0]

        # Process predictions
        labels = list(zip(self.tag_names, predictions[0].astype(float)))

        # Get ratings
        ratings_names = [labels[i] for i in self.rating_indexes]
        rating = dict(ratings_names)

        # Process general tags
        general_names = [labels[i] for i in self.general_indexes]
        
        if general_mcut:
            general_probs = np.array([x[1] for x in general_names])
            general_thresh = mcut_threshold(general_probs)

        general_res = dict([x for x in general_names if x[1] > general_thresh])

        # Process character tags
        character_names = [labels[i] for i in self.character_indexes]
        
        if character_mcut:
            character_probs = np.array([x[1] for x in character_names])
            character_thresh = mcut_threshold(character_probs)
            character_thresh = max(0.15, character_thresh)

        character_res = dict([x for x in character_names if x[1] > character_thresh])

        return {
            "rating": rating,
            "general": general_res,
            "characters": character_res
        }

def download_image(url: str) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert('RGBA')

# Initialize tagger globally
tagger = WaifuDiffusionTagger()

def handler(event):
    try:
        # Extract parameters from the event
        image_url = event["input"]["image_url"]
        general_thresh = event["input"].get("general_threshold", 0.35)
        character_thresh = event["input"].get("character_threshold", 0.85)
        general_mcut = event["input"].get("general_mcut", False)
        character_mcut = event["input"].get("character_mcut", False)

        # Download and process image
        image = download_image(image_url)
        
        # Get predictions
        predictions = tagger.predict(
            image,
            general_thresh=general_thresh,
            character_thresh=character_thresh,
            general_mcut=general_mcut,
            character_mcut=character_mcut
        )
        
        return {
            "status": "success",
            "predictions": predictions
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

runpod.serverless.start({"handler": handler})

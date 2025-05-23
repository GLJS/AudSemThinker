import torch
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    RTDetrForObjectDetection, RTDetrImageProcessor
)
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
import torchvision
import torchvision.transforms as trn
from collections import defaultdict
import os
from PIL import Image
os.environ['YOLO_VERBOSE'] = 'False'
import urllib.request

class ImageModelHandler:
    def __init__(self, device: torch.device = torch.device("cuda:0"), batch_size: int = 4):
        self.device = device
        self.device_num = device.index
        self.batch_size = batch_size
        
        # Initialize image captioning model
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", torch_dtype=torch.float16).to(device)
        self.tokenizer = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        
        # Generation config for image captioning
        self.gen_kwargs = {
            "max_length": 20,
            "num_beams": 4
        }
        
        # Replace YOLO initialization with RT-DETR
        self.rtdetr_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd").to(device)
        self.rtdetr_model = self.rtdetr_model.eval()
        self.rtdetr_processor = RTDetrImageProcessor.from_pretrained("PekingU/rtdetr_r50vd")
        
        # Initialize Places365 CNN for scene recognition
        rel_path = os.path.join(os.path.dirname(__file__))
        weights_path = os.path.join(rel_path, "weights")
        data_path = os.path.join(rel_path, "categories")
        self.places365_model = torchvision.models.__dict__['resnet50'](num_classes=365)
        if not os.path.exists(os.path.join(weights_path, "resnet50_places365.pth.tar")):
            os.makedirs(weights_path, exist_ok=True)
            urllib.request.urlretrieve("http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar", os.path.join(weights_path, "resnet50_places365.pth.tar"))
        checkpoint = torch.load(os.path.join(weights_path, "resnet50_places365.pth.tar"), map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.places365_model.load_state_dict(state_dict)
        self.places365_model.to(device).eval()
        
        # Load categories
        with open(os.path.join(data_path, "categories_imagenet.txt")) as f:
            self.object_categories = [line.strip() for line in f.readlines()]
        with open(os.path.join(data_path, "categories_places365.txt")) as f:
            self.places365_categories = [line.strip() for line in f.readlines()]

        # Move models to GPU and set to eval mode once
        self.caption_model = self.caption_model.eval()
        self.places365_model = self.places365_model.eval()
        
        # Create category tensors for faster lookup
        self.object_categories_tensor = torch.tensor([hash(cat) for cat in self.object_categories], device=device)
        self.places365_categories_tensor = torch.tensor([hash(cat) for cat in self.places365_categories], device=device)

    def _deduplicate_predictions(self, labels: Union[List[str], torch.Tensor], scores: torch.Tensor, dtype: torch.dtype = torch.float16) -> Tuple[List[str], torch.Tensor]:
        """Helper function to deduplicate predictions and average scores using torch operations."""
        # Convert string labels to hashes for faster comparison
        if isinstance(labels, list):
            labels = [l for l in labels for l in l]
        label_hashes = torch.tensor([hash(label) for label in labels], device=self.device)
        
        if isinstance(scores, list):
            scores = torch.cat(scores, dim=0).to(dtype)
        
        # Get unique labels and their indices
        unique_hashes, inverse_indices = torch.unique(label_hashes, return_inverse=True)
        
        # Calculate mean scores for each unique label using scatter_add
        # Create tensors with same dtype as input scores
        unique_scores = torch.zeros_like(unique_hashes, dtype=dtype, device=self.device)
        count = torch.zeros_like(unique_hashes, dtype=dtype, device=self.device)
        
        # Use scatter_add_ to sum scores and counts
        unique_scores.scatter_add_(0, inverse_indices, scores)
        count.scatter_add_(0, inverse_indices, torch.ones_like(scores, dtype=dtype))
        
        # Calculate mean scores
        mean_scores = unique_scores / count
        
        # Convert back to original label format
        unique_labels = []
        for hash_val in unique_hashes.cpu().numpy():
            # Find the original label for this hash
            mask = label_hashes == hash_val
            idx = torch.where(mask)[0][0]
            unique_labels.append(labels[idx])
            
        return unique_labels, mean_scores

    def detect_objects(self, inputs: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor]:
        """Detect objects in the image using RT-DETR."""
        # Process batch with RT-DETR
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.rtdetr_model(**inputs)
        
        # Process results for each image
        results = self.rtdetr_processor.post_process_object_detection(
            outputs,
            target_sizes=torch.tensor([[img.shape[1], img.shape[2]] for img in inputs['pixel_values']]).to(self.device),
            threshold=0.3
        )
        
        # Format results
        all_labels = []
        all_scores = []
        for result in results:
            labels = [self.rtdetr_model.config.id2label[label_id.item()] for label_id in result["labels"]]
            all_labels.append(labels)
            all_scores.append(result["scores"] if result["scores"].shape[0] > 0 else torch.tensor([0.0]).to(self.device))
            
        return all_labels, all_scores

    def classify_places(self, places_input: torch.Tensor) -> Tuple[List[str], torch.Tensor]:
        """Classify places/scenes in the image."""
        outputs = self.places365_model(places_input)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top predictions
        top_probs, top_indices = probs.topk(1)
        top_probs = top_probs.squeeze(dim=1)
        top_indices = top_indices.squeeze(dim=1)
        mask = top_probs > 0.3
        
        # Filter valid predictions
        valid_probs = torch.where(mask, top_probs, torch.zeros_like(top_probs))
        valid_indices = torch.where(mask, top_indices, torch.zeros_like(top_indices))
        
        # Get categories for valid predictions
        categories = [self.places365_categories[idx.item()] if idx.item() != 0 else "unknown" for idx in valid_indices]
        
        return categories, valid_probs

    def generate_captions(self, pixel_values: torch.Tensor) -> List[str]:
        """Generate captions for images using the ViT-GPT2 model."""
        output_ids = self.caption_model.generate(pixel_values, **self.gen_kwargs)
        captions = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [caption.strip() for caption in captions]

    def process_video_batch(self, pixel_values: torch.Tensor,
                           places_values: torch.Tensor,
                           video_ids: List[str],
                           start_times: List[int],
                           end_times: List[int],
                           file_names: List[str],
                           pil_counts: List[int],
                           rtdetr_inputs: List[Dict[str, torch.Tensor]]) -> List[Dict[str, Any]]:
        """Process a batch of videos using multiple vision models."""
        # Pre-allocate GPU memory for inputs
        with torch.amp.autocast('cuda'), torch.no_grad():
            # Move all inputs to GPU at start
            pixel_values = pixel_values.to(self.device, non_blocking=True)
            places_values = places_values.to(self.device, non_blocking=True)

            # Process everything in parallel where possible
            object_labels, object_scores = self.detect_objects(rtdetr_inputs)
            place_labels, place_scores = self.classify_places(places_values)
            captions = self.generate_captions(pixel_values)

        assert len(object_labels) \
             == len(object_scores) \
             == len(place_labels) \
             == len(place_scores) \
             == len(captions) \
             == sum(pil_counts), f"Lengths do not match: {len(object_labels)} != {len(object_scores)} != {len(place_labels)} != {len(place_scores)} != {len(captions)} != {sum(pil_counts)}"

        # Process all results in a single loop
        results = []
        start_idx = 0
        for i, count in enumerate(pil_counts):
            end_idx = start_idx + count

            if len(object_labels[start_idx:end_idx]) == 0:
                video_object_scores = [{"label": "unknown", "score": 0.0}]
            else:
                # Get predictions for current video
                video_object_labels = [l for label in object_labels[start_idx:end_idx] for l in label]
                video_object_scores = torch.tensor([s for scores in object_scores[start_idx:end_idx] for s in scores if s > 0.0001])

                assert len(video_object_labels) == len(video_object_scores), f"Lengths do not match: {len(video_object_labels)} != {len(video_object_scores)}"

                index_dict = defaultdict(list)
                for j, label in enumerate(video_object_labels):
                    index_dict[label].append(video_object_scores[j].item())
                video_object_scores = [{'label': label, 'score': np.mean(scores)} for label, scores in index_dict.items()]
            
            if len(place_labels[start_idx:end_idx]) == 0:
                video_place_scores = [{"label": "unknown", "score": 0.0}]
            else:
                video_place_labels = place_labels[start_idx:end_idx]
                video_place_scores = place_scores[start_idx:end_idx]
                index_dict = defaultdict(list)
                for j, label in enumerate(video_place_labels):
                    index_dict[label].append(video_place_scores[j].item())
                video_place_scores = [{'label': label, 'score': np.mean(scores)} for label, scores in index_dict.items()]
            

            captions = captions[start_idx:end_idx]
            deduped_captions = list(set(captions))
            

            
            # Create result dict for current video
            results.append({
                'descriptions': deduped_captions,
                'objects': video_object_scores,
                'places': video_place_scores,
                'video_id': video_ids[i],
                'start_time': start_times[i],
                'end_time': end_times[i],
                'file_name': file_names[i]
            })
            
            start_idx = end_idx

        return results
import torch
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor, ASTForAudioClassification
from typing import Dict, Any, List, Optional
import os
import pandas as pd
from BEATs.BEATs import BEATs, BEATsConfig
from conette import CoNeTTEConfig, CoNeTTEModel
from lpmc.music_captioning.model.bart import BartCaptionModel
from lpmc.utils.eval_utils import load_pretrained
from omegaconf import OmegaConf

class AudioModelHandler:
    def __init__(self, device: str = "cuda:0"):
        self.device = device
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
        self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-Audio-7B-Instruct", 
            device_map="auto",
            torch_dtype="bfloat16"
        )
        self.model.eval()
        
        # Initialize AST model
        self.ast_model = ASTForAudioClassification.from_pretrained("shreyahegde/ast-finetuned-audioset-10-10-0.450_ESC50")
        self.ast_model.to(device)
        self.ast_model.eval()
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        checkpoint = torch.load(os.path.join(current_dir, "weights/BEATs_iter3_plus_AS2M_finetuned_on_AS2M_cpt1.pt"))
        cfg = BEATsConfig(checkpoint['cfg'])
        self.beats_model = BEATs(cfg)
        self.beats_model.load_state_dict(checkpoint['model'])
        self.beats_model.to(device)
        self.beats_model.eval()
        self.label_dict = checkpoint['label_dict']

        self.class_labels_indices = pd.read_csv(os.path.join(current_dir, "data/class_labels_indices.csv"))
        self.class_labels_indices = self.class_labels_indices.set_index('mid')['display_name'].to_dict()
        self.label_dict = {k: self.class_labels_indices[v] for k, v in self.label_dict.items()}

        config = CoNeTTEConfig.from_pretrained("Labbeti/conette")
        self.conette_model = CoNeTTEModel.from_pretrained("Labbeti/conette", config=config).to(self.device)
        self.conette_model.eval()

        # Initialize music captioning model
        current_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(current_dir, "lpmc/music_captioning/exp/transfer/lp_music_caps")
        config = OmegaConf.load(os.path.join(save_dir, "hparams.yaml"))
        self.music_caption_model = BartCaptionModel(max_length=config.max_length)
        self.music_caption_model, _ = load_pretrained(
            args={"gpu": 0}, 
            save_dir=save_dir, 
            model=self.music_caption_model,
            mdp=config.multiprocessing_distributed
        )
        self.music_caption_model.to(device)
        self.music_caption_model.eval()

    def get_audio_tags(self, waveform: torch.Tensor, waveform_padding_mask: torch.Tensor) -> List[Dict[str, float]]:
        """Get audio tags using BEATs model"""
        if self.beats_model is None:
            return []
            
        with torch.no_grad():
            
            # Get predictions
            probs = self.beats_model.extract_features(waveform, padding_mask=waveform_padding_mask)[0]
            
            # Get top 5 predictions for each sample
            top_k = 5
            top_probs, top_indices = probs.topk(k=top_k)
            
            results = []
            for sample_probs, sample_indices in zip(top_probs, top_indices):
                tags = {}
                for prob, idx in zip(sample_probs, sample_indices):
                    label = self.label_dict[idx.item()]
                    tags[label] = prob.item()
                results.append(tags)
                
            return results

    def get_music_caption(self, waveform: torch.Tensor) -> Optional[str]:
        """Generate music caption if audio contains music"""
        with torch.no_grad():
            output = self.music_caption_model.generate(
                samples=waveform,
                use_nucleus_sampling=True
            )
            return output if output and len(output) > 0 else None

    @torch.no_grad()
    def process_audio_batch(self, 
                          caption_inputs: Dict[str, torch.Tensor],
                          tagging_inputs: Dict[str, torch.Tensor], 
                          ast_inputs: Dict[str, torch.Tensor],
                          music_inputs: torch.Tensor,
                          conette_inputs: Dict[str, torch.Tensor],
                          video_ids: List[str],
                          start_times: List[float],
                          end_times: List[float],
                          file_names: List[str]) -> List[Dict[str, Any]]:
        """
        Process a batch of preprocessed audio inputs using Qwen2Audio model, BEATs model, and AST model.
        """
        # Generate caption response
        generate_ids = self.model.generate(
            input_ids=caption_inputs['input_ids'].to(self.device),
            input_features=caption_inputs['input_features'].to(self.device),
            attention_mask=caption_inputs['attention_mask'].to(self.device),
            feature_attention_mask=caption_inputs['feature_attention_mask'].to(self.device),
            max_length=1024
        )
        generate_ids = generate_ids[:, caption_inputs['input_ids'].size(1):]

        # Decode responses
        responses = self.processor.batch_decode(
            generate_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=False
        )
        
        # Get audio tags using waveform and padding mask
        audio_tags = self.get_audio_tags(
            tagging_inputs['waveform'].to(self.device),
            tagging_inputs['waveform_padding_mask'].to(self.device)
        )
        
        # Process with AST model
        ast_inputs = ast_inputs.to(self.device)
        ast_outputs = self.ast_model(**ast_inputs)
        ast_logits = ast_outputs.logits
        ast_predictions = torch.argmax(ast_logits, dim=-1)
        ast_labels = [self.ast_model.config.id2label[pred.item()] for pred in ast_predictions]

        # Process with Conette model
        outputs = self.conette_model(conette_inputs, sr=[32000]*len(conette_inputs))
        conette_candidates = outputs["cands"]

        # Check for music in tags and generate music captions if needed
        music_captions = self.get_music_caption(
            music_inputs.to(self.device)
        )

        assert len(responses) == len(audio_tags) == len(ast_labels) == len(conette_candidates) == len(music_captions) == len(video_ids) == len(start_times) == len(end_times) == len(file_names)

        return [
            {
                "audio_caption": response,
                "audio_tags": tags,
                "ast_classification": ast_label,
                "conette_candidates": conette_candidate,
                "music_caption": music_captions,
                "video_id": video_id,
                "start_time": start_time,
                "end_time": end_time,
                "file_name": file_name
            }
            for response, tags, ast_label, conette_candidate, music_captions, video_id, start_time, end_time, file_name
            in zip(responses, audio_tags, ast_labels, conette_candidates, music_captions, video_ids, start_times, end_times, file_names)
        ]
# Import necessary libraries
import os
import dotenv
import argparse
import torch
from transformers import Qwen2_5OmniThinkerForConditionalGeneration, AutoTokenizer
from peft import PeftModel

def export_peft_model(checkpoint_path, output_path, push_to_hub=False, repo_id="gijs/semthinker", private=True):
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B")
    
    # Load the base model
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-Omni-7B",
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )

    # Load the PEFT model
    model = PeftModel.from_pretrained(model, checkpoint_path)

    # Merge the adapter weights with the base model
    model = model.merge_and_unload()

    # Fix PEFT config issue before saving
    model._hf_peft_config_loaded = False

    # Either save to local path or push to Hub based on argument
    if push_to_hub:
        print(f"Pushing model to the Hub at {repo_id}")
        model.push_to_hub(repo_id, private=private)
        tokenizer.push_to_hub(repo_id, private=private)
    else:
        print(f"Saving model to {output_path}")
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
    
    print("Export completed successfully!")

def main():
    # Load environment variables
    dotenv.load_dotenv()
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Export and merge PEFT model")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="./checkpoints_final/audsemthinker",
                        help="Path to the checkpoint directory")
    parser.add_argument("--output_path", type=str,
                        default="./checkpoints_merged/audsemthinker",
                        help="Path to save the merged model")
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push the model to Hugging Face Hub instead of saving locally")
    parser.add_argument("--repo_id", type=str, default="gijs/semthinker",
                        help="Repository ID for Hugging Face Hub")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Whether to make the HF repo private")
    
    args = parser.parse_args()
    
    # Call the export function with parsed arguments
    export_peft_model(
        args.checkpoint_path,
        args.output_path,
        push_to_hub=args.push_to_hub,
        repo_id=args.repo_id,
        private=args.private
    )

if __name__ == "__main__":
    main()

import os
import dotenv
dotenv.load_dotenv("./training/.env")
import fire
import json
import logging
from tqdm import tqdm

import torch

from dataset import Dataset
from model import Model

# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  = 
logger = logging.getLogger(__name__)
logging.basicConfig(
    format  = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt = "%m/%d/%Y %H:%M:%S",
    level   = logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  = 

def do_model_prediction(input_data, model, batch_size):

    if batch_size not in [1, -1]:
        logger.info(f"model: {model.model_name}")
        if not (model.model_name == "audsemthinker" or model.model_name == "audsemthinker-qa" or "Qwen2.5-Omni-7B" in model.model_name or model.model_name == "audsemthinker-grpo"):
            raise NotImplementedError("Batch size {} not implemented yet".format(batch_size))
        else:
            model_predictions = model.generate(input_data)
    
    if batch_size == -1:
        model_predictions = model.generate(input_data)
    
    elif batch_size == 1:
        model_predictions = []
        for inputs in tqdm(input_data, leave=False):
            outputs = model.generate(inputs)
            if isinstance(outputs, list):
                model_predictions.extend(outputs)
            else:
                model_predictions.append(outputs)
                
    return model_predictions


def main(
        dataset_name      : str  = None,
        model_name        : str  = None,
        batch_size        : int  = 1,     # it is now a dummy parameter
        overwrite         : bool = False,
        metrics           : str  = None,
        number_of_samples : int  = -1,
        only_generate_predictions : bool = False,
        ):

    logger.info("= = "*20)
    logger.info(f"Dataset name: {dataset_name}")
    logger.info(f"Model name: {model_name}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Overwrite: {overwrite}")
    logger.info(f"Metrics: {metrics}")
    logger.info(f"Number of samples: {number_of_samples}")
    logger.info(f"Only generate predictions: {only_generate_predictions}")
    logger.info("= = "*20)

    # If the final score log exists, skip the evaluation
    if not overwrite and os.path.exists(f'log/{model_name}/{dataset_name}_{metrics}_{number_of_samples}_score.json'):
        logger.info("Evaluation has been done before. Skip the evaluation.")
        logger.info("\n\n\n\n\n")
        return

    if model_name == 'WavLLM_fairseq':
        batch_size = -1
        logger.info("Batch size is set to -1 for WavLLM_fairseq model.")


    dataset = Dataset(dataset_name, number_of_samples)
    
    if model_name.count("/") > 2:
        model_name_path = "_".join(model_name.split("/")[-2:])
    else:
        model_name_path = model_name

    if overwrite or not os.path.exists(f'log/{model_name_path}/{dataset_name}_{number_of_samples}.json'):
        logger.info(f"Overwrite is enabled or the results are not found in log/{model_name_path}/{dataset_name}_{number_of_samples}.json. Try to infer with the model: {model_name}.")
    
        # Load model
        model = Model(model_name)

        # Specific current dataset name for evaluation
        model.dataset_name = dataset.dataset_name

        # Infer with model
        model_predictions           = do_model_prediction(dataset.input_data, model, batch_size=batch_size)
        data_with_model_predictions = dataset.dataset_processor.format_model_predictions(dataset.input_data, model_predictions)

            
        # Save the result with predictions
        os.makedirs(f'log/{model_name_path}', exist_ok=True)
        with open(f'log/{model_name_path}/{dataset_name}_{number_of_samples}.json', 'w') as f:
            json.dump(data_with_model_predictions, f, indent=4, ensure_ascii=False)
    else:
        logger.info(f"Loading results from log/{model_name_path}/{dataset_name}_{number_of_samples}.json")

    data_with_model_predictions = json.load(open(f'log/{model_name_path}/{dataset_name}_{number_of_samples}.json'))

    if overwrite or not os.path.exists(f'log/{model_name_path}/{dataset_name}_{metrics}_{number_of_samples}_score.json'):
        # Metric evaluation
        try:
            # Clear the cache to avoid memory leak
            logger.info("Clear the cache to avoid memory leak")
            del model
            torch.cuda.empty_cache()
        except: 
            pass
        
        if only_generate_predictions:
            return
        
        results = dataset.dataset_processor.compute_score(data_with_model_predictions, metrics=metrics)
        
        # Print the result with metrics
        logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')
        logger.info(f'Dataset name: {dataset_name.upper()}')
        logger.info(f'Model name: {model_name_path.upper()}')
        logger.info(json.dumps({metrics: results[metrics]}, indent=4, ensure_ascii=False))
        logger.info('=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =')

        # Save the scores
        with open(f'log/{model_name_path}/{dataset_name}_{metrics}_{number_of_samples}_score.json', 'w') as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
            
        logger.info(f"Completed evaluation for {model_name_path}")
        logger.info(f"Stored in log/{model_name_path}/{dataset_name}_{metrics}_{number_of_samples}_score.json")


if __name__ == "__main__":
    fire.Fire(main)

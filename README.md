## Mistral 7B Instruct fine tune and deploy on SageMaker

Mistral 7B is the open LLM from Mistral AI. The Mistral 7B Instruct model is a quick demonstration that the base model can be easily fine-tuned to achieve compelling performance. See more at https://mistral.ai/news/announcing-mistral-7b/

In this code sample, we will guide, step by step, fine tune the Mistral 7B instruct model
1. Locally with small sample of data 
2. With SageMaker on full dataset

Readers can start with `Local_Finetune_Mistral.ipynb`, which is the notebook for fine tuning the Mistral 7B model with QLoRA (Efficient Finetuning of Quantized LLMs). After fine tune several steps, you should be able to observe the difference between original model and fine tuned version, where you can use this notebook `Local_Compare_Finetune_Mistral.ipynb` to compare locally. 

The following script can be run locally with GPUs instance (we execute the script with single GPU on SageMaker Notebook g5 2xlarge instance, see more details on `test_training_script_local_gpu.ipynb`)

python ./training/train.py --dataset_path "./dataset/dolly.hf" --model_save_path "./results" --job_output_path "./results" --per_device_train_batch_size 1 --epochs 1


Additionally, readers have the option to run end-to-end fine-tuning code with SageMaker for a larger dataset using the `Finetune_Mistral_7B_on_Amazon_SageMaker.ipynb` notebook. 

Once the fine-tuning process is completed, the model can be deployed as a SageMaker endpoint using the `Deploy_Mistral_7B_on_Amazon_SageMaker_with_vLLM.ipynb` notebook.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the MIT-0 License. See the LICENSE file.


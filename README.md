# fine-tune-slm
This project is to fine tune sLM Llama 3.1 8B model using LoRA technique from Hugging Face for medical cancer-specific information extraction (IE) on EC2 instance

The current architecture design of the prject as follows: 
- The dataset is synthetic, curated and split into train and validation set in the synthetic-instruction-tuning-dataset/ folder. The training set has 4500 entries, and the validation set has 500 entries. One entry of the data is: {"instruction": "Extract all cancer-related entities from the text.", "input": "70-year-old man with widely metastatic cutaneous melanoma. PD-L1 was 5% on IHC and NGS reported TMB-high; BRAF testing was not performed prior to treatment. Given multiple symptomatic brain metastases he received combination immunotherapy with nivolumab plus ipilimumab and stereotactic radiosurgery to dominant intracranial lesions. Imaging after two cycles demonstrated some shrinking of index lesions but appearance of a new small lesion \u2014 overall assessment called a mixed response.", "output": {"cancer_type": "melanoma (cutaneous)", "stage": "IV", "gene_mutation": null, "biomarker": "PD-L1 5%; TMB-high", "treatment": "nivolumab and ipilimumab; stereotactic radiosurgery", "response": "mixed response", "metastasis_site": "brain"}}
- Retrieve the Llama 3.1 8B model via Hugging Face
- Running the fine tune scripts on EC2. Suggest type of EC2 instance I should use for this project. According to my research, I'm targeting g5 with VRAM greater or equal than 16 gB, cost-efficient and powerful enough

Planned CI/CD pipeline:
- Github actions (fully automated): build docker image and push to ECR
- Setup scripts (execute from local terminal): 
1. Start EC2 instance
1.1 Wait for status ok
2. Send SSM run command to deploy
2.1 CloudWatch to log the output of SSM command

- Fine tune scripts (execute from local terminal):
1. Run the fine tune job, save to S3. Suggest if I should mount EBS or use S3
2. Push to Hugging Face Hub
3. Stop EC2 instance from running

Note: I would like to send command from the local terminal and run everything remotely on EC2 via AWS SSM (no need to connect to EC2 instance via SSH or the need of .pem key). With a lot of setup and moving parts, I would want the command to be execute with precaution to errors, enable failsafe measures, etc. especially when running the fine tune script. 

- Keys need to be set up (there may be more than this, please suggest all the keys needed for set up): 
+ HF access token: to retrieve llama model
+ AWS access key + secret access key 
+ Docker hub token
+ ECR

Requirements:
- use AWS secrets manager to store keys, access tokens, etc. then use SSM parameter store to reference the key --> SSM and secrets manager integration for ease-of-use and privacy. Not using regular .env file because we would run command remotely on EC2 instance
- Use CloudWatch to log output and session from SSM run command
- use poetry for dependencies management (if needed for local)
- use loguru logger instead of printing statements (if needed for local)

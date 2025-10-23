# LoRa Fine-tuning Recipes

This example shows how to use Weco Agent to optimize hyper-parameters for LoRa fine-tuning on downstream tasks. 


## Setup
1.  git clone https://github.com/WecoAI/weco-cli.git 
2.  Download this "fine-tune-lora-weco" folder and ensure you put "fine-tune-lora-weco" folder under weco-cli/examples directory.
3.  Install the CLI and dependencies for the example:
    ```bash
    pip install -r requirements.txt
    ```

## Run Weco

Run the following command to start optimizing the model:

```bash
weco run --source train.py \
     --eval-command "python evaluate.py" \
     --metric loss \
     --goal min \
     --steps 20 \
     --model o4-mini \
     --additional-instructions instruction-0.txt --save-logs
```

### Explanation
instruction-0.txt -- No Hint

instruction-1.txt -- Hint A (background information)

instruction-2.txt -- Hint B (additional recipe)

### Dashboard results 

No hint: https://dashboard.weco.ai/runs/29b4d790-bbf8-46b5-b770-dce4b5395368 

Hint A: https://dashboard.weco.ai/runs/71e9c3a6-01f7-4d38-a523-3063027ed288 

Hint B: https://dashboard.weco.ai/runs/f3a5b3fc-20e9-4c79-a55a-0dfd13d7a646 


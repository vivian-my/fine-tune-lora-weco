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

No hint: https://weco.ai/share/7T3d53XPHguMGsGBH3ZUef7LPa3DQvjF  

Hint A: https://weco.ai/share/Eos4QYDEjBO8SEecvX_uRZvBHKxFkF20 

Hint B: https://weco.ai/share/klISso3_RQjtEw20tjPuRMTZsNLYWUFb


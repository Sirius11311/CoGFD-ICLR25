from pathlib import Path

# OPENAI_API_KEY = "sk-zk2e1e65b3d7b3551c49ebae268849667e467e771e9045b3"
# BASE_URL = "https://api.zhizengzeng.com/v1/"
OPENAI_API_KEY = "YOUR_KEY"
BASE_URL = "YOUR_BASE_URL"

PROJECT_ROOT = str(Path(__file__).parent.absolute())

PRETRAINED_MODEL_PATH = PROJECT_ROOT + "/pretrained_models" + "/stable-diffusion-v1-5"

HARMFUL_CMB_DIR = PROJECT_ROOT + "/HarmfulCmb"

OUTPUT_DIR = PROJECT_ROOT + "/output" + '/{concept_combination}'

LOGICGRAPH_DIR = OUTPUT_DIR + "/concept_logic_graph"

FINETUNE_MODEL_DIR = OUTPUT_DIR + "/model_finetune"

PREPARED_DATA_DIR = OUTPUT_DIR + "/data"

GENERATED_IMG_DIR = OUTPUT_DIR + "/img_generation"

EVAL_RESULTS_DIR = OUTPUT_DIR + "/eval_results"
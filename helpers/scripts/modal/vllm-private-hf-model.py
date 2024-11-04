import os
from pydantic import BaseModel
from typing import List

import modal


MODEL_DIR = "/model"
MODEL_NAME = "google/gemma-7b-it"


def download_model_to_image(model_dir, model_name):
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache

    os.makedirs(model_dir, exist_ok=True)

    snapshot_download(
        model_name,
        local_dir=model_dir,
        token=os.environ["HF_TOKEN"],
        ignore_patterns=["*.pt", "*.gguf"],  # Using safetensors
    )
    move_cache()


image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "vllm==0.4.0.post1",
        "torch==2.1.2",
        "transformers==4.39.3",
        "ray==2.10.0",
        "huggingface_hub==0.19.4",
        "hf-transfer==0.1.4",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .run_function(
        download_model_to_image,
        secrets=[
            modal.Secret.from_name(
                "my-huggingface-secret", required_keys=["HF_TOKEN"] # set your token in the platform
            )
        ],
        timeout=60 * 20,
        kwargs={"model_dir": MODEL_DIR, "model_name": MODEL_NAME},
    )
)

app = modal.App("vllm-gemma", image=image)

with image.imports():
    import vllm

GPU_CONFIG = modal.gpu.A100(size="80GB", count=1)


@app.cls(gpu=GPU_CONFIG, cpu=1, secrets=[modal.Secret.from_name("my-huggingface-secret")])
class Model:
    @modal.enter()
    def load(self):
        self.template = (
            "<start_of_turn>user\n{user}<end_of_turn>\n<start_of_turn>model\n"
        )

        # Load the model. Tip: Some models, like MPT, may require `trust_remote_code=true`.
        self.llm = vllm.LLM(
            MODEL_DIR,
            enforce_eager=True,  # skip graph capturing for faster cold starts
            tensor_parallel_size=GPU_CONFIG.count,
            gpu_memory_utilization=0.9,
        )

    @modal.method()
    def generate(self, user_questions):
        prompts = [self.template.format(user=q) for q in user_questions]

        sampling_params = vllm.SamplingParams(
            temperature=0.75,
            top_p=0.99,
            max_tokens=256,
            presence_penalty=1.15,
        )
        result = self.llm.generate(prompts, sampling_params)
        return result[0].outputs[0].text
    

    @modal.exit()
    def stop_engine(self):
        if GPU_CONFIG.count > 1:
            import ray

            ray.shutdown()

class TextsRequest(BaseModel):
    prompt: List[str]

# set up the web endpoint 
@app.function(image=image)
@modal.web_endpoint(method="POST", label="gemma-generate", docs=True)
def generate_web(request: TextsRequest):
    questions = request.prompt
    model = Model()
    answer = model.generate.remote(questions)
    return {"output": answer}

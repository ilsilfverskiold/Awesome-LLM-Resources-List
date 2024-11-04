import modal
from pydantic import BaseModel
from typing import List

app = modal.App("text-generation") # set an app name
model_repo_id = "ilsilfverskiold/tech-keywords-extractor" # decide on your model repo
cache_dir = "/cache"

image = (
    modal.Image.debian_slim()
    .pip_install(
        "huggingface-hub==0.16.4",
        "transformers",
        "torch"
    )
)

# have these loaded in modal rather than locally
with image.imports():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# set up the function to run for extracting keywords from texts (as per the model we're using)
# the snapshot download should download the model on build and cache it 
@app.cls(gpu="T4", cpu=0.5, memory=3000, image=image) # define cpu (cores), memory and/if gpu - default CPU request is 0.1 cores the soft CPU limit is 4.1 cores - default 128 MiB of memory
class TextExtraction:
    @modal.build()
    def download_model(self):
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_repo_id, cache_dir=cache_dir)

    @modal.enter()
    def load_model(self):
        self.tokenizer = AutoTokenizer.from_pretrained(model_repo_id, cache_dir=cache_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_repo_id, cache_dir=cache_dir)

    @modal.method()
    def extract_text(self, texts):
        inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=100)
        generated_texts = [self.tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

        return generated_texts

class TextsRequest(BaseModel):
    texts: List[str]

# set up the web endpoint 
@app.function(image=image)
@modal.web_endpoint(method="POST", label=f"{model_repo_id.split('/')[-1]}-web", docs=True)
def generate_web(request: TextsRequest):
    texts = request.texts
    extracted_texts = TextExtraction().extract_text.remote(texts)
    return {"extracted_texts": extracted_texts}
    # add potential error handling

from beam import endpoint, Image, Volume, env

if env.is_remote():
    import torch
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Model parameters
MODEL_NAME = "ilsilfverskiold/tech-keywords-extractor"
BEAM_VOLUME_PATH = "./cached_models"

def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=BEAM_VOLUME_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        cache_dir=BEAM_VOLUME_PATH,
        torch_dtype=torch.float16, 
    )
    return model, tokenizer

@endpoint(
    secrets=["HF_TOKEN"], # set secret at beam.cloud dashboard
    on_start=load_models,
    name="tech-keywords-extractor",
    cpu=1,
    memory="3Gi",
    keep_warm_seconds=60,
    image=Image(
        python_version="python3.9",
        python_packages=["torch", "transformers", "accelerate"],
    ),
    volumes=[
        Volume(
            name="cached_models",
            mount_path=BEAM_VOLUME_PATH,
        )
    ],
)

def extract_keywords(context, **inputs):
    model, tokenizer = context.on_start_value

    texts = inputs.get("texts", None)
    if not texts or not isinstance(texts, list):
        return {"error": "Please provide a list of texts for keyword extraction."}

    model_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=50,  # Adjusted for expected keyword lengths
            num_beams=5,  # Use beams for higher quality outputs
            no_repeat_ngram_size=2,  # Avoid repeating n-grams
        )
        output_text = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        return {"output": output_text}

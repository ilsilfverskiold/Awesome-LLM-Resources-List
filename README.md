# 🌟 Awesome LLM Resources

A Curated Collection of LLM resources. 💡✨

**🌐 Updated: 8th November 2024**

### 'Serverless' Hosting of Private/OS Models

| Platform/Tool                   | Rel. | Scale Down | OS 🔓 | GH | Start | GPU Machine | One-Click | Dev Exp. | Free-Tier |
| --------------------------------| -------- | -------------| -------------- | ----------|----------------|----------------|---------------|-------------------------|--------------|
| [Beam.Cloud](https://www.beam.cloud/) | 2021     | > 1 min      | [![GitHub Repo stars](https://img.shields.io/github/stars/beam-cloud/beta9?style=flat-square&color=purple)](https://github.com/beam-cloud/beta9) | [![GitHub followers](https://img.shields.io/github/followers/beam-cloud?style=flat-square&color=teal)](https://github.com/beam-cloud) | [Helpers](https://github.com/ilsilfverskiold/Awesome-LLM-Resources-List/tree/main/helpers/scripts/beam) | ❌ | ❌ | 👍 | 🆓 15h |
| [Baseten](https://www.baseten.com/) | 2019     | > 15 min     | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/basetenlabs?style=flat-square&color=teal)](https://github.com/basetenlabs) | [Guide](https://docs.baseten.co/deploy/guides/private-model) | ❌ | 🟡 | 👍 | $30 |
| [Modal](https://modal.com/)      | 2021     | < 1 min      | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/modal-labs?style=flat-square&color=teal)](https://github.com/modal-labs) | [Helpers](https://github.com/ilsilfverskiold/Awesome-LLM-Resources-List/tree/main/helpers/scripts/modal) | ❌ | ❌ | 👍 | $30/m |
| [HF Endpoints](https://ui.endpoints.huggingface.co/) | 2023 | > 15 min     | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/huggingface?style=flat-square&color=teal)](https://github.com/huggingface) | None Needed             | ❌ | ✅ | 😓 | ❌ |
| [Replicate](https://replicate.com/) | 2019     | < 1 min      | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/replicate?style=flat-square&color=teal)](https://github.com/replicate) | [Guide](https://replicate.com/docs/guides/push-a-transformers-model) | ✅ | 🟡 | 🤷 | ❌ |
| [Sagemaker (Serverless)](https://aws.amazon.com/sagemaker/) | 2017 | N/A          | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/aws?style=flat-square&color=teal)](https://github.com/aws/amazon-sagemaker-examples) | N/A                     | 🔵 | ❌ | ❌ | 300,000s |
| [Lambda w/ EFS (AWS)](https://aws.amazon.com/pm/lambda/) | 2014 | < 1 min     | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/awsdocs?style=flat-square&color=teal)](https://github.com/awsdocs/aws-lambda-developer-guide) | [Guide](https://aws.amazon.com/blogs/compute/hosting-hugging-face-models-on-aws-lambda/) | 🟡 | ❌ | ❌ | ✅ |
| [RunPod Serverless](https://www.runpod.io/serverless-gpu) | 2022 | > 30s       | 🔴            | [![GitHub followers](https://img.shields.io/github/followers/runpod?style=flat-square&color=teal)](https://github.com/runpod) | N/A                     | 🟡 | ❌ | 🤷 | ❌ |
| [BentoML](https://www.bentoml.com/) | 2019     | > 5 min      | [![GitHub Repo stars](https://img.shields.io/github/stars/bentoml/BentoML?style=flat-square&color=purple)](https://github.com/bentoml/BentoML) | [![GitHub followers](https://img.shields.io/github/followers/bentoml?style=flat-square&color=teal)](https://github.com/bentoml) | [Gallery](https://www.bentoml.com/gallery) | 🟡 | 🟡 | 👍 | 🆓 $10 |

It goes without saying that these platforms can usually do more than LLM serving**

### Access Off-the-Shelf OS Models (via API):

| Platform/Tool                           | Released | GitHub |
| --------------------------------------- | -------- | ----------- |
| [Together.ai](https://Together.ai)                                | N/A      | 🔴          |
| [Fireworks.ai](https://Fireworks.ai)                            | N/A      | [![GitHub followers](https://img.shields.io/github/followers/fw-ai?style=flat-square&color=teal)](https://github.com/fw-ai) |
| [Replicate](https://replicate.com/)     | 2019     | [![GitHub followers](https://img.shields.io/github/followers/replicate?style=flat-square&color=teal)](https://github.com/replicate) |
| [Groq](https://groq.com/)               | N/A      | [![GitHub followers](https://img.shields.io/github/followers/groq?style=flat-square&color=teal)](https://github.com/groq) |
| [DeepInfra](https://deepinfra.com/)     | N/A      | [![GitHub followers](https://img.shields.io/github/followers/deepinfra?style=flat-square&color=teal)](https://github.com/deepinfra) |
| [Bedrock](https://aws.amazon.com/bedrock) | N/A    | [![GitHub followers](https://img.shields.io/github/followers/aws-samples?style=flat-square&color=teal)](https://github.com/aws-samples/amazon-bedrock-workshop) |
| [Lepton](https://www.lepton.ai/)        | N/A      | [![GitHub followers](https://img.shields.io/github/followers/leptonai?style=flat-square&color=teal)](https://github.com/leptonai) |
| [Fal.ai](https://fal.ai/)               | N/A      | [![GitHub followers](https://img.shields.io/github/followers/fal-ai?style=flat-square&color=teal)](https://github.com/fal-ai) |
| [VertexAI](https://cloud.google.com/vertex-ai) | N/A | [![GitHub followers](https://img.shields.io/github/followers/GoogleCloudPlatform?style=flat-square&color=teal)](https://github.com/GoogleCloudPlatform/vertex-ai-samples) |

### Local Inference 

| Framework                                         | Browser Chat 🖥️ | Organization      | Open Source | GitHub |
|-----------------------------------------------|------------------|-------------------|-------------|--------|
| [Llama.cpp](https://github.com/ggerganov/llama.cpp) | ❌               | ggerganov        | [![GitHub Repo stars](https://img.shields.io/github/stars/ggerganov/llama.cpp?style=flat-square&color=purple)](https://github.com/ggerganov/llama.cpp) | [![GitHub followers](https://img.shields.io/github/followers/ggerganov?style=flat-square&color=teal)](https://github.com/ggerganov) |
| [Ollama](https://ollama.com/)                  | ❌               | Ollama           | [![GitHub Repo stars](https://img.shields.io/github/stars/ollama/ollama?style=flat-square&color=purple)](https://github.com/ollama/ollama) | [![GitHub followers](https://img.shields.io/github/followers/ollama?style=flat-square&color=teal)](https://github.com/ollama) |
| [gpt4all](https://www.nomic.ai/gpt4all)        | ✅               | Nomic.ai         | [![GitHub Repo stars](https://img.shields.io/github/stars/nomic-ai/gpt4all?style=flat-square&color=purple)](https://github.com/nomic-ai/gpt4all) | [![GitHub followers](https://img.shields.io/github/followers/nomic-ai?style=flat-square&color=teal)](https://github.com/nomic-ai) |
| [LMStudio](https://lmstudio.ai/)               | ✅               | LMStudio AI      | 🔴 | [![GitHub followers](https://img.shields.io/github/followers/lmstudio-ai?style=flat-square&color=teal)](https://github.com/lmstudio-ai) |
| [OpenLLM](https://github.com/bentoml/OpenLLM)  | ✅               | BentoML          | [![GitHub Repo stars](https://img.shields.io/github/stars/bentoml/OpenLLM?style=flat-square&color=purple)](https://github.com/bentoml/OpenLLM) | [![GitHub followers](https://img.shields.io/github/followers/bentoml?style=flat-square&color=teal)](https://github.com/bentoml) |

### LLM Serving Frameworks

| Framework                                                      | Open Source                                                                                                                | GitHub                                                                                   |
|----------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [vLLM](https://github.com/vllm-project/vllm)                   | [![GitHub Repo stars](https://img.shields.io/github/stars/vllm-project/vllm?style=flat-square&color=purple)](https://github.com/vllm-project/vllm)                | [![GitHub followers](https://img.shields.io/github/followers/vllm-project?style=flat-square&color=teal)](https://github.com/vllm-project) |
| [OpenLLM](https://github.com/bentoml/OpenLLM)                  | [![GitHub Repo stars](https://img.shields.io/github/stars/bentoml/OpenLLM?style=flat-square&color=purple)](https://github.com/bentoml/OpenLLM)                    | [![GitHub followers](https://img.shields.io/github/followers/bentoml?style=flat-square&color=teal)](https://github.com/bentoml) |
| [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference) | [![GitHub Repo stars](https://img.shields.io/github/stars/huggingface/text-generation-inference?style=flat-square&color=purple)](https://github.com/huggingface/text-generation-inference) | [![GitHub followers](https://img.shields.io/github/followers/huggingface?style=flat-square&color=teal)](https://github.com/huggingface) |
| [TensorRT LLM](https://docs.nvidia.com/tensorrt-llm/index.html) | [![GitHub Repo stars](https://img.shields.io/github/stars/NVIDIA/TensorRT-LLM?style=flat-square&color=purple)](https://github.com/NVIDIA/TensorRT-LLM)            | [![GitHub followers](https://img.shields.io/github/followers/NVIDIA?style=flat-square&color=teal)](https://github.com/NVIDIA) |
| [Ray Serve](https://docs.ray.io/en/latest/serve/index.html)    | [![GitHub Repo stars](https://img.shields.io/github/stars/ray-project/ray?style=flat-square&color=purple)](https://github.com/ray-project/ray)                    | [![GitHub followers](https://img.shields.io/github/followers/ray-project?style=flat-square&color=teal)](https://github.com/ray-project) |
| [LMDeploy](https://github.com/InternLM/lmdeploy)               | [![GitHub Repo stars](https://img.shields.io/github/stars/InternLM/lmdeploy?style=flat-square&color=purple)](https://github.com/InternLM/lmdeploy)                | [![GitHub followers](https://img.shields.io/github/followers/InternLM?style=flat-square&color=teal)](https://github.com/InternLM) |
| [Ollama](https://github.com/ollama/ollama)                     | [![GitHub Repo stars](https://img.shields.io/github/stars/ollama/ollama?style=flat-square&color=purple)](https://github.com/ollama/ollama)                        | [![GitHub followers](https://img.shields.io/github/followers/ollama?style=flat-square&color=teal)](https://github.com/ollama) |
| [MLC-LLM](https://github.com/mlc-ai/mlc-llm)                   | [![GitHub Repo stars](https://img.shields.io/github/stars/mlc-ai/mlc-llm?style=flat-square&color=purple)](https://github.com/mlc-ai/mlc-llm)                      | [![GitHub followers](https://img.shields.io/github/followers/mlc-ai?style=flat-square&color=teal)](https://github.com/mlc-ai) |


### Open-Source LLM Web Chat UIs

| Tool                                            | Organization       | Description                                                                                     | Open Source | GitHub |
|-------------------------------------------------|--------------------|-------------------------------------------------------------------------------------------------|-------------|--------|
| [Text Generation WebUI](https://github.com/oobabooga/text-generation-webui) | oobabooga          | A Gradio web UI for Large Language Models.       | [![GitHub Repo stars](https://img.shields.io/github/stars/oobabooga/text-generation-webui?style=flat-square&color=purple)](https://github.com/oobabooga/text-generation-webui) | [![GitHub followers](https://img.shields.io/github/followers/oobabooga?style=flat-square&color=teal)](https://github.com/oobabooga) |
| [Jan AI](https://jan.ai/)                       | Jan HQ            | Jan is an open source alternative to ChatGPT that runs 100% offline on your computer. Multiple engine support (llama.cpp, TensorRT-LLM)                           | [![GitHub Repo stars](https://img.shields.io/github/stars/janhq/jan?style=flat-square&color=purple)](https://github.com/janhq/jan) | [![GitHub followers](https://img.shields.io/github/followers/janhq?style=flat-square&color=teal)](https://github.com/janhq) |
| [AnythingLLM](https://github.com/Mintplex-Labs/anything-llm) | Mintplex Labs      | The all-in-one Desktop & Docker AI application with built-in RAG, AI agents, and more.              | [![GitHub Repo stars](https://img.shields.io/github/stars/Mintplex-Labs/anything-llm?style=flat-square&color=purple)](https://github.com/Mintplex-Labs/anything-llm) | [![GitHub followers](https://img.shields.io/github/followers/Mintplex-Labs?style=flat-square&color=teal)](https://github.com/Mintplex-Labs) |
| [Superagent](https://github.com/superagent-ai/superagent) | Superagent AI     | Superagent allows any developer to add powerful AI assistants to their applications. These assistants use large language models (LLM), retrieval augmented generation (RAG), and generative AI to help users.   | [![GitHub Repo stars](https://img.shields.io/github/stars/superagent-ai/superagent?style=flat-square&color=purple)](https://github.com/superagent-ai/superagent) | [![GitHub followers](https://img.shields.io/github/followers/superagent-ai?style=flat-square&color=teal)](https://github.com/superagent-ai) |
| [Bionic-GPT](https://bionic-gpt.com/)           | Bionic GPT        | Replacement for ChatGPT, offering the advantages of Generative AI while maintaining strict data confidentiality. | [![GitHub Repo stars](https://img.shields.io/github/stars/bionic-gpt/bionic-gpt?style=flat-square&color=purple)](https://github.com/bionic-gpt/bionic-gpt) | [![GitHub followers](https://img.shields.io/github/followers/bionic-gpt?style=flat-square&color=teal)](https://github.com/bionic-gpt) |
| [Open WebUI](https://github.com/open-webui/open-webui) | Open WebUI        | A user-friendly web interface for interacting with Large Language Models (LLMs). | [![GitHub Repo stars](https://img.shields.io/github/stars/open-webui/open-webui?style=flat-square&color=purple)](https://github.com/open-webui/open-webui) | [![GitHub followers](https://img.shields.io/github/followers/open-webui?style=flat-square&color=teal)](https://github.com/open-webui) |


### Rent GPUs (Fine-Tuning, Deploying, Training)

| Platform                                      | Templates                | Beginner Friendly | GitHub |
|-----------------------------------------------|--------------------------|-------------------|--------|
| [Brev.dev](https://www.brev.dev/)             | Fine-tuning              | ❌                | [![GitHub followers](https://img.shields.io/github/followers/brevdev?style=flat-square&color=teal)](https://github.com/brevdev) |
| [Modal](https://modal.com/)                   | Fine-tuning              | ❌                | [![GitHub followers](https://img.shields.io/github/followers/modal-labs?style=flat-square&color=teal)](https://github.com/modal-labs) |
| [Hyperbolic AI](https://hyperbolic.xyz/)      | None                     | ❌                | [![GitHub followers](https://img.shields.io/github/followers/HyperbolicLabs?style=flat-square&color=teal)](https://github.com/HyperbolicLabs) |
| [RunPod](https://www.runpod.io/)              | None                     | ❌                | [![GitHub followers](https://img.shields.io/github/followers/runpod?style=flat-square&color=teal)](https://github.com/runpod) |
| [Paperspace](https://www.paperspace.com/)     | Fine-tuning              | ✅                | [![GitHub followers](https://img.shields.io/github/followers/Paperspace?style=flat-square&color=teal)](https://github.com/Paperspace) |
| [Colab](https://colab.research.google.com/)   | Small models only        | ✅                | [![GitHub followers](https://img.shields.io/github/followers/googlecolab?style=flat-square&color=teal)](https://github.com/googlecolab) |


### Fine-Tuning with No-Code UI

| Tool                                           | Beginner Friendly | Open Source | GitHub |
|------------------------------------------------|-------------------|-------------|--------|
| [Together.ai](https://www.together.ai/products#fine-tuning) | ✅               | ❌          | N/A    |
| [Hugging Face AutoTrain](https://huggingface.co/autotrain) | ✅               | ❌          | [![GitHub Repo stars](https://img.shields.io/github/stars/huggingface/autotrain-advanced?style=flat-square&color=purple)](https://github.com/huggingface/autotrain-advanced) |
| [AutoML](https://github.com/mljar/automl-app)  | ❌               | ✅          | [![GitHub Repo stars](https://img.shields.io/github/stars/mljar/automl-app?style=flat-square&color=purple)](https://github.com/mljar/automl-app) |
| [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) | ❌               | ✅          | [![GitHub Repo stars](https://img.shields.io/github/stars/hiyouga/LLaMA-Factory?style=flat-square&color=purple)](https://github.com/hiyouga/LLaMA-Factory) |
| [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) | ✅               | ✅          | [![GitHub Repo stars](https://img.shields.io/github/stars/h2oai/h2o-llmstudio?style=flat-square&color=purple)](https://github.com/h2oai/h2o-llmstudio) |

### Fine-Tuning Frameworks

| Framework                                 | Open Source | GitHub |
|-------------------------------------------|-------------|--------|
| [Axolotl](https://axolotl.ai/)            | [![GitHub Repo stars](https://img.shields.io/github/stars/axolotl-ai-cloud/axolotl?style=flat-square&color=purple)](https://github.com/axolotl-ai-cloud/axolotl) | [![GitHub followers](https://img.shields.io/github/followers/axolotl-ai-cloud?style=flat-square&color=teal)](https://github.com/axolotl-ai-cloud) |
| [Unsloth](https://unsloth.ai/)            | [![GitHub Repo stars](https://img.shields.io/github/stars/unslothai/unsloth?style=flat-square&color=purple)](https://github.com/unslothai/unsloth) | [![GitHub followers](https://img.shields.io/github/followers/unslothai?style=flat-square&color=teal)](https://github.com/unslothai) |


### OS Agentic/AI Workflow

| Framework                                        | Open Source                                                                                                                | Beginner Friendly | Released | GitHub                                                                                   |
|--------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|-------------------|----------|------------------------------------------------------------------------------------------|
| [LangChain](https://www.langchain.com/)          | [![GitHub Repo stars](https://img.shields.io/github/stars/langchain-ai/langchain?style=flat-square&color=purple)](https://github.com/langchain-ai/langchain) | ✅               | 2022     | [![GitHub followers](https://img.shields.io/github/followers/langchain-ai?style=flat-square&color=teal)](https://github.com/langchain-ai) |
| [LangGraph](https://www.langchain.com/langgraph) | [![GitHub Repo stars](https://img.shields.io/github/stars/langchain-ai/langgraph?style=flat-square&color=purple)](https://github.com/langchain-ai/langgraph) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/langchain-ai?style=flat-square&color=teal)](https://github.com/langchain-ai) |
| [LlamaIndex](https://docs.llamaindex.ai/en/stable/use_cases/agents/) | [![GitHub Repo stars](https://img.shields.io/github/stars/run-llama/llama_index?style=flat-square&color=purple)](https://github.com/run-llama/llama_index) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/run-llama?style=flat-square&color=teal)](https://github.com/run-llama) |
| [Langroid](https://langroid.github.io/langroid/) | [![GitHub Repo stars](https://img.shields.io/github/stars/langroid/langroid?style=flat-square&color=purple)](https://github.com/langroid/langroid) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/langroid?style=flat-square&color=teal)](https://github.com/langroid) |
| [Flowise](https://flowiseai.com/)                | [![GitHub Repo stars](https://img.shields.io/github/stars/FlowiseAI/Flowise?style=flat-square&color=purple)](https://github.com/FlowiseAI/Flowise) | ✅               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/FlowiseAI?style=flat-square&color=teal)](https://github.com/FlowiseAI) |
| [Swarms](https://swarms.world/)                  | [![GitHub Repo stars](https://img.shields.io/github/stars/kyegomez/swarms?style=flat-square&color=purple)](https://github.com/kyegomez/swarms) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/kyegomez?style=flat-square&color=teal)](https://github.com/kyegomez) |
| [CrewAI](https://www.crewai.com/)               | [![GitHub Repo stars](https://img.shields.io/github/stars/crewaiinc/crewai?style=flat-square&color=purple)](https://github.com/crewaiinc/crewai) | ✅               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/crewaiinc?style=flat-square&color=teal)](https://github.com/crewaiinc) |
| [Autogen](https://microsoft.github.io/autogen/0.2/) | [![GitHub Repo stars](https://img.shields.io/github/stars/microsoft/autogen?style=flat-square&color=purple)](https://github.com/microsoft/autogen) | ✅               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/microsoft?style=flat-square&color=teal)](https://github.com/microsoft) |
| [AutoChain](https://autochain.forethought.ai/)   | [![GitHub Repo stars](https://img.shields.io/github/stars/Forethought-Technologies/AutoChain?style=flat-square&color=purple)](https://github.com/Forethought-Technologies/AutoChain) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/Forethought-Technologies?style=flat-square&color=teal)](https://github.com/Forethought-Technologies) |
| [SuperAGI](https://superagi.com/)               | [![GitHub Repo stars](https://img.shields.io/github/stars/TransformerOptimus/SuperAGI?style=flat-square&color=purple)](https://github.com/TransformerOptimus/SuperAGI) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/TransformerOptimus?style=flat-square&color=teal)](https://github.com/TransformerOptimus) |
| [AILegion](https://github.com/eumemic/ai-legion) | [![GitHub Repo stars](https://img.shields.io/github/stars/eumemic/ai-legion?style=flat-square&color=purple)](https://github.com/eumemic/ai-legion) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/eumemic?style=flat-square&color=teal)](https://github.com/eumemic) |
| [MemGPT (Letta)](https://www.letta.com/)         | [![GitHub Repo stars](https://img.shields.io/github/stars/cpacker/MemGPT?style=flat-square&color=purple)](https://github.com/cpacker/MemGPT) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/cpacker?style=flat-square&color=teal)](https://github.com/cpacker) |
| [uAgents](https://pypi.org/project/uagents/)     | [![GitHub Repo stars](https://img.shields.io/github/stars/fetchai/uAgents?style=flat-square&color=purple)](https://github.com/fetchai/uAgents) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/fetchai?style=flat-square&color=teal)](https://github.com/fetchai) |
| [Phidata](https://docs.phidata.com/agents)       | [![GitHub Repo stars](https://img.shields.io/github/stars/phidatahq/phidata?style=flat-square&color=purple)](https://github.com/phidatahq/phidata) | ❌               | 2023     | [![GitHub followers](https://img.shields.io/github/followers/phidatahq?style=flat-square&color=teal)](https://github.com/phidatahq) |
| [Dify](https://dify.ai/)                         | [![GitHub Repo stars](https://img.shields.io/github/stars/langgenius/dify?style=flat-square&color=purple)](https://github.com/langgenius/dify) | ✅               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/langgenius?style=flat-square&color=teal)](https://github.com/langgenius) |
| [TaskingAI](https://www.tasking.ai/)             | [![GitHub Repo stars](https://img.shields.io/github/stars/TaskingAI/TaskingAI?style=flat-square&color=purple)](https://github.com/TaskingAI/TaskingAI) | ✅               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/TaskingAI?style=flat-square&color=teal)](https://github.com/TaskingAI) |
| [Bee Agent Framework](https://i-am-bee.github.io/bee-agent-framework/#/) | [![GitHub Repo stars](https://img.shields.io/github/stars/i-am-bee/bee-agent-framework?style=flat-square&color=purple)](https://github.com/i-am-bee/bee-agent-framework) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/i-am-bee?style=flat-square&color=teal)](https://github.com/i-am-bee) |
| [Swarms](https://swarms.world/)                  | [![GitHub Repo stars](https://img.shields.io/github/stars/kyegomez/swarms?style=flat-square&color=purple)](https://github.com/kyegomez/swarms) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/kyegomez?style=flat-square&color=teal)](https://github.com/kyegomez) |
| [IoA](https://openbmb.github.io/IoA/)            | [![GitHub Repo stars](https://img.shields.io/github/stars/OpenBMB/IoA?style=flat-square&color=purple)](https://github.com/OpenBMB/IoA) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/OpenBMB?style=flat-square&color=teal)](https://github.com/OpenBMB) |
| [Atomic Agents](https://github.com/BrainBlend-AI/atomic-agents)     | [![GitHub Repo stars](https://img.shields.io/github/stars/BrainBlend-AI/atomic-agents?style=flat-square&color=purple)](https://github.com/BrainBlend-AI/atomic-agents) | ❌               | 2024     | [![GitHub followers](https://img.shields.io/github/followers/BrainBlend-AI?style=flat-square&color=teal)](https://github.com/BrainBlend-AI) |


### AI Agents

| Framework                                           | Organization                 | Open Source                                                                                                                | Released | GitHub                                                                                   |
|-----------------------------------------------------|------------------------------|-----------------------------------------------------------------------------------------------------------------------------|----------|------------------------------------------------------------------------------------------|
| [GPT Engineer](https://gptengineer.app/)            | GPT Engineer Org             | [![GitHub Repo stars](https://img.shields.io/github/stars/gpt-engineer-org/gpt-engineer?style=flat-square&color=purple)](https://github.com/gpt-engineer-org/gpt-engineer) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/gpt-engineer-org?style=flat-square&color=teal)](https://github.com/gpt-engineer-org) |
| [XAgent](https://github.com/OpenBMB/XAgent)         | OpenBMB                      | [![GitHub Repo stars](https://img.shields.io/github/stars/OpenBMB/XAgent?style=flat-square&color=purple)](https://github.com/OpenBMB/XAgent) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/OpenBMB?style=flat-square&color=teal)](https://github.com/OpenBMB) |
| [FinRobot](https://ai4finance.org/)                 | AI4Finance Foundation        | [![GitHub Repo stars](https://img.shields.io/github/stars/AI4Finance-Foundation/FinRobot?style=flat-square&color=purple)](https://github.com/AI4Finance-Foundation/FinRobot) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/AI4Finance-Foundation?style=flat-square&color=teal)](https://github.com/AI4Finance-Foundation) |
| [STORM](https://storm.genie.stanford.edu/)          | Stanford OVAL                | [![GitHub Repo stars](https://img.shields.io/github/stars/stanford-oval/storm?style=flat-square&color=purple)](https://github.com/stanford-oval/storm) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/stanford-oval?style=flat-square&color=teal)](https://github.com/stanford-oval) |
| [Multion](https://www.multion.ai/)                  | MULTI-ON                     | 🔴                                                                                                                          | N/A      | [![GitHub followers](https://img.shields.io/github/followers/MULTI-ON?style=flat-square&color=teal)](https://github.com/MULTI-ON) |
| [Minion](https://minion.ai/)                        | Minion AI                    | 🔴                                                                                                                          | N/A      | [![GitHub followers](https://img.shields.io/github/followers/minionai?style=flat-square&color=teal)](https://github.com/minionai) |



### Co-Pilots

| Framework                                  | Open Source                                                                                                                | GitHub                                                                                   |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [Aider](https://aider.chat/)               | [![GitHub Repo stars](https://img.shields.io/github/stars/Aider-AI/aider?style=flat-square&color=purple)](https://github.com/Aider-AI/aider) | [![GitHub followers](https://img.shields.io/github/followers/Aider-AI?style=flat-square&color=teal)](https://github.com/Aider-AI) |
| [Cursor](https://www.cursor.com/)          | [![GitHub Repo stars](https://img.shields.io/github/stars/getcursor/cursor?style=flat-square&color=purple)](https://github.com/getcursor/cursor) | [![GitHub followers](https://img.shields.io/github/followers/getcursor?style=flat-square&color=teal)](https://github.com/getcursor) |
| [Continue](https://docs.continue.dev/)     | [![GitHub Repo stars](https://img.shields.io/github/stars/continuedev/continue?style=flat-square&color=purple)](https://github.com/continuedev/continue) | [![GitHub followers](https://img.shields.io/github/followers/continuedev?style=flat-square&color=teal)](https://github.com/continuedev) |


### Voice API

| Framework                                  | Open Source                                                                                                                | GitHub                                                                                   |
|--------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------|
| [VAPI.ai](https://vapi.ai/)                | 🔴    | [![GitHub followers](https://img.shields.io/github/followers/vapiai?style=flat-square&color=teal)](https://github.com/vapiai) |
| [Bland.ai](https://www.bland.ai/)          | 🔴                                                                                                                          | N/A                                                                                      |
| [CallAnnie](https://callannie.ai/)         | 🔴                                                                                                                          | N/A                                                                                      |
| [RealtimeTTS](https://github.com/KoljaB/RealtimeTTS) | [![GitHub Repo stars](https://img.shields.io/github/stars/KoljaB/RealtimeTTS?style=flat-square&color=purple)](https://github.com/KoljaB/RealtimeTTS) | [![GitHub followers](https://img.shields.io/github/followers/KoljaB?style=flat-square&color=teal)](https://github.com/KoljaB) |
| [RealtimeSTT](https://github.com/KoljaB/RealtimeSTT) | [![GitHub Repo stars](https://img.shields.io/github/stars/KoljaB/RealtimeSTT?style=flat-square&color=purple)](https://github.com/KoljaB/RealtimeSTT) | [![GitHub followers](https://img.shields.io/github/followers/KoljaB?style=flat-square&color=teal)](https://github.com/KoljaB) |
| [Coqui TTS](https://github.com/coqui-ai/TTS) | [![GitHub Repo stars](https://img.shields.io/github/stars/coqui-ai/TTS?style=flat-square&color=purple)](https://github.com/coqui-ai/TTS) | [![GitHub followers](https://img.shields.io/github/followers/coqui-ai?style=flat-square&color=teal)](https://github.com/coqui-ai) |

### OS RAG Frameworks

| Framework                                           | Organization              | Open Source                                                                                                                | Released | GitHub                                                                                   |
|-----------------------------------------------------|---------------------------|-----------------------------------------------------------------------------------------------------------------------------|----------|------------------------------------------------------------------------------------------|
| [Haystack](https://haystack.deepset.ai/)            | deepset.ai                | [![GitHub Repo stars](https://img.shields.io/github/stars/deepset-ai/haystack?style=flat-square&color=purple)](https://github.com/deepset-ai/haystack) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/deepset-ai?style=flat-square&color=teal)](https://github.com/deepset-ai) |
| [RAGflow](https://ragflow.io/)                      | Infiniflow                | [![GitHub Repo stars](https://img.shields.io/github/stars/infiniflow/ragflow?style=flat-square&color=purple)](https://github.com/infiniflow/ragflow) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/infiniflow?style=flat-square&color=teal)](https://github.com/infiniflow) |
| [txtai](https://neuml.github.io/txtai/)             | Neuml                     | [![GitHub Repo stars](https://img.shields.io/github/stars/neuml/txtai?style=flat-square&color=purple)](https://github.com/neuml/txtai) | 2022     | [![GitHub followers](https://img.shields.io/github/followers/neuml?style=flat-square&color=teal)](https://github.com/neuml) |
| [LLM App](https://pathway.com/developers/templates) | Pathway                   | [![GitHub Repo stars](https://img.shields.io/github/stars/pathwaycom/llm-app?style=flat-square&color=purple)](https://github.com/pathwaycom/llm-app) | 2023     | [![GitHub followers](https://img.shields.io/github/followers/pathwaycom?style=flat-square&color=teal)](https://github.com/pathwaycom) |
| [Cognita](https://cognita.truefoundry.com/)         | Truefoundry               | [![GitHub Repo stars](https://img.shields.io/github/stars/truefoundry/cognita?style=flat-square&color=purple)](https://github.com/truefoundry/cognita) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/truefoundry?style=flat-square&color=teal)](https://github.com/truefoundry) |
| [R2R](https://r2r-docs.sciphi.ai/introduction)      | SciPhi AI                 | [![GitHub Repo stars](https://img.shields.io/github/stars/SciPhi-AI/R2R?style=flat-square&color=purple)](https://github.com/SciPhi-AI/R2R) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/SciPhi-AI?style=flat-square&color=teal)](https://github.com/SciPhi-AI) |
| [Raptor](https://arxiv.org/abs/2401.18059)          | Parth Sarthi              | [![GitHub Repo stars](https://img.shields.io/github/stars/parthsarthi03/raptor?style=flat-square&color=purple)](https://github.com/parthsarthi03/raptor) | 2024     | [![GitHub followers](https://img.shields.io/github/followers/parthsarthi03?style=flat-square&color=teal)](https://github.com/parthsarthi03) |

See [RAG_Techniques](https://github.com/NirDiamant/RAG_Techniques) if you get stuck (not always needed)


### Research Papers on Chain-of-Thought Prompting

| Publication Date | Title | 🔗 | Authors | Organization | Technique |
|------------------|-------|----|---------|--------------|-----------|
| January 28, 2022 | Chain-of-Thought Prompting Elicits Reasoning in Large Language Models | [🔗](https://arxiv.org/abs/2201.11903) | Jason Wei, et al. | DeepMind | CoT Prompting |
| March 21, 2022 | Self-Consistency Improves Chain of Thought Reasoning in Language Models | [🔗](https://arxiv.org/abs/2203.11171) | Xuezhi Wang et al. | DeepMind | CoT with Self-Consistency |
| May 21, 2022 | Least-to-Most Prompting Enables Complex Reasoning in Large Language Models | [🔗](https://arxiv.org/abs/2205.10625) | Denny Zhou et al. | DeepMind | Least-to-Most Prompting |
| May 21, 2022 | Large Language Models are Zero-Shot Reasoners | [🔗](https://arxiv.org/abs/2205.11916) | Takeshi Kojima, et al. | DeepMind | Zero-shot-CoT |
| October 6, 2022 | ReAct: Synergizing Reasoning and Acting in Language Models | [🔗](https://arxiv.org/abs/2210.03629) | Shunyu Yao et al. | Princeton University | ReAct |
| April 1, 2023 | Teaching Large Language Models to Self-Debug | [🔗](https://arxiv.org/abs/2304.05128) | Xiang Lisa Li, et al. | DeepMind, Stanford University | Self-Debugging |
| May 6, 2023 | Plan-and-Solve Prompting: Improving Zero-Shot Chain-of-Thought Reasoning by Large Language Models | [🔗](https://arxiv.org/abs/2305.04091) | Lei Wang et al. | The Chinese University of Hong Kong, SenseTime Research | Plan-and-Solve Prompting |
| May 23, 2023 | Let’s Verify Step by Step | [🔗](https://arxiv.org/pdf/2305.20050) | Anya Goyal, et al. | DeepMind | Verification for CoT |
| October 3, 2023 | Large Language Models Cannot Self-Correct Reasoning Yet | [🔗](https://arxiv.org/abs/2310.01798) | Qingxiu Dong, et al. | The Chinese University of Hong Kong, Huawei Noah's Ark Lab | Self-Correction in LLMs |
| November 2023 | Universal Self-Consistency for Large Language Model Generation | [🔗](https://arxiv.org/pdf/2311.17311) | Xinyun Chen, Renat Aksitov, Uri Alon, Jie Ren, Kefan Xiao, Pengcheng Yin, Sushant Prakash, Charles Sutton, Xuezhi Wang, Denny Zhou | DeepMind | Universal Self-Consistency |
| May 17, 2023 | Tree of Thoughts: Deliberate Problem Solving with Large Language Models | [🔗](https://export.arxiv.org/abs/2305.10601) | Shunyu Yao, et al. | Princeton University, DeepMind | Tree-of-Thought |
| February 15, 2024 | Chain-of-Thought Reasoning Without Prompting | [🔗](https://arxiv.org/abs/2402.10200) | Xuezhi Wang, Denny Zhou | DeepMind | Chain-of-Thought Decoding |
| March 21, 2024 | ChainLM: Empowering Large Language Models with Improved Chain-of-Thought Prompting | [🔗](https://arxiv.org/abs/2403.14238) | Xiaoxue Cheng et al. | Renmin University of China | CoTGenius |
| June 2024 | Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models | [🔗](https://arxiv.org/pdf/2310.04406) | Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, Yu-Xiong Wang |  | Language Agent Tree Search (LATS) |
| May 2024 | Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning | [🔗](https://arxiv.org/pdf/2405.00451) | Yuxi Xie, et al. | National University of Singapore, DeepMind | MCTS |
| September 18, 2024 | To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning | [🔗](https://arxiv.org/abs/2409.12183) | Zayne Sprague, et al. | The University of Texas at Austin, Johns Hopkins University, Princeton University | Meta-analysis of CoT |
| September 25, 2024 | Chain-of-Thoughtlessness? An Analysis of CoT in Planning | [🔗](https://arxiv.org/abs/2305.12147) | Kaya Stechly, et al. | Arizona State University | Analysis of CoT in Planning |
| October 18, 2024 | Supervised Chain of Thought | [🔗](https://arxiv.org/abs/2410.14198) | Xiang Zhang, Dujian Ding | University of British Columbia | Supervised Chain of Thought |
| October 24, 2024 | On examples: A Theoretical Understanding of Chain-of-Thought: Coherent Reasoning and Error-Aware Demonstration | [🔗](https://arxiv.org/abs/2410.16540) | Zhiqiang Hu, et al. | Amazon, Michigan State University | Theoretical Analysis of CoT |



### CoT Implementations

| Implementation | Link | Author | GitHub Stars | GitHub Followers |
|----------------|------|---------|--------------|------------------|
| CoT | [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub) | Franx Yao | [![Stars](https://img.shields.io/github/stars/FranxYao/chain-of-thought-hub?style=flat-square&color=purple)](https://github.com/FranxYao/chain-of-thought-hub) | [![Followers](https://img.shields.io/github/followers/FranxYao?style=flat-square&color=teal)](https://github.com/FranxYao) |
| CoT | [optillm](https://github.com/codelion/optillm) | Codelion | [![Stars](https://img.shields.io/github/stars/codelion/optillm?style=flat-square&color=purple)](https://github.com/codelion/optillm) | [![Followers](https://img.shields.io/github/followers/codelion?style=flat-square&color=teal)](https://github.com/codelion) |
| CoT | [auto-cot](https://github.com/amazon-science/auto-cot) | Amazon Science | [![Stars](https://img.shields.io/github/stars/amazon-science/auto-cot?style=flat-square&color=purple)](https://github.com/amazon-science/auto-cot) | [![Followers](https://img.shields.io/github/followers/amazon-science?style=flat-square&color=teal)](https://github.com/amazon-science) |
| CoT | [g1](https://github.com/bklieger-groq/g1) | BKlieger Groq | [![Stars](https://img.shields.io/github/stars/bklieger-groq/g1?style=flat-square&color=purple)](https://github.com/bklieger-groq/g1) | [![Followers](https://img.shields.io/github/followers/bklieger-groq?style=flat-square&color=teal)](https://github.com/bklieger-groq) |
| Decoding CoT | [optillm/cot_decoding.py](https://github.com/codelion/optillm/blob/main/optillm/cot_decoding.py) | Codelion | [![Stars](https://img.shields.io/github/stars/codelion/optillm?style=flat-square&color=purple)](https://github.com/codelion/optillm) | [![Followers](https://img.shields.io/github/followers/codelion?style=flat-square&color=teal)](https://github.com/codelion) |
| Tree of Thoughts | [tree-of-thought-llm](https://github.com/princeton-nlp/tree-of-thought-llm) | Princeton NLP | [![Stars](https://img.shields.io/github/stars/princeton-nlp/tree-of-thought-llm?style=flat-square&color=purple)](https://github.com/princeton-nlp/tree-of-thought-llm) | [![Followers](https://img.shields.io/github/followers/princeton-nlp?style=flat-square&color=teal)](https://github.com/princeton-nlp) |
| Tree of Thoughts | [tree-of-thoughts](https://github.com/kyegomez/tree-of-thoughts) | Kye Gomez | [![Stars](https://img.shields.io/github/stars/kyegomez/tree-of-thoughts?style=flat-square&color=purple)](https://github.com/kyegomez/tree-of-thoughts) | [![Followers](https://img.shields.io/github/followers/kyegomez?style=flat-square&color=teal)](https://github.com/kyegomez) |
| Tree of Thoughts | [saplings](https://github.com/shobrook/saplings) | Shobrook | [![Stars](https://img.shields.io/github/stars/shobrook/saplings?style=flat-square&color=purple)](https://github.com/shobrook/saplings) | [![Followers](https://img.shields.io/github/followers/shobrook?style=flat-square&color=teal)](https://github.com/shobrook) |
| MCTS | [optillm/mcts.py](https://github.com/codelion/optillm/blob/main/optillm/mcts.py) | Codelion | [![Stars](https://img.shields.io/github/stars/codelion/optillm?style=flat-square&color=purple)](https://github.com/codelion/optillm) | [![Followers](https://img.shields.io/github/followers/codelion?style=flat-square&color=teal)](https://github.com/codelion) |
| Graph of Thoughts | [graph-of-thoughts](https://github.com/spcl/graph-of-thoughts) | SPCL | [![Stars](https://img.shields.io/github/stars/spcl/graph-of-thoughts?style=flat-square&color=purple)](https://github.com/spcl/graph-of-thoughts) | [![Followers](https://img.shields.io/github/followers/spcl?style=flat-square&color=teal)](https://github.com/spcl) |
| Other | [CPO](https://github.com/sail-sg/CPO) | SAIL SG | [![Stars](https://img.shields.io/github/stars/sail-sg/CPO?style=flat-square&color=purple)](https://github.com/sail-sg/CPO) | [![Followers](https://img.shields.io/github/followers/sail-sg?style=flat-square&color=teal)](https://github.com/sail-sg) |
| Other | [Everything-of-Thoughts-XoT](https://github.com/microsoft/Everything-of-Thoughts-XoT) | Microsoft | [![Stars](https://img.shields.io/github/stars/microsoft/Everything-of-Thoughts-XoT?style=flat-square&color=purple)](https://github.com/microsoft/Everything-of-Thoughts-XoT) | [![Followers](https://img.shields.io/github/followers/microsoft?style=flat-square&color=teal)](https://github.com/microsoft) |

### CoT Fine-Tuned Models & Datasets

#### Models
| Model Name | Author | Size | Link |
|------------|--------|------|------|
| CoT-T5-3B | KAIST AI | 3B | [🔗](https://huggingface.co/kaist-ai/CoT-T5-3B) |
| CoT-T5-11B | KAIST AI | 11B | [🔗](https://huggingface.co/kaist-ai/CoT-T5-11B) |
| Llama-3.2V-11B-cot | Xkev | 11B | [🔗](https://huggingface.co/Xkev/Llama-3.2V-11B-cot) |
| Llama-3.1-8B-Instruct-Reasoner-1o1_v0.3 | Lyte | 8B | [🔗](https://huggingface.co/Lyte/Llama-3.1-8B-Instruct-Reasoner-1o1_v0.3) |


#### Datasets
| Dataset Name | Author | Data Size | Likes | Link |
|--------------|--------|-----------|-------|------|
| chain-of-thought-sharegpt | Isaiah Bjork | 7.14k rows | 🌟 8 | [🔗](https://huggingface.co/datasets/isaiahbjork/chain-of-thought-sharegpt) |
| CoT-Collection | KAIST AI | 1.84 million rows | 🌟 122 | [🔗](https://huggingface.co/datasets/kaist-ai/CoT-Collection?row=4) |
| Reasoner-1o1-v0.3-HQ | Lyte | 370 rows | 🌟 7 | [🔗](https://huggingface.co/datasets/Lyte/Reasoner-1o1-v0.3-HQ) |
| OpenLongCoT-Pretrain | qq8933 | 103k rows | 🌟 86 | [🔗](https://huggingface.co/datasets/qq8933/OpenLongCoT-Pretrain?row=0) |






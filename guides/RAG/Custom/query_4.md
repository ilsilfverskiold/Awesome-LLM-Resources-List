# Query 4

## Result 
(latency retrieval is ~4.9 seconds)

**Original Query:** give me innovations in RAG from the papers anything unconventional?

**LLM answer:**

Here are unconventional RAG innovations from recent papers:

- Retrieval on demand using model signals: Trigger search only when needed—e.g., when token probabilities drop (FLARE), when an external hallucination detector flags entities (DRAD), or when the model’s own internal states show high uncertainty (SeaKR). Some even trigger when answers across languages/models disagree (Rowen) [1]. Example: if the model hesitates on a date, it pauses and searches; if it invents a person’s affiliation, a detector forces a corrective lookup.

- Corrective/critic loops over retrieved context: CRAG judges retrieved docs as correct/incorrect/ambiguous and rewrites the query if needed; SIM-RAG trains a critic to decide if context is sufficient before proceeding; CoV-RAG scores truthfulness, reference, and answer correctness, then either finalizes or rewrites and retrieves again [1]. Example: if top docs are off-topic or partial, the system auto-reforms the query and re-searches.

- Agentic/self-triggered retrieval: Systems decompose queries and plan retrieval themselves (ReAct, SelfAsk, Searchain). SMR adds explicit actions—Refine (rewrite query using retrieved feedback), Rerank (ensure relevance), and Stop (avoid overthinking), reducing redundancy and drift [1][3]. Example: after pulling similar docs, the agent reranks to focus, then stops once evidence quality is “good enough.”

- Search as guided exploration: MCTS-RAG uses Monte Carlo Tree Search to explore retrieval paths; Search-O1 lets large reasoning models detect “salient uncertain” words to decide when to fetch and includes a reason-in-documents phase to strip noise before answering [1].

- Adaptive routing: Predict whether to retrieve at all and how many rounds based on query complexity, not a fixed pipeline [1].

- Train the generator to “think for RAG”: HIRAG instruction-tunes three skills—filtering, combining across paragraphs, and RAG-specific reasoning—via multi-level, think-before-answer chains [2].

- Efficiency-first RL: FrugalRAG’s two-stage RL learns to retrieve enough to answer while minimizing the number of searches, unlike methods that encourage more searches [3].


**Citations:**

[1] doc::53b521e646b84289b46e648c66dde56a — Test-time Corpus Feedback: From Retrieval to RAG

[2] doc::b0610cc6134b401db0ea68a77096e883 — HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation

[3] doc::ece5670065a746a59112bfbd2cd1053b — FrugalRAG: Learning to retrieve and reason for multi-hop QA


--------------------------------------------------------------------------------------------------------------------------------------

# RAG Pipeline

**Latency Breakdown:**

- Query optimizer: 2.6s (boots up Qdrant client in parallel)
- Retrieve: 0.87s
- Rerank: 0.59s
- Context expansion: 0.8s
- LLM answer: 10–15s

## Query Optimizer (2.6s)

**Tech:** GPT-4o-mini, structured responses

Generated queries:
- hybrid: unconventional innovations in retrieval-augmented generation

## Retrieve (0.87s)

**Tech:** Qdrant, LlamaIndex

**Results:**

Query 1 (hybrid) top 20 for query: unconventional innovations in retrieval-augmented generation

[1] score=0.5000 doc=docs_ingestor/docs/arxiv/2507.07634.pdf chunk=S5::C03::251104135156
  text: 2 Related Work retrieval and generation iteratively, such that a model's response is incorporated in the reasoning trace. However, both IRCoT [Trivedi et al., 2022a] and Iter-RetGen [Shao et al., 2023] rely on a fixed or predefined number of retrieval loops at inference, offering limited control over latency.

[2] score=0.5000 doc=docs_ingestor/docs/arxiv/2508.09535.pdf chunk=S7::C03::251104160635
  text: 5. Methodology: Script Production and Narrative Construction 5.1 Thematic Exploration and Query Generation The first stage involves generating thematic ideas that introduce ironic, absurd, or paradoxical perspectives on the selected central theme. Following the principles of lateral thinking (De Bono and Zimbalist 1970), which emphasize creative exploration beyond traditional or linear reasoning, this stage deliberately seeks unconventional viewpoints to provoke novel and unexpected interpretations. From these thematic ideas, the system generates a set of related phrases to be used as queries for exploring the semantic database. This ensures diverse entry points for semantic retrieval, broadening the scope of archival material available for satirical reinterpretation.

[3] score=0.4938 doc=docs_ingestor/docs/arxiv/2507.07634.pdf chunk=S5::C02::251104135156
  text: 2 Related Work Prompting-based RAG approaches. With the recent advancements in the capabilities of large API-based LMs, some works explored prompting to call external search/retrievers at inference. Toolformer[Schick et al., 2023] uses a self-supervised objective to train an external model that decides to call tools (like Bing and Google search engines). ReAct [Yao et al., 2023] is another powerful prompting technique that allows the model to structure its outputs as thoughts, actions and observations, yielding significant improvements in the ability of LLMs to interact with external environments. Trivedi et al. [2022a] proposed IRCoT, another prompting strategy that alternates between chain-of-thought [Wei et al., 2022] steps and gathering evidence through retrievals. By using the intermediate traces, the IRCoT is able to decide what to retrieve by issuing the right search queries. Iter-RetGen [Shao et al., 2023] improves evidence gathering in multi-hop scenarios by combining

[4] score=0.4721 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C04::251104164223
  text: 5 Generation-time feedback 5.2 Retrieval-on-Demand via Feedback Signals LLM's internal knowledge. However, the judgment is solely based on LLM, and without context, they try to be overconfident (Xiong et al., 2024). FLARE (Jiang et al., 2023) retrieves the documents only if the token probability is below a predefined threshold and uses the last generated sentence as a query for retrieval (excluding the uncertain tokens) and generates the response until the next uncertain token or completion is done. However, these uncertain tokens are not equally important to trigger a retrieval round. Based on this, DRAD (Su et al., 2024a) uses an external module for hallucination detection on entities in the generated answer; if the answer contains hallucination, the retrieval is triggered. The last generated sentence (without a hallucinated entity) is used as a query for retrieval. However, the choice of the new query for retrieval relies on heuristic strategies. Since the model's information needs may extend beyond the last sentence or C

[5] score=0.4266 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S4::C03::251104134926
  text: 2 Related Work Upgrading of retrieval modules. From the perspective of retrieval methods, some studies have enhanced the quality of context by employing multistage retrieval reasoning (Asai et al., 2023b) (Gan et al., 2024), while others have designed adaptive retrieval modules that allow models to adjust retrieval behavior according to different tasks (Jeong et al., 2024). In terms of question understanding, some studies have improved search queries by rewriting, decomposing, and disambiguating (Chan et al., 2024). After retrieving articles, incorporating a ranking module can significantly enhance the final generation outcome (Glass et al., 2022)(Ram et al., 2023). RankRAG effectively integrates the ranking module with the generation module (Yu et al., 2024). These approaches have effectively improved the quality of retrieved articles in RAG systems. However, there is no such thing as a perfect context, and the generative model needs to be capable of handling contexts in various situations.

[6] score=0.4257 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C01::251104164223
  text: 5 Generation-time feedback RAG systems face two fundamental challenges: determining when to retrieve external knowledge, since not all queries benefit from it, and how to retrieve relevant content effectively (Su et al., 2024b). Classical RAG pipelines rigidly follow a fixed sequence of retrieval, optionally ranking, followed by generation, limiting their ability to adapt to the context or task. To address these limitations, recent work has introduced adaptive RAG , where the retrieval strategy is dynamically adjusted according to the query, the model feedback, or the complexity of the task. We categorize this emerging line of work into three main classes.

[7] score=0.4246 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S6::C08::251104164223
  text: 3 Query-level feedback 3.2 Generative Relevance Feedback (GRF) Feedback from Generated Answers. Beyond generating expansions, some methods use LLM-generated answers as implicit feedback. Generation-Augmented Retrieval (GAR) (Mao et al., 2021) generates answer-like contexts (titles, passages, summaries) using a model like BART (Lewis et al., 2020a), which are then concatenated to the query. However, this introduces risks of hallucination and irrelevant additions. To refine this idea, RRR (Arora et al., 2023) iteratively updates the query based on retrieval performance, using a feedback loop constrained by a document budget. LameR (Shen et al., 2024) first generates multiple answers, augments them with the query, and performs a second retrieval pass-effectively building a feedback loop from generation to retrieval. InteR (Feng et al., 2024) and Iter-RetGen (Shao et al., 2023) perform tighter integration between RAG and GAR by alternating between generation and retrieval for iterative refinement.

[8] score=0.3965 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C11::251104164223
  text: 5 Generation-time feedback 5.3 Self-Triggered Retrieval via Reasoning MCTS-RAG (Hu et al., 2025) combines Monte Carlo Tree Search (MCTS) with RetrievalAugmented Generation (RAG) to improve reasoning and retrieval in language models. It guides the search for relevant information using MCTS to explore promising retrieval paths, enhancing answer accuracy. However, it is computationally expensive due to the iterative tree search process and may struggle with highly noisy or irrelevant documents. Search-O1 (Li et al., 2025a) proposes an agentic search workflow for reasoning augmented retrieval by letting the Large Reasoning Models (LRMs) like O1 decide autonomously when to retrieve by detecting the presence of salient uncertain words in their output. Additionally, they augment the workflow with a reason-in-documents step, where LRMs analyze the documents in depth to remove noise and reduce redundancy before employing them to generate the final answer.

[9] score=0.3539 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S5::C03::251104164223
  text: 2 Preliminaries 2.2 Retrieval-Augmented Generation RAG introduces an explicit retrieval step: for a query q , a retriever selects a set of topk documents { d 1 , . . . , d k } from an external corpus. A generator G then conditions on both q and the retrieved context to produce the output a = G ( q, d 1 , . . . , d k ) where G is typically an encoderdecoder or decoder-only LLM fine-tuned to integrate retrieved evidence into its generation process.

[10] score=0.3186 doc=docs_ingestor/docs/arxiv/2508.09874.pdf chunk=S8::C01::251104161352
  text: 7. Related Work Retrieval-Augmented Generation Retrieval-Augmented Generation (RAG) enhances language models by incorporating knowledge from external sources, with retrieval granularity ranging from documents [Chen et al., 2017] to passages [Guu et al., 2020, Izacard et al., 2023, Lewis et al., 2020] to tokens [He et al., 2021, Khandelwal et al., 2019, Min et al., 2022, Yogatama et al., 2021]. Tokenlevel retrieval achieves superior performance for rare patterns and out-of-domain scenarios but introduces substantial computation overhead during inference. While non-differentiable retrieval mechanisms prevent end-to-end optimization and memory token approaches [Chevalier et al., 2023] enable differentiable access but are limited to local contexts, Memory Decoder provides both differentiable optimization and full-dataset knowledge access without expensive retrieval operations or model-specific datastores.

[11] score=0.2864 doc=docs_ingestor/docs/arxiv/2508.04442.pdf chunk=S4::C03::251104155058
  text: 2 Related Work 2.2 Grounding Generative Models: The Rise of Retrieval-Augmented Generation (RAG) Retrieval-Augmented Generation (RAG) has emerged as the foremost technique for addressing the 'grounding problem' of LLMs.[9] First proposed by Lewis et al. (2020), the RAG architecture combines the strengths of a parametric memory (the pre-trained LLM) with a non-parametric memory (an external, retrievable knowledge source).[10] In this hybrid model, the LLM does not rely solely on its internal, static knowledge. Instead, given a prompt, it first retrieves relevant documents from the external source and then uses this retrieved context to inform and constrain its generated output. This approach has been shown to significantly reduce hallucinations and improve the factual accuracy of LLM responses. In educational technology, RAG is particularly valuable. It allows for the creation of AI systems that can provide factually consistent, curriculum-aligned answers and content.[9] Furthermore, because the knowledge base is external, it

[12] score=0.2543 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S3::C02::251104134926
  text: 1 Introduction by incorporating an information retrieval component. While LLMs often use in-context learning (Gao et al., 2024) for generation, practical issues such as low-quality or poorly ranked retrieved documents can hinder RAG's effectiveness. These challenges emphasize the need for instruction-tuning tailored to RAG tasks. Fine-tuning generative models specifically for RAG improves their ability to integrate retrieved information (Zhang et al., 2024) (Yu et al., 2024), resulting in more accurate and contextually relevant responses compared to generalpurpose models.

[13] score=0.2005 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S2::C01::251104134926
  text: Abstract Retrieval-augmented generation (RAG) has become a fundamental paradigm for addressing the challenges faced by large language models in handling real-time information and domainspecific problems. Traditional RAG systems primarily rely on the in-context learning (ICL) capabilities of the large language model itself. Still, in-depth research on the specific capabilities needed by the RAG generation model is lacking, leading to challenges with inconsistent document quality and retrieval system imperfections. Even the limited studies that fine-tune RAG generative models often lack a granular focus on RAG task or a deeper utilization of chain-of-thought processes . To address this, we propose that RAG models should possess three progressively hierarchical abilities (1) Filtering: the ability to select relevant information; (2) Combination: the ability to combine semantic information across paragraphs; and (3) RAGspecific reasoning: the ability to further process external knowledge using internal knowledge. Thus, we introd

[14] score=0.1215 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C02::251104164223
  text: 5 Generation-time feedback 5.1 Rule-Based and Discriminative Approaches In-Context RALM (Retrieval-Augmented Language Model) (Ram et al., 2023) proposes retrieving relevant context documents during inference at fixed intervals (every s tokens, known as the retrieval stride), using the last l tokens of the input as the retrieval query. In a similar spirit, IRCoT (Interleaving Retrieval in a CoT) (Trivedi et al., 2023) dynamically retrieves documents if the CoT (Wei et al., 2022) step has not provided the answer. At first, it uses the original question to retrieve the context and then uses the last generated CoT sentence as a query for subsequent retrieval. However, both of these methods retrieve the context regardless of whether the LLM needs external context or not. Hence, the unnecessary retrieval steps add additional latency cost during answer generation. Also, the noisy retrieved context can lead to a wrong answer. CtRLA (Huanshuo et al., 2025) devises a latent space probing-based approach for making decisions regarding r

[15] score=0.1143 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C06::251104164223
  text: 5 Generation-time feedback 5.2 Retrieval-on-Demand via Feedback Signals However, these approaches consider all queries equally complex and might end up with noisy context retrieval and hence a wrong answer. Adaptive RAG (Jeong et al., 2024) uses a query routing mechanism that predicts whether the query needs retrieval or not. Further, it also decides on the number of retrieval rounds based on query complexity. However, it assumes that the retrieved context is relevant to the query without assessing its relevancy or sufficiency. Towards the relevancy, CRAG

[16] score=0.1074 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C03::251104164223
  text: 5 Generation-time feedback 5.2 Retrieval-on-Demand via Feedback Signals The feedback signals can come from different sources, including the answer uncertainty, the model's internal states, or context faithfulness and sufficiency. SKR (Wang et al., 2023b) asks LLM itself if additional information is needed to answer the query. If yes, then the retrieval round is triggered; otherwise, the answer is generated from the

[17] score=0.1026 doc=docs_ingestor/docs/arxiv/2507.07634.pdf chunk=S5::C05::251104135156
  text: 2 Related Work Traditional RAG approaches. Early work in grounding generation with real world documents focused on end-to-end differentiable encoder-decoder pipeline REALM [Guu et al., 2020], which augments Masked-Language Modeling (MLM) with a latent retriever model, backpropagating through retrieval to learn both retriever and generator jointly. However, this approach incurs significant computational cost and has only been shown to work with relatively smaller models like T5 [Raffel et al., 2020]. Building on this, Lewis et al. [2020] proposed a general finetuning strategy, RAG-Token which demonstrated that join-training outperforms fixed dense retrieval and BM25.

[18] score=0.0990 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S14::C02::251104164223
  text: Key Insights Learned or Agentic Methods. The Agentic models go beyond prompt instructions and use search/retrieval as a tool. These models are trained to trigger this tool during answer generation. The training process mainly focuses on giving rewards for correct tool calls and context usage. In addition, similar to RAG methods, the retrieved documents are used as context to generate intermediate answers or the final answer. The search tool might have access to a local database or a web search engine to retrieve up-to-date knowledge.

[19] score=0.0832 doc=docs_ingestor/docs/arxiv/2508.10677.pdf chunk=S5::C02::251104162117
  text: B. Building the Knowledge Base (a) RAG Retrieval component Retreived VT contexts VIRUSTOTAL Chunking Embed documents Embedding Model Retreived IOCParser API Crowdstrike CTI Reports Chunks of text Alert contexts YROWDSTRIKE Retreived CTI LLM Augmented Alert contexts Sccurity Information Event Management SIEM Vector DataBase Standard Retrieval NLP Retrieval (Similarity) Retreived VT contexts Prompt template Retreived Generate Crowdstrike Alert Incident Response Plan contexts concatenated contexts Pre-trained LLM incident response team Retreived CTI contexts Augmented Generation (b) RAG Augmented generation component Fig. 1: Proposed Retrieval-Augmented Generation RAG Incident Response Architecture

[20] score=0.0674 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S5::C02::251104164223
  text: 2 Preliminaries 2.2 Retrieval-Augmented Generation Retrieval-Augmented Generation (RAG) (Lewis et al., 2020b) is a hybrid paradigm that enhances the generative capabilities of large language models (LLMs) by incorporating non-parametric external knowledge during inference. This design mitigates well-documented limitations of standalone LLMs, including hallucinations, confident but incorrect outputs, and inability to reflect up-to-date or domain-specific information due to static pretraining (Hurst et al., 2024; Grattafiori et al., 2024; Yang et al., 2025a).

## Rerank (0.74s)

**Tech:** Cohere API

**Results**

Rerank summary:
- strategy=cohere
- model=rerank-english-v3.0
- candidates=20
- eligible_above_threshold=20
- kept=6 (threshold=0)

Reranked Relevant (6/20 kept ≥ 0) top 6 for query: give me innovations in RAG from the papers anything unconventional?

[1] score=0.0032 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S5::C02::251104164223
  text: 2 Preliminaries 2.2 Retrieval-Augmented Generation Retrieval-Augmented Generation (RAG) (Lewis et al., 2020b) is a hybrid paradigm that enhances the generative capabilities of large language models (LLMs) by incorporating non-parametric external knowledge during inference. This design mitigates well-documented limitations of standalone LLMs, including hallucinations, confident but incorrect outputs, and inability to reflect up-to-date or domain-specific information due to static pretraining (Hurst et al., 2024; Grattafiori et al., 2024; Yang et al., 2025a).

[2] score=0.0021 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S2::C01::251104134926
  text: Abstract Retrieval-augmented generation (RAG) has become a fundamental paradigm for addressing the challenges faced by large language models in handling real-time information and domainspecific problems. Traditional RAG systems primarily rely on the in-context learning (ICL) capabilities of the large language model itself. Still, in-depth research on the specific capabilities needed by the RAG generation model is lacking, leading to challenges with inconsistent document quality and retrieval system imperfections. Even the limited studies that fine-tune RAG generative models often lack a granular focus on RAG task or a deeper utilization of chain-of-thought processes . To address this, we propose that RAG models should possess three progressively hierarchical abilities (1) Filtering: the ability to select relevant information; (2) Combination: the ability to combine semantic information across paragraphs; and (3) RAGspecific reasoning: the ability to further process external knowledge using internal knowledge. Thus, we introd

[3] score=0.0006 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C11::251104164223
  text: 5 Generation-time feedback 5.3 Self-Triggered Retrieval via Reasoning MCTS-RAG (Hu et al., 2025) combines Monte Carlo Tree Search (MCTS) with RetrievalAugmented Generation (RAG) to improve reasoning and retrieval in language models. It guides the search for relevant information using MCTS to explore promising retrieval paths, enhancing answer accuracy. However, it is computationally expensive due to the iterative tree search process and may struggle with highly noisy or irrelevant documents. Search-O1 (Li et al., 2025a) proposes an agentic search workflow for reasoning augmented retrieval by letting the Large Reasoning Models (LRMs) like O1 decide autonomously when to retrieve by detecting the presence of salient uncertain words in their output. Additionally, they augment the workflow with a reason-in-documents step, where LRMs analyze the documents in depth to remove noise and reduce redundancy before employing them to generate the final answer.

[4] score=0.0006 doc=docs_ingestor/docs/arxiv/2507.07634.pdf chunk=S5::C02::251104135156
  text: 2 Related Work Prompting-based RAG approaches. With the recent advancements in the capabilities of large API-based LMs, some works explored prompting to call external search/retrievers at inference. Toolformer[Schick et al., 2023] uses a self-supervised objective to train an external model that decides to call tools (like Bing and Google search engines). ReAct [Yao et al., 2023] is another powerful prompting technique that allows the model to structure its outputs as thoughts, actions and observations, yielding significant improvements in the ability of LLMs to interact with external environments. Trivedi et al. [2022a] proposed IRCoT, another prompting strategy that alternates between chain-of-thought [Wei et al., 2022] steps and gathering evidence through retrievals. By using the intermediate traces, the IRCoT is able to decide what to retrieve by issuing the right search queries. Iter-RetGen [Shao et al., 2023] improves evidence gathering in multi-hop scenarios by combining

[5] score=0.0005 doc=docs_ingestor/docs/arxiv/2507.07634.pdf chunk=S5::C05::251104135156
  text: 2 Related Work Traditional RAG approaches. Early work in grounding generation with real world documents focused on end-to-end differentiable encoder-decoder pipeline REALM [Guu et al., 2020], which augments Masked-Language Modeling (MLM) with a latent retriever model, backpropagating through retrieval to learn both retriever and generator jointly. However, this approach incurs significant computational cost and has only been shown to work with relatively smaller models like T5 [Raffel et al., 2020]. Building on this, Lewis et al. [2020] proposed a general finetuning strategy, RAG-Token which demonstrated that join-training outperforms fixed dense retrieval and BM25.

[6] score=0.0004 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C01::251104164223
  text: 5 Generation-time feedback RAG systems face two fundamental challenges: determining when to retrieve external knowledge, since not all queries benefit from it, and how to retrieve relevant content effectively (Su et al., 2024b). Classical RAG pipelines rigidly follow a fixed sequence of retrieval, optionally ranking, followed by generation, limiting their ability to adapt to the context or task. To address these limitations, recent work has introduced adaptive RAG , where the retrieval strategy is dynamically adjusted according to the query, the model feedback, or the complexity of the task. We categorize this emerging line of work into three main classes.

  
## Context Expansion (0.8s)

**Tech:** Redis (to fetch neighbors and additional informtion)

**Results:**

### Document #1 — Test-time Corpus Feedback: From Retrieval to RAG
- `doc_id`: `doc::53b521e646b84289b46e648c66dde56a`
- **Last Updated:** 2025-11-04T16:42:23.358191+00:00
- **Context:** This document surveys advancements in retrieval-augmented NLP systems.
- **Content fetched inside document:**
```text
[start on page 2]
    2 Preliminaries
    2.1 Retrieval System
    The core objective of a retrieval system is to identify and rank a subset of documents ( d 1 , d 2 , ..., d k ) from a large corpus C based on their estimated relevance to a query q . Classical retrieval approaches, such as BM25 (Robertson et al., 1995), rely on exact term matching and produce sparse relevance scores. In contrast, dense retrieval methods employ neural encoders to project queries and documents into a shared embedding space, enabling semantic similarity matching (Karpukhin et al., 2020). Since first-stage retrievers often produce noisy candidates, modern pipelines incorporate a second-stage re-ranking step using more expressive models. This includes LLM-based rankers (Pradeep et al., 2023b;
    Ma et al., 2024; Sun et al., 2023) and reasoningaugmented models such as ReasonIR (Shao et al., 2025), Rank-1 (Weller et al., 2025), and RankR1 (Zhuang et al., 2025), which refine the initial rankings by modeling deeper interactions between the query and candidate documents.

    ---------------------- this was the passage that we matched to the query -------------
    2.2 Retrieval-Augmented Generation
    Retrieval-Augmented Generation (RAG) (Lewis et al., 2020b) is a hybrid paradigm that enhances the generative capabilities of large language models (LLMs) by incorporating non-parametric external knowledge during inference. This design mitigates well-documented limitations of standalone LLMs, including hallucinations, confident but incorrect outputs, and inability to reflect up-to-date or domain-specific information due to static pretraining (Hurst et al., 2024; Grattafiori et al., 2024; Yang et al., 2025a).
    --------------------------------------------------------------------------------------

    RAG introduces an explicit retrieval step: for a query q , a retriever selects a set of topk documents { d 1 , . . . , d k } from an external corpus. A generator G then conditions on both q and the retrieved context to produce the output a = G ( q, d 1 , . . . , d k ) where G is typically an encoderdecoder or decoder-only LLM fine-tuned to integrate retrieved evidence into its generation process.

    2.3 Challenges in RAG
    A central challenge in RAG is that generation quality is tightly coupled with retrieval quality. If relevant (i.e., answer-containing) documents are not retrieved, or if irrelevant context is included, the generator is prone to producing incorrect or misleading outputs (Cuconasu et al., 2024, 2025; Liu et al., 2024).

    Consequently, improving the topk retrieval results is crucial. This can be viewed both as a selection problem (how to retrieve the most relevant documents) and a filtering problem (how to suppress distracting or noisy context). To this end, several methods have been proposed that incorporate various forms of feedback , ranging from simple lexical overlap to more sophisticated agentic or reasoning-based signals, to guide and refine the retrieval process.
    In this survey, we systematically categorize these feedback mechanisms and analyze their effectiveness across different components of the RAG pipeline. We focus on how feedback is acquired, represented, and integrated into retrieval, with the
    Figure 1: Illustration of feedback signals across the RAG pipeline. Feedback can modify the query (e.g., rewriting), the retrieved pool (e.g., ranker-based expansion), or the generation loop (e.g., retrieval triggers based on uncertainty).
    Retrieval Reranker results Generation Query Encoder representation new results Ranker Relevance Dynamic Understanding Generated Output Feedback levels Query Level Retrieval Level Generation Time Query Understanding Ranker Relevance Dynamic Retrieval - Query Expansion - Query Rewriting - Query Vector Adaption - Query Decomposition - Result refinement - When to retrieve - What (new query) to retrieve aim of providing a comprehensive taxonomy and highlighting open research challenges.

    5 Generation-time feedback
    RAG systems face two fundamental challenges: determining when to retrieve external knowledge, since not all queries benefit from it, and how to retrieve relevant content effectively (Su et al., 2024b). Classical RAG pipelines rigidly follow a fixed sequence of retrieval, optionally ranking, followed by generation, limiting their ability to adapt to the context or task. To address these limitations, recent work has introduced adaptive RAG , where the retrieval strategy is dynamically adjusted according to the query, the model feedback, or the complexity of the task. We categorize this emerging line of work into three main classes.

    5.1 Rule-Based and Discriminative Approaches
    In-Context RALM (Retrieval-Augmented Language Model) (Ram et al., 2023) proposes retrieving relevant context documents during inference at fixed intervals (every s tokens, known as the retrieval stride), using the last l tokens of the input as the retrieval query. In a similar spirit, IRCoT (Interleaving Retrieval in a CoT) (Trivedi et al., 2023) dynamically retrieves documents if the CoT (Wei et al., 2022) step has not provided the answer. At first, it uses the original question to retrieve the context and then uses the last generated CoT sentence as a query for subsequent retrieval. However, both of these methods retrieve the context regardless of whether the LLM needs external context or not. Hence, the unnecessary retrieval steps add additional latency cost during answer generation. Also, the noisy retrieved context can lead to a wrong answer. CtRLA (Huanshuo et al., 2025) devises a latent space probing-based approach for making decisions regarding retrieval timings for adaptive retrieval augmented generation. The authors extract latent vectors that represent abstract concepts like honesty and confidence and use these dimensions to steer retrieval and control LLM behavior, leading to better performance and robust answers. To overcome the over-retrieval limitation of rule-based dynamic RAG methods, retrieval-ondemand approaches have been proposed. These methods trigger retrieval only when the LLM needs it, based on either external feedback (Section 5.2)or the LLM's own assessment (Section 5.3).

    5.2 Retrieval-on-Demand via Feedback Signals
    The feedback signals can come from different sources, including the answer uncertainty, the model's internal states, or context faithfulness and sufficiency. SKR (Wang et al., 2023b) asks LLM itself if additional information is needed to answer the query. If yes, then the retrieval round is triggered; otherwise, the answer is generated from the

    LLM's internal knowledge. However, the judgment is solely based on LLM, and without context, they try to be overconfident (Xiong et al., 2024). FLARE (Jiang et al., 2023) retrieves the documents only if the token probability is below a predefined threshold and uses the last generated sentence as a query for retrieval (excluding the uncertain tokens) and generates the response until the next uncertain token or completion is done. However, these uncertain tokens are not equally important to trigger a retrieval round. Based on this, DRAD (Su et al., 2024a) uses an external module for hallucination detection on entities in the generated answer; if the answer contains hallucination, the retrieval is triggered. The last generated sentence (without a hallucinated entity) is used as a query for retrieval. However, the choice of the new query for retrieval relies on heuristic strategies. Since the model's information needs may extend beyond the last sentence or CoT, it could require context from a broader span of the generation to effectively build confidence. Based on this motivation, DRAGIN (Su et al., 2024b), similar to FLARE, also considers the token probabilities as a criterion of the retrieval round but does not consider the uncertain tokens as a part of the new query. Further, it also reformulates the query using the keywords based on the model's internal attention weights and reasoning. SeaKR (Yao et al., 2025) computes the self-aware uncertainty using internal states of the LLM. If the uncertainty is above a threshold, then a retrieval round is triggered.

    Other types of works, like Rowen (Ding et al., 2024), consider the LLM's answer consistency as feedback. Rowen considers answer consistency across languages of the same question with semantically similar variations, and the consistency over answers generated by different LLMs. If the total consistency is below a predefined threshold, then the retrieval round is triggered. However, similar to SUNAR (V et al., 2025), the consistency can be toward wrong answers.
    However, these approaches consider all queries equally complex and might end up with noisy context retrieval and hence a wrong answer. Adaptive RAG (Jeong et al., 2024) uses a query routing mechanism that predicts whether the query needs retrieval or not. Further, it also decides on the number of retrieval rounds based on query complexity. However, it assumes that the retrieved context is relevant to the query without assessing its relevancy or sufficiency. Towards the relevancy, CRAG
    (Corrective RAG) (Yan et al., 2024) evaluates the relevance scores using a fine-tuned model, and classifies the retrieved document into correct, incorrect, and ambiguous. If the context is not correct, then a rewritten query is issued to the web search engine. Similar fashion, SIM-RAG (Yang et al., 2025b) focuses on the context sufficiency angle, and trains a lightweight critic model that provides feedback if the retrieved context is sufficient to generate the answer. If the information is not sufficient, then a new query is formulated using the original query and the already retrieved context, and a retrieval round is triggered. Further CoV-RAG (He et al., 2024) identifies errors, including reference and answer correctness, and truthfulness, and then scores them using a trained verifier. Based on the scores, either provide a final or rewrite the query and do a further retrieval round.

    5.3 Self-Triggered Retrieval via Reasoning
    
    In this section, we discuss works where LLM autonomously makes the decision on when to retrieve and how to retrieve through query decomposition or planning-based approaches without external triggers. These approaches are also termed Reasoning RAG or Agentic RAG . These approaches can be divided into mainly two categories: first, where the instructions for query decomposition, when to retrieve, and what to retrieve are provided in the prompt along with few-shot examples; second, where the language models are trained to decide by themselves whether to decompose the query, when to retrieve, and what to retrieve.
    Prompt-Based Methods. DeComP (Khot et al., 2023) divides a task into granular sub-tasks and delegates them to different components through actions. However, DeComP only acts as a trigger for when to retrieve and employs a BM25 retriever for getting relevant documents in a single shot. It does not subsequently generate reasoning steps to improve retrieval, thus not providing much indication as to how to retrieve. ReAct (Yao et al., 2022) interleaves the generation of verbal reasoning traces with actions that interact with the external environment. The verbal reasoning traces act as indicators of how to retrieve, and the actions themselves serve as triggers (when to retrieve). Similarly, SelfAsk (Press et al., 2023) proposes to decompose the original complex query into simpler sub-questions iteratively interleaved by a retrieval step and intermediate answer generation. At each step, the LLM makes a decision to generate a follow-up question if more information is needed, or it may generate the final answer. Authors observed that this approach helped cover diverse aspects of complex queries and improved search and downstream answering performance.
    However, these approaches do not have provision for correction of the entire reasoning trajectory, and an intermediate error may cause cascading failures. Searchain (Xu et al., 2024) proposes to mitigate this by constructing a global reasoning chain first, where each node comprises a retrievaloriented query, an answer from LLM to the query, and a flag indicating if additional knowledge is needed to arrive at a better answer. SMR (State Machine Reasoning) (Lee et al., 2025) identifies the issues of the CoT-based query decomposition and retrieval methods like ReAct (Yao et al., 2022), where the CoT might result in redundant reasoning (new queries that result in the retrieval of the same documents) and misguided reasoning (new query diverges from the user's intent). To address these limitations, SMR proposes three actions: Refine, Rerank, and Stop. Action Refine updates the query using the feedback from the already retrieved documents, and a retrieval round is triggered. Then the retrieved documents are ranked according to the old query to make sure only the relevant information is used to answer. Finally, the Stop action is called to stop the reasoning if a sufficient retrieval quality is achieved, which helps in token efficiency and prevents overthinking.
    ---------------------- this was the passage that we matched to the query -------------
    MCTS-RAG (Hu et al., 2025) combines Monte Carlo Tree Search (MCTS) with RetrievalAugmented Generation (RAG) to improve reasoning and retrieval in language models. It guides the search for relevant information using MCTS to explore promising retrieval paths, enhancing answer accuracy. However, it is computationally expensive due to the iterative tree search process and may struggle with highly noisy or irrelevant documents. Search-O1 (Li et al., 2025a) proposes an agentic search workflow for reasoning augmented retrieval by letting the Large Reasoning Models (LRMs) like O1 decide autonomously when to retrieve by detecting the presence of salient uncertain words in their output. Additionally, they augment the workflow with a reason-in-documents step, where LRMs analyze the documents in depth to remove noise and reduce redundancy before employing them to generate the final answer.
    --------------------------------------------------------------------------------------
[end on page 8]
```

### Document #2 — HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation
- `doc_id`: `doc::b0610cc6134b401db0ea68a77096e883`
- **Last Updated:** 2025-11-04T13:49:26.359552+00:00
- **Context:** This document examines advancements in retrieval-augmented generation methods for language models.
- **Content fetched inside document:**
```text
[start on page 1]
    ---------------------- this was the passage that we matched to the query -------------
    Abstract
    Retrieval-augmented generation (RAG) has become a fundamental paradigm for addressing the challenges faced by large language models in handling real-time information and domainspecific problems. Traditional RAG systems primarily rely on the in-context learning (ICL) capabilities of the large language model itself. Still, in-depth research on the specific capabilities needed by the RAG generation model is lacking, leading to challenges with inconsistent document quality and retrieval system imperfections. Even the limited studies that fine-tune RAG generative models often lack a granular focus on RAG task or a deeper utilization of chain-of-thought processes . To address this, we propose that RAG models should possess three progressively hierarchical abilities (1) Filtering: the ability to select relevant information; (2) Combination: the ability to combine semantic information across paragraphs; and (3) RAGspecific reasoning: the ability to further process external knowledge using internal knowledge. Thus, we introduce our new RAG instruction fine-tuning method, Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation (HIRAG) incorporates a "think before answering" strategy. This method enhances the model's open-book examination capability by utilizing multi-level progressive chainof-thought. Experiments show that the HIRAG training strategy significantly improves the model's performance on datasets such as RGB, PopQA, MuSiQue, HotpotQA, and PubmedQA.
    --------------------------------------------------------------------------------------
[end on page 1]
```

### Document #3 — FrugalRAG: Learning to retrieve and reason for multi-hop QA
- `doc_id`: `doc::ece5670065a746a59112bfbd2cd1053b`
- **Last Updated:** 2025-11-04T13:51:56.886468+00:00
- **Context:** Research on enhancing QA through retrieval-augmented generation frameworks.
- **Content fetched inside document:**
```text
[start on page 2]
    2 Related Work
    Which metric to optimize. Multi-hop QA involves two sub-tasks: retrieving relevant documents, and then answering the question based on the documents. Some methods report document retrievalspecific metrics such as recall [Hsu et al., 2024] whereas others report final answer metrics such as exact match [Jin et al., 2025]. Typically, a model is trained to optimize a particular metric (such as recall) and also evaluated on the same metric. For robustness, in this work we train on the recall metric and test on all metrics, including final answer metrics.
    ---------------------- this was the passage that we matched to the query -------------
    Prompting-based RAG approaches. With the recent advancements in the capabilities of large API-based LMs, some works explored prompting to call external search/retrievers at inference. Toolformer[Schick et al., 2023] uses a self-supervised objective to train an external model that decides to call tools (like Bing and Google search engines). ReAct [Yao et al., 2023] is another powerful prompting technique that allows the model to structure its outputs as thoughts, actions and observations, yielding significant improvements in the ability of LLMs to interact with external environments. Trivedi et al. [2022a] proposed IRCoT, another prompting strategy that alternates between chain-of-thought [Wei et al., 2022] steps and gathering evidence through retrievals. By using the intermediate traces, the IRCoT is able to decide what to retrieve by issuing the right search queries. Iter-RetGen [Shao et al., 2023] improves evidence gathering in multi-hop scenarios by combining
    --------------------------------------------------------------------------------------
    retrieval and generation iteratively, such that a model's response is incorporated in the reasoning trace. However, both IRCoT [Trivedi et al., 2022a] and Iter-RetGen [Shao et al., 2023] rely on a fixed or predefined number of retrieval loops at inference, offering limited control over latency.
    Finetuning-based techniques. A prevalent method for multi-hop QA using small LMs is supervised finetuning using reasoning traces from a large LM such as GPT-4 [Asai et al., 2023, Chan et al., 2024]. Other methods are trained to predict the next query to be retrieved [Chan et al., 2024]. Methods that scale the test-time compute that infer using multiple trajectories have also been proposed [Wang et al., 2025]. Recently, reinforcement learning-based techniques have been proposed that develop a reward based on outputting the ground-truth answer [Jin et al., 2025]. However, none of the techniques focus on efficiency of the solution. In fact, in Search-R1, the goal of RL is to increase the number of searches. Instead, we use RL to decrease the average number of searches done by our model.
    Traditional RAG approaches. Early work in grounding generation with real world documents focused on end-to-end differentiable encoder-decoder pipeline REALM [Guu et al., 2020], which augments Masked-Language Modeling (MLM) with a latent retriever model, backpropagating through retrieval to learn both retriever and generator jointly. However, this approach incurs significant computational cost and has only been shown to work with relatively smaller models like T5 [Raffel et al., 2020]. Building on this, Lewis et al. [2020] proposed a general finetuning strategy, RAG-Token which demonstrated that join-training outperforms fixed dense retrieval and BM25.
    RL-based Retrieval Augmented Generation. Recently, framing search query as an RL problem has received attention. LeReT [Hsu et al., 2024] performs preference optimization using diverse few shot prompts leveraging hundred-thousands of ground truth annotated documents. However, LeReT utilizes a fixed amount of compute per instance during inference and cannot be readily generalized to variable-hop scenarios. Similarly, concurrent works, Jin et al. [2025] and Chen et al. [2025] propose end-to-end RL-based optimization that only leverages the final answer annotation. These methods show that RL can effectively be used to teach the search query generator model to issue more search queries for multi-hop problems without considering latency . Our two-stage RL framework, by contrast, first explores without RL to maximize recall and then learns to stop at test time using RL.
[end on page 3]
```


## LLM Answers (12.72s)
**Tech:** OpenAI's GPT-5 (reasoning low)

**Result:**

Here are unconventional RAG innovations from recent papers:

- Retrieval on demand using model signals: Trigger search only when needed—e.g., when token probabilities drop (FLARE), when an external hallucination detector flags entities (DRAD), or when the model’s own internal states show high uncertainty (SeaKR). Some even trigger when answers across languages/models disagree (Rowen) [1]. Example: if the model hesitates on a date, it pauses and searches; if it invents a person’s affiliation, a detector forces a corrective lookup.

- Corrective/critic loops over retrieved context: CRAG judges retrieved docs as correct/incorrect/ambiguous and rewrites the query if needed; SIM-RAG trains a critic to decide if context is sufficient before proceeding; CoV-RAG scores truthfulness, reference, and answer correctness, then either finalizes or rewrites and retrieves again [1]. Example: if top docs are off-topic or partial, the system auto-reforms the query and re-searches.

- Agentic/self-triggered retrieval: Systems decompose queries and plan retrieval themselves (ReAct, SelfAsk, Searchain). SMR adds explicit actions—Refine (rewrite query using retrieved feedback), Rerank (ensure relevance), and Stop (avoid overthinking), reducing redundancy and drift [1][3]. Example: after pulling similar docs, the agent reranks to focus, then stops once evidence quality is “good enough.”

- Search as guided exploration: MCTS-RAG uses Monte Carlo Tree Search to explore retrieval paths; Search-O1 lets large reasoning models detect “salient uncertain” words to decide when to fetch and includes a reason-in-documents phase to strip noise before answering [1].

- Adaptive routing: Predict whether to retrieve at all and how many rounds based on query complexity, not a fixed pipeline [1].

- Train the generator to “think for RAG”: HIRAG instruction-tunes three skills—filtering, combining across paragraphs, and RAG-specific reasoning—via multi-level, think-before-answer chains [2].

- Efficiency-first RL: FrugalRAG’s two-stage RL learns to retrieve enough to answer while minimizing the number of searches, unlike methods that encourage more searches [3].


**Citations:**

[1] doc::53b521e646b84289b46e648c66dde56a — Test-time Corpus Feedback: From Retrieval to RAG

[2] doc::b0610cc6134b401db0ea68a77096e883 — HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation

[3] doc::ece5670065a746a59112bfbd2cd1053b — FrugalRAG: Learning to retrieve and reason for multi-hop QA

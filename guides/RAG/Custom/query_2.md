# Query 2

## Result 
(latency retrieval is ~5.3 seconds)

**Original Query:** does long context windows now remove the need for RAG?

**LLM answer:**

No. Bigger context windows help, but they haven’t removed the need for retrieval-augmented generation (RAG).

What we know from recent studies:
- Cost and citations: Even when models can read lots of text, RAG is typically more cost‑efficient and better at surfacing sources, and many systems now let the model decide when to retrieve versus rely on its long context (hybrid use) [1].
- Real‑world tasks: In electronic health records, RAG matched or beat using the latest notes and came close to full‑context performance while using far fewer tokens—showing it remains competitive and efficient as contexts grow [4].
- Long narratives still strain models: Long‑context LLMs suffer from limits like “lost in the middle,” especially for 200k+ token narratives; RAG (and improved, stateful variants like ComoRAG) are needed to maintain and integrate evolving evidence over long stories [3].
- Practical constraints of stuffing context: Simply dumping many documents into context can hurt performance and add compute. New RAG designs either compress/organize knowledge or select only useful passages to ease context pressure while keeping quality high [2][6].

When long context may suffice: If the relevant information already sits neatly inside the available window and doesn’t require up‑to‑date knowledge, explicit retrieval may add little. But for accuracy, fresh knowledge, traceable citations, efficiency, and long, messy corpora, RAG (often in hybrid form) remains the preferred approach [1][4][3].

**Citations:**

[1] doc::688cfbc0abdc4520a73e219ac26aff41 — A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions

[2] doc::92acc4a2fba74f329db33189ec5cfef4 — PL-CA: A Parametric Legal Case Augmentation Framework

[3] doc::6f44119d59b4473aa1bb4b535ab6400f — ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning

[4] doc::0cc9a576aa8a43b58ee25a1c3e4879bf — Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs

[6] doc::5f5181c36d374a038646a72443b62fa7 — Distilling a Small Utility-Based Passage Selector to Enhance Retrieval-Augmented Generation



--------------------------------------------------------------------------------------------------------------------------------------

# RAG Pipeline

**Latency Breakdown:**

- Query optimizer: 3.83s (boots up Qdrant client in parallel)
- Retrieve: 0.78s
- Rerank: 0.76 s
- Context expansion: 1.11 s
- LLM answer: 14–20 s

## Query Optimizer (3.83s)

**Tech:** GPT-4o-mini, structured responses

1. **Query:** impact of long context windows on retrieval augmentation generation
2. **Search Method:** hybrid

## Retrieve (0.78s)

**Tech:** Qdrant, LlamaIndex

**Results:**

Query 1 (hybrid) top 20 for query: impact of long context windows on retrieval augmentation generation

[1] score=0.5000 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C02::251104164223
  text: 5 Generation-time feedback 5.1 Rule-Based and Discriminative Approaches In-Context RALM (Retrieval-Augmented Language Model) (Ram et al., 2023) proposes retrieving relevant context documents during inference at fixed intervals (every s tokens, known as the retrieval stride), using the last l tokens of the input as the retrieval query. In a similar spirit, IRCoT (Interleaving Retrieval in a CoT) (Trivedi et al., 2023) dynamically retrieves documents if the CoT (Wei et al., 2022) step has not provided the answer. At first, it uses the original question to retrieve the context and then uses the last generated CoT sentence as a query for subsequent retrieval. However, both of these methods retrieve the context regardless of whether the LLM needs external context or not. Hence, the unnecessary retrieval steps add additional latency cost during answer generation. Also, the noisy retrieved context can lead to a wrong answer. CtRLA (Huanshuo et al., 2025) devises a latent space probing-based approach for making decisions regarding r

[2] score=0.5000 doc=docs_ingestor/docs/arxiv/2508.10419.pdf chunk=S3::C02::251104161914
  text: 1 Introduction plot, characters, and their evolving motivations (JohnsonLaird 1983). The complexity of this process is well exemplified by a classic narrative question 'Why did Snape kill Dumbledore?' from the Harry Potter series. Answering this requires weaving a complete web of evidence from disparate clues spanning multiple books-Dumbledore's terminal illness, the Unbreakable Vow, and Snape's deeply concealed loyalty. The true significance of these clues is only fully reconciled in hindsight. This capability is what we term stateful reasoning : it demands more than linking static evidence; it requires maintaining a dynamic memory of the narrative, one that is constantly updated as new revelations emerge. Long-context LLMs have demonstrated promising performance on benchmarks such as the 'Needle in a Haystack' test (Eisenschlos, Yogatama, and Al-Rfou 2023) in recent years. However, their capacity to process long narratives (200k+ tokens) remains limited by finite context windows. Furthermore, as the input length increases,

[3] score=0.4344 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C06::251104164223
  text: 5 Generation-time feedback 5.2 Retrieval-on-Demand via Feedback Signals However, these approaches consider all queries equally complex and might end up with noisy context retrieval and hence a wrong answer. Adaptive RAG (Jeong et al., 2024) uses a query routing mechanism that predicts whether the query needs retrieval or not. Further, it also decides on the number of retrieval rounds based on query complexity. However, it assumes that the retrieved context is relevant to the query without assessing its relevancy or sufficiency. Towards the relevancy, CRAG

[4] score=0.4278 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S4::C03::251104134926
  text: 2 Related Work Upgrading of retrieval modules. From the perspective of retrieval methods, some studies have enhanced the quality of context by employing multistage retrieval reasoning (Asai et al., 2023b) (Gan et al., 2024), while others have designed adaptive retrieval modules that allow models to adjust retrieval behavior according to different tasks (Jeong et al., 2024). In terms of question understanding, some studies have improved search queries by rewriting, decomposing, and disambiguating (Chan et al., 2024). After retrieving articles, incorporating a ranking module can significantly enhance the final generation outcome (Glass et al., 2022)(Ram et al., 2023). RankRAG effectively integrates the ranking module with the generation module (Yu et al., 2024). These approaches have effectively improved the quality of retrieved articles in RAG systems. However, there is no such thing as a perfect context, and the generative model needs to be capable of handling contexts in various situations.

[5] score=0.3379 doc=docs_ingestor/docs/arxiv/2508.14817.pdf chunk=S2::C01::251104163608
  text: Abstract Electronic health records (EHRs) are long, noisy, and often redundant, posing a major challenge for the clinicians who must navigate them. Large language models (LLMs) offer a promising solution for extracting and reasoning over this unstructured text, but the length of clinical notes often exceeds even stateof-the-art models' extended context windows. Retrieval-augmented generation (RAG) offers an alternative by retrieving task-relevant passages from across the entire EHR, potentially reducing the amount of required input tokens. In this work, we propose three clinical tasks designed to be replicable across health systems with minimal effort: 1) extracting imaging procedures, 2) generating timelines of antibiotic use, and 3) identifying key diagnoses. Using EHRs from actual hospitalized patients, we test three state-of-the-art LLMs with varying amounts of provided context, using either targeted text retrieval or the most recent clinical notes. We find that RAG closely matches or exceeds the performance of using rec

[6] score=0.2861 doc=docs_ingestor/docs/arxiv/2507.19102.pdf chunk=S10::C06::251104143003
  text: 1 Introduction To this end, we propose a distilling approach that jointly learn pseudo-answer generation and utility judgments from teacher LLMs. For utility judgments with the student selector on a long initial ranking list, we propose a sliding window method that moves from higher to lower positions. At each step, the selector generates pseudo answers based on the selected useful results, and slides to the next window, which is comprised of the so-far selected useful results and the unseen passages. New selected useful results will be prepended to the selected result pool, and duplicates in the pool will be deleted, maintaining an ordered list of selected useful results. This process is repeated until all the candidate results are judged. This process ensures that the final selected useful results are based on the information of the entire candidate results. It also incurs a smaller cost than the above-mentioned ranking distillation due to smaller overlap between windows.

[7] score=0.2511 doc=docs_ingestor/docs/arxiv/2509.03787.pdf chunk=S4::C02::251104170357
  text: 2 Related Work 2.1 RAG Systems Recent research has increasingly centered on evaluating and improving the performance of RAG systems, while also introducing tailored architectures and task-specific adaptations [23, 27, 34, 35, 44, 48, 72, 74, 84, 87]. For instance, [44] explored the impact of the positioning of the documents within the context window on the output generated by the LLM. In a similar study, Cuconasu et al. [23] investigated the impact of the positioning of not only relevant documents but also noisy irrelevant documents on the RAG system. Gao et al. [27] introduced a RAG pipeline that jointly optimizes its policy network, retriever, and answer generator via reinforcement learning, achieving better performance and lower retrieval cost than separately trained modules. While these works provide valuable insights into RAG behavior, they primarily focus on the RAG settings and their performance. In contrast, our work investigates RAG systems under adversarial conditions, where the context may be intentionally crafted

[8] score=0.2249 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S9::C22::251104142800
  text: 4 Year-by-Year Progress in RAG 4.2 Major Milestones (2020-2024) 2023 -RAG Meets LLMs. By 2023, mainstream LLM-based applications (e.g., ChatGPT with plugins, Bing Chat, enterprise chatbots) widely incorporated retrieval [27]. This retrieve-then-generate paradigm was used to mitigate hallucinations and update factual knowledge post-training. The debate emerged as to whether long-context LLMs (with tens of thousands of tokens) could negate the need for retrieval systems. Studies like [54] showed that while large context windows can absorb more text, RAG remains more cost-efficient and better at exposing citations. Hybrid approaches also arose: letting a model choose between retrieving or just using a long context. Overall, RAG became a cornerstone for credible LLM deployments needing up-to-date knowledge and interpretability. SOme of the areas that received attention in 2023 are discussed below: Scale and few-shot learning: Atlas[36], an 11B-parameter retrieval-augmented model, achieved 42.4% accuracy on Natural Questions with

[9] score=0.2126 doc=docs_ingestor/docs/arxiv/2508.10419.pdf chunk=S4::C10::251104161914
  text: 2 Methodology 2.2 The Hierarchical Knowledge Source Episodic Layer: Reconstructing Narrative Flow. The previous two layers equip views of both factual details and high-level concepts. However, they lack temporal development or plot progression that can be especially crucial for narratives. To enable such view with long-range causal chains, we introduce the episodic layer, X epi , which aims to reconstruct the plotline and story arc by capturing the sequential narrative development. The process features a sliding window summarization across text chunks; each resulting node is then a summary that aggregates the narrative development of continuous or causally related events according to the timeline. Optionally, the sliding window process can be applied recursively to form higher-level views of content progression, extracting different levels of narrative flow as part of the knowledge source.

[10] score=0.2080 doc=docs_ingestor/docs/arxiv/2509.06356.pdf chunk=S2::C01::251104172827
  text: Abstract Conventional RAG is considered one of the most effective methods for addressing model knowledge insufficiency and hallucination, particularly in the judicial domain that requires high levels of knowledge rigor, logical consistency, and content integrity. However, the conventional RAG method only injects retrieved documents directly into the model's context, which severely constrains models due to their limited context windows and introduces additional computational overhead through excessively long contexts, thereby disrupting models' attention and degrading performance on downstream tasks. Moreover, many existing benchmarks lack expert annotation and focus solely on individual downstream tasks while real-world legal scenarios consist of multiple mixed legal tasks, indicating conventional benchmarks' inadequacy for reflecting models' true capabilities. To address these limitations, we propose PL-CA, which introduces a parametric RAG (P-RAG) framework to perform data augmentation on corpus knowledge and encode this l

[11] score=0.1763 doc=docs_ingestor/docs/arxiv/2508.14817.pdf chunk=S10::C06::251104163608
  text: 8 Discussion Across all tasks and models, we observed a consistent trend: retrieval-augmented generation was able to closely match the performance of fullcontext inputs with far fewer tokens.

[12] score=0.1684 doc=docs_ingestor/docs/arxiv/2508.09497.pdf chunk=S18::C02::251104160546
  text: Conclusion While DPS shows strong performance across settings, several limitations remain for future research. DPS is limited by the maximum input token length of the LLM, and when the candidate passage set is too large, DPS needs to employ a sliding window strategy for inference. Moreover, he approach still depends on the initial retrieval step; missing key candidates can hinder downstream selection. Moreover, current selection does not explicitly model reasoning chains or intermediate steps, which may be necessary for more complex queries. Although DPS does not require generator finetuning, it still relies on supervised data to train the selector, limiting its applicability in low-resource domains. Inference cost, though moderate, is higher than pointwise rerankers, which may impact latency in time-sensitive scenarios.

[13] score=0.1561 doc=docs_ingestor/docs/arxiv/2507.20059.pdf chunk=SDOC::SUM::251104151210
  text: This study evaluates the effectiveness of Retrieval-Augmented Generation (RAG) systems across diverse knowledge sources and tasks. Utilizing MASSIVEDS, a large-scale datastore, the authors found that retrieval mainly benefits smaller models while larger models exhibit diminishing returns. The analysis shows no single retrieval source consistently outperforms others, highlighting the necessity for adaptive retrieval strategies. Furthermore, efforts such as reranking and query routing have limited improvement on overall performance. Key challenges include inaccurate relevance estimation and training-inference mismatch, suggesting a need for improved integration of retrieval mechanisms with language models. The findings indicate a critical gap in deploying RAG systems effectively in real-world scenarios, particularly for larger models that increasingly internalize domain knowledge without retrieval support.

[14] score=0.1383 doc=docs_ingestor/docs/arxiv/2508.09874.pdf chunk=S8::C01::251104161352
  text: 7. Related Work Retrieval-Augmented Generation Retrieval-Augmented Generation (RAG) enhances language models by incorporating knowledge from external sources, with retrieval granularity ranging from documents [Chen et al., 2017] to passages [Guu et al., 2020, Izacard et al., 2023, Lewis et al., 2020] to tokens [He et al., 2021, Khandelwal et al., 2019, Min et al., 2022, Yogatama et al., 2021]. Tokenlevel retrieval achieves superior performance for rare patterns and out-of-domain scenarios but introduces substantial computation overhead during inference. While non-differentiable retrieval mechanisms prevent end-to-end optimization and memory token approaches [Chevalier et al., 2023] enable differentiable access but are limited to local contexts, Memory Decoder provides both differentiable optimization and full-dataset knowledge access without expensive retrieval operations or model-specific datastores.

[15] score=0.1341 doc=docs_ingestor/docs/arxiv/2507.23334.pdf chunk=S5::C07::251104154519
  text: 3. RETRIEVAL AUGMENTED GENERATION 3.1.3 Generation The retrieved context c is provided to a generator LLM, which produces an output sequence using next-token prediction. Each token x i is generated conditioned on the input query q , the retrieved context c , and the previously generated tokens x <i : This structure enables the model to dynamically incorporate external knowledge during inference, improving factual accuracy and adaptability without retraining.

[16] score=0.1304 doc=docs_ingestor/docs/arxiv/2507.19562.pdf chunk=S12::C02::251104143937
  text: B. Retrieval-Augmented Generation (RAG) for Code Generalization Retrieve topk most similar instructions from the PennyLang corpus using cosine similarity in the vector space. Concatenate the retrieved instruction-code pairs into the model input context window.

[17] score=0.1252 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S5::C12::251104134926
  text: 3 HIRAG 3.2 Training Strategies model excel in selecting relevant information, integrating it, and reasoning about it within document contexts.

[18] score=0.1144 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C01::251104164223
  text: 5 Generation-time feedback RAG systems face two fundamental challenges: determining when to retrieve external knowledge, since not all queries benefit from it, and how to retrieve relevant content effectively (Su et al., 2024b). Classical RAG pipelines rigidly follow a fixed sequence of retrieval, optionally ranking, followed by generation, limiting their ability to adapt to the context or task. To address these limitations, recent work has introduced adaptive RAG , where the retrieval strategy is dynamically adjusted according to the query, the model feedback, or the complexity of the task. We categorize this emerging line of work into three main classes.

[19] score=0.1002 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S13::C01::251104164223
  text: Key Insights The external feedback signals help in reducing retrieval rounds. These signals can come from different sources, at the LLM level (e.g., token generation confidence), at the answer level ( e.g., uncertainty or hallucination), and at the context level (e.g., relevancy or sufficiency). However, these methods may still retrieve noisy or irrelevant context, and complexity assessment remains a challenge.

[20] score=0.0942 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S3::C02::251104134926
  text: 1 Introduction by incorporating an information retrieval component. While LLMs often use in-context learning (Gao et al., 2024) for generation, practical issues such as low-quality or poorly ranked retrieved documents can hinder RAG's effectiveness. These challenges emphasize the need for instruction-tuning tailored to RAG tasks. Fine-tuning generative models specifically for RAG improves their ability to integrate retrieved information (Zhang et al., 2024) (Yu et al., 2024), resulting in more accurate and contextually relevant responses compared to generalpurpose models.


## Rerank (0.76s)

**Tech:** Cohere API

**Results**

Rerank summary:
- strategy=cohere
- model=rerank-english-v3.0
- candidates=20
- eligible_above_threshold=20
- kept=6 (threshold=0)

Reranked Relevant (6/20 kept ≥ 0) top 6 for query: does long context windows now removed the need for RAG?

[1] score=0.9875 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S9::C22::251104142800
  text: 4 Year-by-Year Progress in RAG 4.2 Major Milestones (2020-2024) 2023 -RAG Meets LLMs. By 2023, mainstream LLM-based applications (e.g., ChatGPT with plugins, Bing Chat, enterprise chatbots) widely incorporated retrieval [27]. This retrieve-then-generate paradigm was used to mitigate hallucinations and update factual knowledge post-training. The debate emerged as to whether long-context LLMs (with tens of thousands of tokens) could negate the need for retrieval systems. Studies like [54] showed that while large context windows can absorb more text, RAG remains more cost-efficient and better at exposing citations. Hybrid approaches also arose: letting a model choose between retrieving or just using a long context. Overall, RAG became a cornerstone for credible LLM deployments needing up-to-date knowledge and interpretability. SOme of the areas that received attention in 2023 are discussed below: Scale and few-shot learning: Atlas[36], an 11B-parameter retrieval-augmented model, achieved 42.4% accuracy on Natural Questions with

[2] score=0.9626 doc=docs_ingestor/docs/arxiv/2509.06356.pdf chunk=S2::C01::251104172827
  text: Abstract Conventional RAG is considered one of the most effective methods for addressing model knowledge insufficiency and hallucination, particularly in the judicial domain that requires high levels of knowledge rigor, logical consistency, and content integrity. However, the conventional RAG method only injects retrieved documents directly into the model's context, which severely constrains models due to their limited context windows and introduces additional computational overhead through excessively long contexts, thereby disrupting models' attention and degrading performance on downstream tasks. Moreover, many existing benchmarks lack expert annotation and focus solely on individual downstream tasks while real-world legal scenarios consist of multiple mixed legal tasks, indicating conventional benchmarks' inadequacy for reflecting models' true capabilities. To address these limitations, we propose PL-CA, which introduces a parametric RAG (P-RAG) framework to perform data augmentation on corpus knowledge and encode this l

[3] score=0.8727 doc=docs_ingestor/docs/arxiv/2508.10419.pdf chunk=S3::C02::251104161914
  text: 1 Introduction plot, characters, and their evolving motivations (JohnsonLaird 1983). The complexity of this process is well exemplified by a classic narrative question 'Why did Snape kill Dumbledore?' from the Harry Potter series. Answering this requires weaving a complete web of evidence from disparate clues spanning multiple books-Dumbledore's terminal illness, the Unbreakable Vow, and Snape's deeply concealed loyalty. The true significance of these clues is only fully reconciled in hindsight. This capability is what we term stateful reasoning : it demands more than linking static evidence; it requires maintaining a dynamic memory of the narrative, one that is constantly updated as new revelations emerge. Long-context LLMs have demonstrated promising performance on benchmarks such as the 'Needle in a Haystack' test (Eisenschlos, Yogatama, and Al-Rfou 2023) in recent years. However, their capacity to process long narratives (200k+ tokens) remains limited by finite context windows. Furthermore, as the input length increases,

[4] score=0.8510 doc=docs_ingestor/docs/arxiv/2508.14817.pdf chunk=S2::C01::251104163608
  text: Abstract Electronic health records (EHRs) are long, noisy, and often redundant, posing a major challenge for the clinicians who must navigate them. Large language models (LLMs) offer a promising solution for extracting and reasoning over this unstructured text, but the length of clinical notes often exceeds even stateof-the-art models' extended context windows. Retrieval-augmented generation (RAG) offers an alternative by retrieving task-relevant passages from across the entire EHR, potentially reducing the amount of required input tokens. In this work, we propose three clinical tasks designed to be replicable across health systems with minimal effort: 1) extracting imaging procedures, 2) generating timelines of antibiotic use, and 3) identifying key diagnoses. Using EHRs from actual hospitalized patients, we test three state-of-the-art LLMs with varying amounts of provided context, using either targeted text retrieval or the most recent clinical notes. We find that RAG closely matches or exceeds the performance of using rec

[5] score=0.0321 doc=docs_ingestor/docs/arxiv/2509.03787.pdf chunk=S4::C02::251104170357
  text: 2 Related Work 2.1 RAG Systems Recent research has increasingly centered on evaluating and improving the performance of RAG systems, while also introducing tailored architectures and task-specific adaptations [23, 27, 34, 35, 44, 48, 72, 74, 84, 87]. For instance, [44] explored the impact of the positioning of the documents within the context window on the output generated by the LLM. In a similar study, Cuconasu et al. [23] investigated the impact of the positioning of not only relevant documents but also noisy irrelevant documents on the RAG system. Gao et al. [27] introduced a RAG pipeline that jointly optimizes its policy network, retriever, and answer generator via reinforcement learning, achieving better performance and lower retrieval cost than separately trained modules. While these works provide valuable insights into RAG behavior, they primarily focus on the RAG settings and their performance. In contrast, our work investigates RAG systems under adversarial conditions, where the context may be intentionally crafted

[6] score=0.0233 doc=docs_ingestor/docs/arxiv/2507.19102.pdf chunk=S10::C06::251104143003
  text: 1 Introduction To this end, we propose a distilling approach that jointly learn pseudo-answer generation and utility judgments from teacher LLMs. For utility judgments with the student selector on a long initial ranking list, we propose a sliding window method that moves from higher to lower positions. At each step, the selector generates pseudo answers based on the selected useful results, and slides to the next window, which is comprised of the so-far selected useful results and the unseen passages. New selected useful results will be prepended to the selected result pool, and duplicates in the pool will be deleted, maintaining an ordered list of selected useful results. This process is repeated until all the candidate results are judged. This process ensures that the final selected useful results are based on the information of the entire candidate results. It also incurs a smaller cost than the above-mentioned ranking distillation due to smaller overlap between windows.
  
## Context Expansion (1.03s)

**Tech:** Redis (to fetch neighbors and additional informtion)

**Results:**

### Document #1 — A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions
- `doc_id`: `doc::688cfbc0abdc4520a73e219ac26aff41`
- **Last Updated:** 2025-11-04T14:28:00.715095+00:00
- **Context:** A review of RAG's evolution and its evolving role in knowledge-intensive NLP.
- **Content fetched inside document:**
```text
[start on page 14]
    4 Year-by-Year Progress in RAG
    
    4.2 Major Milestones (2020-2024)
    
    performance while using 25 × fewer parameters, underscoring retrieval's efficiency gains.

    Google's ATLAS [35] unifies retrieval and generation during pre-training . Using the unsupervised Contriever dense retriever [37] and a Fusion-in-Decoder reader, ATLAS achieves new state-of-the-art results on Natural Questions and TriviaQA and, in few-shot mode, outperforms much larger PaLM-540B by 3 EM on NQ with only 64 examples. These results highlight that high-quality retrieval plus multi-document fusion can outperform sheer parameter count.

    Beyond QA, RAG became core to knowledge-grounded dialogue . Meta's BlenderBot 3 [80] couples a 175B LLaMA-style generator with live internet search and long-term memory, reducing hallucinations and increasing user engagement in open-domain conversation. BlenderBot 3's deployment study shows that users prefer retrieval-grounded responses and that continual online learning can keep the retriever index current. Retrieval quality itself improved through unsupervised contrastive training. Contriever [37] dispenses with labelled (question, passage) pairs, yet surpasses BM25 and even DPR on BEIR benchmarks, making high-recall indexes available for any corpus. Such retrievers power ATLAS and other 2022 RAG systems, demonstrating that scalable training data is no longer a bottleneck.

    Finally, 2022 work extended RAG to fact-checking, summarization, and few-shot learning . ATLAS gains 5F1 on FEVER, indicating that retrieved evidence helps verdict generation. Early studies in retrieval-augmented summarization show improved factual consistency by grounding summaries in external documents. Few-shot evaluations reveal that retrieval narrows the data gap: ATLAS and RETRO deliver strong accuracy with under 100 task examples, whereas closed-book baselines require orders of magnitude more data.

    Outlook By the end of 2022, RAG had broadened from open-domain QA into a general recipe for knowledge-intensive NLP. Parameter-efficient hybrids like RETRO and ATLAS challenge the notion that bigger models alone yield better knowledge; instead, high-quality retrieval and multi-document reasoning emerge as key levers. Open challenges include faster retrieval over trillion-token corpora, differentiable multi-hop reasoning, and robust evaluation of evidential faithfulness, but 2022 firmly established retrieval-augmented generation as a premier path toward up-to-date, factual, and data-efficient language models.

    ---------------------- this was the passage that we matched to the query -------------
    2023 -RAG Meets LLMs. By 2023, mainstream LLM-based applications (e.g., ChatGPT with plugins, Bing Chat, enterprise chatbots) widely incorporated retrieval [27]. This retrieve-then-generate paradigm was used to mitigate hallucinations and update factual knowledge post-training. The debate emerged as to whether long-context LLMs (with tens of thousands of tokens) could negate the need for retrieval systems. Studies like [54] showed that while large context windows can absorb more text, RAG remains more cost-efficient and better at exposing citations. Hybrid approaches also arose: letting a model choose between retrieving or just using a long context. Overall, RAG became a cornerstone for credible LLM deployments needing up-to-date knowledge and interpretability. SOme of the areas that received attention in 2023 are discussed below:
    
    Scale and few-shot learning: Atlas[36], an 11B-parameter retrieval-augmented model, achieved 42.4% accuracy on Natural Questions with only 64 training examples, outperforming a 540B closed-book model by 3%. Atlas also set new few-shot records on TriviaQA and FEVER (gains of +3-5%), matching 540B-scale performance on multi-task benchmarks. Crucially, Atlas's dense document index can be easily updated with new text, demonstrating updatable knowledge.
    --------------------------------------------------------------------------------------

    Adaptive retrieval: Self-RAG[5] trains a single language model to generate special 'reflection' tokens that trigger on-demand retrieval and self-critique. In experiments, 7B and 13B Self-RAG models substantially outperformed ChatGPT and a RAG-augmented Llama-2-chat baseline on open-domain QA, reasoning, and fact-verification tasks, yielding much higher factual accuracy and citation precision.

    Knowledge-grounded dialogue: Retrieval augments dialogue systems to improve consistency and informativeness. Kumari et al.[47] incorporate retrieved persona and context snippets in long conversation modeling, showing that adding relevant knowledge improves response quality. Similarly, Kang et al.[43] propose SURGE , which retrieves relevant subgraphs from a knowledge graph and uses them to bias the response generation. SURGE produces more coherent, factual responses grounded in the retrieved

    knowledge.

    Summarization and explanation: RAG has been applied to summarization and explanation tasks. By retrieving source documents or evidence passages, RAG-augmented summarizers produce more accurate and detailed summaries than closed-book models. Likewise, in fact-checking pipelines, retrieving evidence before verification leads to more reliable verdicts and explanations. These applications extend RAG's grounding advantages beyond QA to a broader range of generative tasks.

    4.3 2025 - The Current Direction
    
    The community is exploring how to marry graph knowledge with text retrieval. A comprehensive survey formalised the GraphRAG paradigm and mapped design choices for graph-aware retrievers and generators [29]. A companion study compared vanilla RAG and GraphRAG across QA and summarisation, showing complementary strengths and proposing hybrid fusion strategies [28].
[end on page 16]
```

### Document #2 — PL-CA: A Parametric Legal Case Augmentation Framework
- `doc_id`: `doc::92acc4a2fba74f329db33189ec5cfef4`
- **Last Updated:** 2025-11-04T17:28:27.571103+00:00
- **Context:** Introducing PL-CA to address LLM limitations in legal applications.
- **Content fetched inside document:**
```text
[start on page 1]
    ---------------------- this was the passage that we matched to the query -------------
    Abstract
    
    Conventional RAG is considered one of the most effective methods for addressing model knowledge insufficiency and hallucination, particularly in the judicial domain that requires high levels of knowledge rigor, logical consistency, and content integrity. However, the conventional RAG method only injects retrieved documents directly into the model's context, which severely constrains models due to their limited context windows and introduces additional computational overhead through excessively long contexts, thereby disrupting models' attention and degrading performance on downstream tasks. Moreover, many existing benchmarks lack expert annotation and focus solely on individual downstream tasks while real-world legal scenarios consist of multiple mixed legal tasks, indicating conventional benchmarks' inadequacy for reflecting models' true capabilities. To address these limitations, we propose PL-CA, which introduces a parametric RAG (P-RAG) framework to perform data augmentation on corpus knowledge and encode this legal knowledge into parametric vectors, and then integrates this parametric knowledge into the LLM's feed-forward networks (FFN) via LoRA, thereby alleviating models' context pressure. Additionally, we also construct a multi-task legal dataset comprising more than 2000 training and test instances, which are all expertannotated and manually verified. We conduct our experiments on our dataset, and the experimental results demonstrate that our method reduces the overhead associated with excessively long contexts while maintaining competitive performance on downstream tasks compared to conventional RAG. Our code and dataset are provided in the appendix.
    --------------------------------------------------------------------------------------
[end on page 1]
```

### Document #3 — ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning
- `doc_id`: `doc::6f44119d59b4473aa1bb4b535ab6400f`
- **Last Updated:** 2025-11-04T16:19:14.656701+00:00
- **Context:** ComoRAG improves long narrative comprehension by mimicking human cognitive processes in retrieval methods.
- **Content fetched inside document:**
```text
[start on page 1]
    1 Introduction
    
    The core challenge of long-context narrative comprehension lies not merely in connecting discrete pieces of evidence, a task more naturally defined as multi-hop Question Answering (QA), but in performing a dynamic cognitive synthesis to grasp necessary background and content progression (Xu et al. 2024a). Unlike multi-hop QA (Yang et al. 2018), which seeks a static path through fixed facts, narrative comprehension requires emulating a human reader: continuously building and revising a global mental model of the
    
    * These authors contributed equally.
    
    † Project lead. Correspondence to: < liyanlxu@tencent.com >
    
    Figure 1: Comparison of RAG reasoning paradigms.
    
    (c) ComoRAG
    (b) Multi-step RAG
    Why Snape Kill
    
    Dumbledore Albus?
    
    Stateless Reasoning Stateful Reasoning Fragmented Evidence Lack of Fusion Memory-Organized Stateful Comprehension Contradictory Evidence Motive Unclear
    
    An act of loyalty
    
    not betrayal Snape Bully Harry Kills Albus Protect (a) Single-step RAG Shallow/Superficial Understanding
    
    Snape was a loyal
    
    Death Eater The Half-Blood Prince where Snape Coherent Context Formed Apparent Contradiction Causally Incomplete Event Unbreakable Vow Context-Grounded Exploratory Probing Dynamic Memory Workspace Knowledge Consolidation Consolida1on New Acquisition Acqulsition

    ---------------------- this was the passage that we matched to the query -------------
    plot, characters, and their evolving motivations (JohnsonLaird 1983). The complexity of this process is well exemplified by a classic narrative question 'Why did Snape kill Dumbledore?' from the Harry Potter series. Answering this requires weaving a complete web of evidence from disparate clues spanning multiple books-Dumbledore's terminal illness, the Unbreakable Vow, and Snape's deeply concealed loyalty. The true significance of these clues is only fully reconciled in hindsight. This capability is what we term stateful reasoning : it demands more than linking static evidence; it requires maintaining a dynamic memory of the narrative, one that is constantly updated as new revelations emerge. Long-context LLMs have demonstrated promising performance on benchmarks such as the 'Needle in a Haystack' test (Eisenschlos, Yogatama, and Al-Rfou 2023) in recent years. However, their capacity to process long narratives (200k+ tokens) remains limited by finite context windows. Furthermore, as the input length increases, these models are prone to the 'lost in the middle' problem (Liu et al. 2024), which raises perplexity and impairs generation quality. This limitation is particularly pronounced in narrative compre- hension tasks which require stateful reasoning. As a result, retrieval-augmented generation (RAG) (Lewis et al. 2020) has emerged as an important strategy for tackling long context comprehension with LLMs.
    --------------------------------------------------------------------------------------

    However, existing RAG methods still struggle to effectively address this challenge. Advanced single-step retrieval remains limited by its static index. This includes methods such as RAPTOR (Sarthi et al. 2024), which clusters and summarizes text chunks to retrieve at different levels of details; HippoRAGv2 (Guti´ errez et al. 2025), which mimics the human hippocampus by building a knowledge graph to achieve multi-hop reasoning in a single retrieval step. Nonetheless, single-step methods rely on one-shot static retrieval, which may lead to shallow comprehension. For example, in Figure 1(a), the evidence about Snape can mislead the model into making a false inference.

    As a remedy, multi-step retrieval methods offer a more promising direction, such as IRCoT (Trivedi et al. 2023), which interleaves the retrieval process with Chain-ofThought reasoning (Wei et al. 2022); Self-RAG (Asai et al. 2024), which trains a model to adaptively retrieve and reflect on evidence; and MemoRAG (Qian et al. 2025), which uses a dual-system architecture to generate clues from compressed global context. These methods all target to obtain richer context through iterative retrieval. However, their retrieval steps are typically independent, which lack coherent reasoning throughout explicit narrative progression, featuring fragmented evidence with a stateless comprehension. As illustrated in Figure 1(b), due to a lack of dynamic memory, multi-step retrieval fails to integrate contradictory evidence such as 'Snape protects/bullies Harry' and cannot understand the evolution of his actions, ultimately unable to yield the correct answer.

    In this work, we seek inspiration from the function of Prefrontal Cortex (PFC) in human brains, which employs a sophisticated reasoning process called Metacognitive Regulation (Fernandez-Duque, Baird, and Posner 2000). This process is not a single action but a dynamic interplay between new evidence acquisition , driven by goal-directed memory probes (Dobbins and Han 2006; Miller and Constantinidis 2024), and subsequent knowledge consolidation . During consolidation, new findings are integrated with past information to construct an evolving, coherent narrative. This iterative cycle allows the PFC to continuously assess its understanding and revise its strategy, providing a direct cognitive blueprint for our framework's stateful reasoning approach.

    We introduce ComoRAG, a cognitive-inspired, memoryorganized RAG framework, imitating the human Prefrontal Cortex (PFC) for achieving true stateful reasoning. At its core is a dynamic cognitive loop operating on a memory workspace, which actively probes and integrates new evidence to build a coherent narrative comprehension.

    This process, as illustrated in Figure 1(c), is a closed loop of evolving reasoning states. Faced with a complex query like 'Why did Snape kill Dumbledore?' , the system's memory state evolves from an initial 'causally incomplete event' ( Snape kills Albus ), to an 'apparent contradiction' upon finding contradictory information ( Snape protects Harry ), and ultimately to a logically consistent coherent context through deeper exploration and evidence fusion. Only in this final, complete cognitive state can ComoRAG perform the correct stateful reasoning, deriving the profound insight that it was 'an act of loyalty, not betrayal' .

    This cognitively-inspired design yields substantial improvements across four challenging long-context narrative benchmarks. ComoRAG is shown to consistently outperform all categories of strong baselines across each dataset. Our analysis reveals several key findings. First, these gains stem directly from the cognitive loop, which transforms a static knowledge base into a dynamic reasoning engine; for instance, accuracy on EN.MC jumps from a static-retrieval baseline of 64.6% to 72.9%, with performance efficiently converging in around 2-3 cycles. Second, our framework excels on narrative queries that require global understanding of plot progression, achieving up to a 19% relative F1 improvement on these challenging question types where others falter. Finally, our framework demonstrates remarkable modularity and generalizability. Its core loop can be flexibly integrated to existing RAG methods such as RAPTOR, which directly yields a 21% relative accuracy gain). Also, switching to a stronger model as the backbone LLM agents can upgrade reasoning in the entire cognitive loop, attaining accuracy from 72.93% to 78.17%. These results collectively validate that ComoRAG provides a principled, cognitivelyinspired new paradigm for retrieval-based long narrative comprehension towards stateful reasoning.
[end on page 2]
```

### Document #4 — Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs
- `doc_id`: `doc::0cc9a576aa8a43b58ee25a1c3e4879bf`
- **Last Updated:** 2025-11-04T16:36:08.460489+00:00
- **Context:** Evaluates RAG's effectiveness for clinical tasks in electronic health records processing.
- **Content fetched inside document:**
```text
[start on page 1]
    ---------------------- this was the passage that we matched to the query -------------
    Abstract
    
    Electronic health records (EHRs) are long, noisy, and often redundant, posing a major challenge for the clinicians who must navigate them. Large language models (LLMs) offer a promising solution for extracting and reasoning over this unstructured text, but the length of clinical notes often exceeds even stateof-the-art models' extended context windows. Retrieval-augmented generation (RAG) offers an alternative by retrieving task-relevant passages from across the entire EHR, potentially reducing the amount of required input tokens. In this work, we propose three clinical tasks designed to be replicable across health systems with minimal effort: 1) extracting imaging procedures, 2) generating timelines of antibiotic use, and 3) identifying key diagnoses. Using EHRs from actual hospitalized patients, we test three state-of-the-art LLMs with varying amounts of provided context, using either targeted text retrieval or the most recent clinical notes. We find that RAG closely matches or exceeds the performance of using recent notes, and approaches the performance of using the models' full context while requiring drastically fewer input tokens. Our results suggest that RAG remains a competitive and efficient approach even as newer models become capable of handling increasingly longer amounts of text.
    --------------------------------------------------------------------------------------
[end on page 1]
```

### Document #5 — Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain
- `doc_id`: `doc::22fe0bef1eda4417841c748de2612d72`
- **Last Updated:** 2025-11-04T17:03:57.601523+00:00
- **Context:** Evaluation of RAG systems in mitigating misinformation in health contexts.
- **Content fetched inside document:**
```text
[start on page 4]
    2 Related Work
    2.1 RAG Systems
    RAG systems emerged in part as an approach to overcoming the hallucination problem of LLMs, and to increase accuracy in generated responses by combining their generative capabilities with factual grounding provided by external knowledge [28, 31, 40]. In a basic RAG pipeline, a retrieval component first identifies and ranks relevant documents from a large corpus. Then, the top-k documents are passed to an LLM as context for generating the final response. This RAG architecture has been widely adopted for question answering and similar applications [2, 45, 71].
    ---------------------- this was the passage that we matched to the query -------------
    Recent research has increasingly centered on evaluating and improving the performance of RAG systems, while also introducing tailored architectures and task-specific adaptations [23, 27, 34, 35, 44, 48, 72, 74, 84, 87]. For instance, [44] explored the impact of the positioning of the documents within the context window on the output generated by the LLM. In a similar study, Cuconasu et al. [23] investigated the impact of the positioning of not only relevant documents but also noisy irrelevant documents on the RAG system. Gao et al. [27] introduced a RAG pipeline that jointly optimizes its policy network, retriever, and answer generator via reinforcement learning, achieving better performance and lower retrieval cost than separately trained modules. While these works provide valuable insights into RAG behavior, they primarily focus on the RAG settings and their performance. In contrast, our work investigates RAG systems under adversarial conditions, where the context may be intentionally crafted to mislead the LLM, thereby exposing vulnerabilities at the intersection of retrieval and generation.
    --------------------------------------------------------------------------------------
    2.2 Prompt Framing in LLMs and RAG Systems
    Recent advancements in LLMs have demonstrated impressive performance across various natural language processing tasks. However, their sensitivity to prompt formulation remains a notable challenge [63, 65]. Minor modifications in the wording, structure, or even punctuation of prompts can cause outputs that are substantially different and often incorrect [32, 42, 49, 50, 57, 63, 65, 75]. Qiang et al. [63] demonstrated that fine-tuned LLMs suffer large performance drops when prompts are perturbed by synonyms or paraphrases, introducing Prompt Perturbation Consistency Learning to enforce prediction stability across such variants. In related work, Mao et al. [49] conducted a comprehensive study of prompt position and demonstrated that the location of prompt tokens in input text has a large effect on zero-shot and few-shot performance, with many widely used placements proving suboptimal.
    In the RAG setting, prompt effects also remain critical. Perçin et al. [57] investigated the robustness of RAG pipelines at the query level, illustrating that subtle prompt variations, such as redundant phrasing and shifts in formal tone, can considerably affect retrieval quality and overall accuracy. Complementing these findings, Hu et al. [32] analyzed prompt perturbations in RAG and proposed the Gradient Guided Prompt Perturbation, an adversarial method that Manuscript submitted to ACM
    steers model outputs toward incorrect answers, while also presenting a detection approach based on neuron activation patterns to mitigate these risks.
    In this paper, we extend this line of work by examining how different query framings interact with retrieval context in RAG systems focused on the health domain. We evaluate consistent, neutral, and inconsistent query styles to measure their influence on ground-truth alignment and robustness under adversarial retrieval conditions. This design captures the ways in which query framing amplifies or mitigates the effects of helpful and harmful evidence, offering a systematic view of prompt sensitivity in a high-stakes domain.
    2.3 Adversarial Attacks on Neural Retrieval Models and LLMs
    A basic RAG system comprises two primary components: a retriever, which retrieves and ranks relevant documents from an external corpus, and an LLM, which generates responses based on the retrieved context. Prior research has investigated the vulnerabilities of each component in isolation. In particular, recent studies have shown that dense retrieval and neural ranking models could serve as an attack vector, through which minor perturbations and manipulation to already existing malicious documents can significantly boost their ranking positions [5, 14, 43, 47, 76, 78]. For instance, Liu et al. [47] shows that a subtle word substitution-based attack can boost the ranking position of random documents in the retrieved list of documents for queries. Liu et al. [43] propose a trigger-based adversarial attack against neural ranking models that leverages a surrogate model to identify tokens highly influential to the model's ranking score. These 'trigger tokens' are then injected into target documents to exploit the surrogate model's vulnerabilities, thereby boosting the documents' rankings in the ranked list of documents given by the victim model. Bigdeli et al. [5] propose an embedding-based perturbation attack that shifts a target document's representation closer to that of the query to generate adversarial sentences that can successfully promote its ranking and deceive neural rankers. Another approach by Chen et al. [14] generates connection sentences between the query and the target document using the BART [39] language model and appends them to the target document to boost it among the top-k documents.
    Researchers have also introduced various attack strategies targeting LLMs, primarily through jailbreak attacks [9, 24, 41, 68, 77, 88] and prompt injection attacks [30, 46, 56, 58, 69]. Jailbreak attacks bypass the safety alignment of LLMs by crafting prompts designed to deceive the model into producing harmful or undesirable behaviors, disclosing sensitive information, or otherwise violating its intended safety constraints. Prompt injection attacks manipulate model behavior by embedding malicious instructions directly into the input prompt, often overriding or subverting the original task. For example, an attacker may insert a directive such as: 'Ignore the instructions above and do...' [58], causing the model to follow the injected command instead of the intended instructions. Although jailbreak and prompt injection attacks have demonstrated the ability to exploit vulnerabilities in LLMs and bypass their safety mechanisms, recent research has shown that many of these weaknesses can be mitigated through targeted defense strategies. Broadly, these defenses fall into two categories: model-level defenses and input-level defenses.
    Model-level defenses aim to make the LLM inherently more resistant to malicious prompts by refining its internal decision-making and alignment mechanisms. Notable approaches include fine-tuning methods such as Direct Preference Optimization (DPO) [7, 12, 13, 60, 64, 81] which optimize the model's responses to better align with human-preferred safe behaviors, as well as other alignment-enhancement techniques [11, 38, 59, 66, 70, 82, 85] that reinforce policy adherence, increase refusal consistency, or provide provable robustness guarantees. Input-level defenses focus on intercepting and neutralizing malicious instructions before they are processed by the LLM. These include prompt filtering and classification systems that detect adversarial intent and block harmful requests at inference time [3, 36, 37, 67]. Such systems can identify suspicious patterns, injection-like structures, or known exploit phrases without significantly Manuscript submitted to ACM
    degrading the model's usability. Together, these defense strategies have significantly improved the robustness of LLMs against prompt injection and jailbreak attacks, making it more difficult for adversaries to induce harmful behaviors or override safety constraints.
    2.4 Adversarial Attacks on RAG Systems
    With the emergence of the RAG paradigm, researchers have investigated its vulnerabilities to poisoning attacks that target the generator phase by either promoting malicious documents into the retrieved context used for grounding and answer generation or by introducing adversarial prompt perturbations [10, 15, 17, 32, 52, 80, 86, 89].
[end on page 6]
```

### Document #6 — Distilling a Small Utility-Based Passage Selector to Enhance Retrieval-Augmented Generation
- `doc_id`: `doc::5f5181c36d374a038646a72443b62fa7`
- **Last Updated:** 2025-11-04T14:30:03.463930+00:00
- **Context:** This research enhances RAG systems by introducing efficient utility-based selection mechanisms.
- **Content fetched inside document:**
```text
[start on page 2]
    1 Introduction
    Figure 1: Different answer generation performance (generator: Llama-3.1-8B-Instruct) directly with different top𝑘 retrieval results (retriever: BM25).
    Generation Performance with Different Top-k Retrieval results
    NQ-EM NQ-F1 HotpotQA-EM HotpotQA-F1

    Retrieval-augmented generation (RAG) leverages retrieved information as external knowledge to empower large language models (LLMs) to answer questions. The criterion for measuring whether a result is helpful for RAG has shifted from relevance to utility [14, 30, 42, 44]. Relevance typically focuses on the topical matching between a query and retrieved passages [28, 29]. Utility, in contrast, emphasizes the usefulness of a passage in facilitating the generation of an accurate and comprehensive answer to the question [43]. Empirical results demonstrate that using retrieval results judged as having utility by LLMs in RAG can enhance the quality of subsequent answer generation [42, 44].

    Due to the high computation cost, using LLMs for utility judgments usually takes 10 to 20 passages as context [42, 44]. This is insufficient for weaker retrievers that rank useful passages at lower positions, and when complex questions require many useful documents to generate comprehensive answers. Although it is promising to scale utility judgments to a large number of candidate passages for RAG, using LLMs to do so is cost-prohibitive. Therefore, we propose to distill the utility judgment capability of LLMs to smaller models that are efficient to do so.

    In this paper, we focus on utility-based selection rather than ranking when distilling smaller models. There are two reasons: 1) Ke et al. [14] find that for effective RAG, the ranking of input passages is less critical than effectively filtering out low-quality passages. 2) The number of passages that should be selected for different questions can vary. As shown in Figure 1, the optimal number of passages used for simple questions (i.e., in NQ) and complex questions (i.e., in HotpotQA) is different. If we conduct utility ranking, a fixed threshold is usually used, which introduces a hyperparameter to tune and can be suboptimal. In contrast, utilitybased selection mechanisms can dynamically determine how many passages to retain. Consequently, our goal is to distill a small utilitybased selector from LLMs that are competent in zero-shot utilitybased selection.

    There have been several studies on distilling the zero-shot listwise ranking capability of LLMs (e.g., ChatGPT, GPT-4) into smaller efficient rankers [17, 22-24, 32], such as RankVicuna [22] and RankMistral [17]. The distilled student rankers ingest a large number of retrieval results using a sliding window-based bubble sort approach. The window is moved from lower to higher positions, popping the most relevant results to the head, until all the results are ranked. This approach, however, cannot be applied to utility-based selection due to the inherent differences. State-of-the-art (SOTA) utility judgment methods are based on pseudo answers that are generated from a group of input documents [42, 44]. This requires the student model to also inherit the answer generation capability from the teacher model. Moreover, to ensure decent pseudo-answer quality, results that are more likely to be useful are needed, indicating that the initial passages fed to the model should be of high ranks. This argues for a dedicated approach for distilling small utilitybased passage selectors and utility judgments of a large number of passages.

    ---------------------- this was the passage that we matched to the query -------------
    To this end, we propose a distilling approach that jointly learn pseudo-answer generation and utility judgments from teacher LLMs. For utility judgments with the student selector on a long initial ranking list, we propose a sliding window method that moves from higher to lower positions. At each step, the selector generates pseudo answers based on the selected useful results, and slides to the next window, which is comprised of the so-far selected useful results and the unseen passages. New selected useful results will be prepended to the selected result pool, and duplicates in the pool will be deleted, maintaining an ordered list of selected useful results. This process is repeated until all the candidate results are judged. This process ensures that the final selected useful results are based on the information of the entire candidate results. It also incurs a smaller cost than the above-mentioned ranking distillation due to smaller overlap between windows.
    --------------------------------------------------------------------------------------

    Following the current works for relevance ranking distillation [17, 23, 32], we also utilize the dataset of 100k queries, sampled from the MS MARCO training set by [32] for training. We employ Qwen332B [34] as teacher model to generate both relevance ranking and utility-based selection outputs. These outputs are then distilled into Qwen3-1.7B, yielding RankQwen1 . 7 𝐵 (for relevance ranking) and UtilityQwen1 . 7 𝐵 (for utility-based selection). To evaluate RAG performance, we utilize two QA datasets from BEIR [35]: NQ [15], and HotpotQA [41], on two kinds of top-100 initial candidate passages retrieved by two retrievers, i.e., BM25 [26] and BGE-base-en-v1.5 [40]. Following extensive experimentation, we found that: (1) For simple questions, such as those in the NQ dataset, relevance ranking (by adjusting various thresholds) demonstrated no statistically significant difference in optimal answer generation performance compared to directly using utility-based selection. (2) However, for complex questions, exemplified by the HotpotQA dataset, relevance ranking proved insufficient; utility-based selection was more effective in helping large language models (LLMs) identify document sets pertinent to answering the query. (3) Our utility-based selection method adaptively determines the number of useful passages based on the query and the passages. A key consequence is that selecting fewer documents per query enables more unprocessed passages to be handled within each sliding window iteration. This results in fewer window iterations for utility-based selection compared to

    relevance ranking, dramatically reducing the computational cost of LLM inference. Using merely 30% of the computational time, this approach yields higher-quality passages and, consequently, superior answers. Additionally, we will release the Qwen3-32B relevance ranking and utility-based selection annotations for the 100k MS MARCO dataset, providing a high-quality dataset for future research in relevance ranking and utility-based selection.
[end on page 3]
```

## LLM Answers (14.00s - 20.00s)
**Tech:** OpenAI's GPT-5 (reasoning low)

**Result:**

No. Bigger context windows help, but they haven’t removed the need for retrieval-augmented generation (RAG).

What we know from recent studies:
- Cost and citations: Even when models can read lots of text, RAG is typically more cost‑efficient and better at surfacing sources, and many systems now let the model decide when to retrieve versus rely on its long context (hybrid use) [1].
- Real‑world tasks: In electronic health records, RAG matched or beat using the latest notes and came close to full‑context performance while using far fewer tokens—showing it remains competitive and efficient as contexts grow [4].
- Long narratives still strain models: Long‑context LLMs suffer from limits like “lost in the middle,” especially for 200k+ token narratives; RAG (and improved, stateful variants like ComoRAG) are needed to maintain and integrate evolving evidence over long stories [3].
- Practical constraints of stuffing context: Simply dumping many documents into context can hurt performance and add compute. New RAG designs either compress/organize knowledge or select only useful passages to ease context pressure while keeping quality high [2][6].

When long context may suffice: If the relevant information already sits neatly inside the available window and doesn’t require up‑to‑date knowledge, explicit retrieval may add little. But for accuracy, fresh knowledge, traceable citations, efficiency, and long, messy corpora, RAG (often in hybrid form) remains the preferred approach [1][4][3].


**Reasoning:** Synthesized claims: debate in 2023 with evidence that RAG is more cost‑efficient and better for citations and hybrids emerged [1]; EHR evaluation shows RAG nearly matches full‑context with far fewer tokens [4]; long-context limits and ‘lost in the middle’ motivate RAG/ComoRAG for narratives [3]; conventional RAG can overload context, so variants reduce context pressure [2] and utility-based selection improves efficiency and answer quality [6].

**Citations:**

[1] doc::688cfbc0abdc4520a73e219ac26aff41 — A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions

[2] doc::92acc4a2fba74f329db33189ec5cfef4 — PL-CA: A Parametric Legal Case Augmentation Framework

[3] doc::6f44119d59b4473aa1bb4b535ab6400f — ComoRAG: A Cognitive-Inspired Memory-Organized RAG for Stateful Long Narrative Reasoning

[4] doc::0cc9a576aa8a43b58ee25a1c3e4879bf — Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs

[6] doc::5f5181c36d374a038646a72443b62fa7 — Distilling a Small Utility-Based Passage Selector to Enhance Retrieval-Augmented Generation

# Query 5

## Result 
(latency retrieval stack is ~5.3s total)

**Original Query:** how are people trying to make RAG give more accurate and less made-up answers?

**LLM answer:** 

Researchers are improving RAG accuracy and reducing made-up answers with a mix of better retrieval, smarter generation, and verification:

- Make retrieval cleaner and more relevant: use domain-adaptive encoders, query reformulation, rerankers, and iterative retrieval that breaks complex questions into simpler sub-queries so the right passages are fetched first [3]. Many pipelines add a lightweight reranking step to promote the most relevant passages [1].

- Choose or combine embeddings wisely: retrieval quality hinges on embeddings, so some use multiple embedding models. Confident RAG runs RAG several times with different embeddings, then picks the answer with the highest confidence (metrics like self-certainty or Distributional Perplexity), yielding ~5–10% accuracy gains over vanilla RAG in studies [2].

- Fine-tune models for RAG: domain-specific fine-tuning (e.g., Bio/Sci variants) and RAG-specific instruction-tuning help models better use retrieved evidence and avoid generic guesses [1]. HIRAG trains three skills—filtering noise, combining sources, and RAG-specific reasoning with a "think before answering" approach—showing state-of-the-art gains across datasets [5]. Other methods like RAFT (training with distractors), EvidenceRAG (indexing for citation), and RankRAG (jointly optimizing ranking and generation) further improve grounding [5].

- Integrate retrieval and generation more tightly: joint retriever–generator training and architectures that inject retrieved facts during generation can reduce drift from evidence [3].

- Add verification and self-checks: self-critique (Self-RAG), penalizing ungrounded claims, external fact-checking, cross-document agreement checks, and requiring source citations reduce hallucinations and raise trust [3]. In multimodal RAG, multi-stage verification (self-consistency + Chain-of-Verification) cut hallucinations sharply while improving truthfulness compared to naïve RAG [4].

- Be aware of limits: RAG still depends on retrieval quality and corpus alignment; noisy or off-target passages can hurt answers without careful tuning [1].

**Citations:**

[1] doc::6371023da29b4bbe8242ffc5caf4a8cd — Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs

[2] doc::3b9c43d010984d4cb11233b5de905555 — Each to Their Own: Exploring the Optimal Embedding in RAG

[3] doc::688cfbc0abdc4520a73e219ac26aff41 — A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions

[4] doc::9a2709238ca846b68a832069a85e77d6 — Multi-Stage Verification-Centric Framework for Mitigating Hallucination in Multi-Modal RAG

[5] doc::b0610cc6134b401db0ea68a77096e883 — HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation


--------------------------------------------------------------------------------------------------------------------------------------

# RAG Pipeline

**Latency Breakdown:**

- Query optimizer: 4.93s (boots up Qdrant client in parallel)
- Retrieve: 0.80s
- Rerank: 0.73s
- Context expansion: 1.04 s
- LLM answer: 14–20 s

## Query Optimizer (4.93s)

**Tech:** GPT-4o-mini, structured responses

Generated queries:
- semantic: improving accuracy of retrieval-augmented generation models
- hybrid: methods to reduce hallucination in RAG systems

## Retrieve (0.80s)

**Tech:** Qdrant, LlamaIndex

**Results:**

Query 1 (semantic) top 20 for query: improving accuracy of retrieval-augmented generation models

[1] score=0.5958 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S4::C03::251104134926
  text: 2 Related Work Upgrading of retrieval modules. From the perspective of retrieval methods, some studies have enhanced the quality of context by employing multistage retrieval reasoning (Asai et al., 2023b) (Gan et al., 2024), while others have designed adaptive retrieval modules that allow models to adjust retrieval behavior according to different tasks (Jeong et al., 2024). In terms of question understanding, some studies have improved search queries by rewriting, decomposing, and disambiguating (Chan et al., 2024). After retrieving articles, incorporating a ranking module can significantly enhance the final generation outcome (Glass et al., 2022)(Ram et al., 2023). RankRAG effectively integrates the ranking module with the generation module (Yu et al., 2024). These approaches have effectively improved the quality of retrieved articles in RAG systems. However, there is no such thing as a perfect context, and the generative model needs to be capable of handling contexts in various situations.

[2] score=0.5936 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S6::C25::251104140038
  text: 4 Experiment 4.3.2 Confident RAG incorrect answers. Meanwhile, the combined effect of RAG and confidence filtering enhances the robustness, leading to significant improvements compared to vanilla LLMs. When employing the optimal confidence metric, all LLMs achieved an accuracy increase of nearly 10%, demonstrating the method's universality. In the experiments, when the number of embedding models N > 3, the accuracy improvement became limited, likely due to redundant or noisy retrievals introduced by additional models. At N=3, the method achieved an optimal balance between diversity and computational efficiency. Increasing the number of models further yields only marginal benefits.

[3] score=0.5885 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S3::C02::251104134926
  text: 1 Introduction by incorporating an information retrieval component. While LLMs often use in-context learning (Gao et al., 2024) for generation, practical issues such as low-quality or poorly ranked retrieved documents can hinder RAG's effectiveness. These challenges emphasize the need for instruction-tuning tailored to RAG tasks. Fine-tuning generative models specifically for RAG improves their ability to integrate retrieved information (Zhang et al., 2024) (Yu et al., 2024), resulting in more accurate and contextually relevant responses compared to generalpurpose models.

[4] score=0.5873 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C04::251104164223
  text: 5 Generation-time feedback 5.2 Retrieval-on-Demand via Feedback Signals LLM's internal knowledge. However, the judgment is solely based on LLM, and without context, they try to be overconfident (Xiong et al., 2024). FLARE (Jiang et al., 2023) retrieves the documents only if the token probability is below a predefined threshold and uses the last generated sentence as a query for retrieval (excluding the uncertain tokens) and generates the response until the next uncertain token or completion is done. However, these uncertain tokens are not equally important to trigger a retrieval round. Based on this, DRAD (Su et al., 2024a) uses an external module for hallucination detection on entities in the generated answer; if the answer contains hallucination, the retrieval is triggered. The last generated sentence (without a hallucinated entity) is used as a query for retrieval. However, the choice of the new query for retrieval relies on heuristic strategies. Since the model's information needs may extend beyond the last sentence or C

[5] score=0.5845 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S2::C01::251104140038
  text: Abstract Recently, as Large Language Models (LLMs) have fundamentally impacted various fields, the methods for incorporating up-to-date information into LLMs or adding external knowledge to construct domainspecific models have garnered wide attention. Retrieval-Augmented Generation (RAG), serving as an inference-time scaling method, is notable for its low cost and minimal effort for parameter tuning. However, due to heterogeneous training data and model architecture, the variant embedding models used in RAG exhibit different benefits across various areas, often leading to different similarity calculation results and, consequently, varying response quality from LLMs. To address this problem, we propose and examine two approaches to enhance RAG by combining the benefits of multiple embedding models, named Mixture-Embedding RAG and Confident RAG. Mixture-Embedding RAG simply sorts and selects retrievals from multiple embedding models based on standardized similarity; however, it does not outperform vanilla RAG. In contrast, Con

[6] score=0.5827 doc=docs_ingestor/docs/arxiv/2508.09874.pdf chunk=S8::C01::251104161352
  text: 7. Related Work Retrieval-Augmented Generation Retrieval-Augmented Generation (RAG) enhances language models by incorporating knowledge from external sources, with retrieval granularity ranging from documents [Chen et al., 2017] to passages [Guu et al., 2020, Izacard et al., 2023, Lewis et al., 2020] to tokens [He et al., 2021, Khandelwal et al., 2019, Min et al., 2022, Yogatama et al., 2021]. Tokenlevel retrieval achieves superior performance for rare patterns and out-of-domain scenarios but introduces substantial computation overhead during inference. While non-differentiable retrieval mechanisms prevent end-to-end optimization and memory token approaches [Chevalier et al., 2023] enable differentiable access but are limited to local contexts, Memory Decoder provides both differentiable optimization and full-dataset knowledge access without expensive retrieval operations or model-specific datastores.

[7] score=0.5775 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S6::C22::251104140038
  text: 4 Experiment 4.3.2 Confident RAG As shown in Figure 2, there exists a positive correlation between confidence and accuracy. Therefore, the Confident RAG method improves overall accuracy by integrating multiple embedding models to generate answers and selecting the highestconfidence results using the most effective metric. This process effectively filters out low-confidence

[8] score=0.5755 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S6::C19::251104140038
  text: 4 Experiment 4.3.1 Mixture-embedding RAG For general LLMs (e.g., Llama-3.1-8B and OLMo-2-1124-7B) without math fine-tuning, their internal math knowledge is limited, leading to lower accuracy in direct answer generation. In these cases, even noisy references retrieved by RAG are more reliable than the LLMs' own outputs, as RAG at least provides partially correct information. However, while the mixtureembedding RAG method may optimize the retrieval ranking process and improve the quality of the references, the general LLMs' capabilities prevent them from fully leveraging higher-quality references, resulting in performance similar to vanilla RAG. Additionally, if different embedding models return highly diverse references, directly combining the top-ranked documents may cause information overload or contextual confusion, negating the potential benefits of mixture-embedding method. Therefore, the performance of general LLMsmatches that of vanilla RAG rather than surpassing it.

[9] score=0.5707 doc=docs_ingestor/docs/arxiv/2509.14750.pdf chunk=S2::C05::251104184057
  text: 1. INTRODUCTION expert models can exhibit overconfidence in their own parametric knowledge, causing them to either bypass the retrieval process entirely or uncritically accept irrelevant search results during self-reflection phases. As shown in Figure 1, even when retrieval provides a low-quality document about 'respiratory distress syndrome' for a question on 'callus formation', a fine-tuned model might erroneously conclude its existing knowledge is sufficient and proceed to generate an incorrect answer. This overconfidence prevents the model from initiating further, more targeted searches, effectively trapping it in a state of hallucination supported by poor retrieval. This challenge is particularly acute in systems that utilize self-correction or reflection mechanisms but are hindered by an overconfident generator [3, 4].

[10] score=0.5696 doc=docs_ingestor/docs/arxiv/2509.14750.pdf chunk=S2::C01::251104184057
  text: 1. INTRODUCTION Retrieval-Augmented Generation (RAG) seeks to enhance Large Language Models (LLMs) by integrating external knowledge bases, thereby mitigating factual inaccuracies and hallucinations [1, 2]. This approach has garnered significant interest for its potential in domain-specific applications. However, the performance of RAG systems is heavily dependent on the quality of retrieved information, and existing methods often face critical challenges that degrade their effectiveness, as illustrated in Figure 1.

[11] score=0.5675 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S13::C01::251104164223
  text: Key Insights The external feedback signals help in reducing retrieval rounds. These signals can come from different sources, at the LLM level (e.g., token generation confidence), at the answer level ( e.g., uncertainty or hallucination), and at the context level (e.g., relevancy or sufficiency). However, these methods may still retrieve noisy or irrelevant context, and complexity assessment remains a challenge.

[12] score=0.5664 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S5::C12::251104134926
  text: 3 HIRAG 3.2 Training Strategies model excel in selecting relevant information, integrating it, and reasoning about it within document contexts.

[13] score=0.5661 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S5::C15::251104134926
  text: 3 HIRAG 3.2 Training Strategies iii. Distractor Documents. In practical RAG scenarios, not every retrieved document is useful. Introducing noisy documents in training is crucial for helping the model learn to distinguish relevant from irrelevant information, thereby improving its ability to handle noise and generate accurate responses.

[14] score=0.5641 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=SDOC::SUM::251104140038
  text: This paper presents two improved methods for Retrieval-Augmented Generation (RAG) in Large Language Models (LLMs): Mixture-Embedding RAG and Confident RAG. While the former does not outperform traditional RAG, the latter shows a notable performance increase, achieving around 10% improvement over vanilla LLMs. The research emphasizes how combining multiple embedding models can enhance response quality and reduce the issues of varying similarity calculations inherent in different model architectures. The findings highlight two main confidence metrics, self-certainty and Distributional Perplexity (DP), which are effective in improving LLM performance. The study concludes that while RAG is efficient, selecting optimal embedding models remains crucial for maximizing accuracy.

[15] score=0.5638 doc=docs_ingestor/docs/arxiv/2507.07634.pdf chunk=S5::C05::251104135156
  text: 2 Related Work Traditional RAG approaches. Early work in grounding generation with real world documents focused on end-to-end differentiable encoder-decoder pipeline REALM [Guu et al., 2020], which augments Masked-Language Modeling (MLM) with a latent retriever model, backpropagating through retrieval to learn both retriever and generator jointly. However, this approach incurs significant computational cost and has only been shown to work with relatively smaller models like T5 [Raffel et al., 2020]. Building on this, Lewis et al. [2020] proposed a general finetuning strategy, RAG-Token which demonstrated that join-training outperforms fixed dense retrieval and BM25.

[16] score=0.5636 doc=docs_ingestor/docs/arxiv/2508.09874.pdf chunk=S2::C03::251104161352
  text: 1. Introduction Retrieval-Augmented Generation (RAG) offers an alternative approach by enhancing model outputs with relevant retrieved information [Izacard et al., 2023, Lewis et al., 2020]. While this method preserves the original model parameters, it introduces substantial computation overhead during inference due to expensive nearest neighbor ( kNN ) searches across large datastores and extended context [He et al., 2021].

[17] score=0.5635 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S6::C08::251104164223
  text: 3 Query-level feedback 3.2 Generative Relevance Feedback (GRF) Feedback from Generated Answers. Beyond generating expansions, some methods use LLM-generated answers as implicit feedback. Generation-Augmented Retrieval (GAR) (Mao et al., 2021) generates answer-like contexts (titles, passages, summaries) using a model like BART (Lewis et al., 2020a), which are then concatenated to the query. However, this introduces risks of hallucination and irrelevant additions. To refine this idea, RRR (Arora et al., 2023) iteratively updates the query based on retrieval performance, using a feedback loop constrained by a document budget. LameR (Shen et al., 2024) first generates multiple answers, augments them with the query, and performs a second retrieval pass-effectively building a feedback loop from generation to retrieval. InteR (Feng et al., 2024) and Iter-RetGen (Shao et al., 2023) perform tighter integration between RAG and GAR by alternating between generation and retrieval for iterative refinement.

[18] score=0.5634 doc=docs_ingestor/docs/arxiv/2507.20059.pdf chunk=SDOC::SUM::251104151210
  text: This study evaluates the effectiveness of Retrieval-Augmented Generation (RAG) systems across diverse knowledge sources and tasks. Utilizing MASSIVEDS, a large-scale datastore, the authors found that retrieval mainly benefits smaller models while larger models exhibit diminishing returns. The analysis shows no single retrieval source consistently outperforms others, highlighting the necessity for adaptive retrieval strategies. Furthermore, efforts such as reranking and query routing have limited improvement on overall performance. Key challenges include inaccurate relevance estimation and training-inference mismatch, suggesting a need for improved integration of retrieval mechanisms with language models. The findings indicate a critical gap in deploying RAG systems effectively in real-world scenarios, particularly for larger models that increasingly internalize domain knowledge without retrieval support.

[19] score=0.5627 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S3::C04::251104140038
  text: 1 Introduction To address this research gap, we propose two methods for improving RAG by combining the benefits of multiple embedding models. The first method is named Mixture-Embedding RAG, which sorts the retrieved materials from multiple embedding models based on normalized similarity and selects the top K materials as final references. The second method is named Confident RAG, where we first utilize vanilla RAG to generate answers multiple times, each time employing a different embedding model and recording the associated confidence metrics, and then select the answer with the highest confidence level as the final response. By validating our approach using multiple LLMs and embedding models, we illustrate the superior performance and generalization of Confident RAG, even though MixtureEmbedding RAG may lose to vanilla RAG. The main contributions of this paper can be summarized as follows: We first point out that in RAG, different embedding models operate within their own prior domains. To leverage the strengths of variou

[20] score=0.5620 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S4::C04::251104134926
  text: 2 Related Work Training Methods for Generative Models. ChatQA (Liu et al., 2024) (Xu et al., 2024) enhances the model's zero-shot dialogue capabilities through synthetic data and a two-stage instruction fine-tuning approach. In terms of identifying noisy documents, RAFT (Zhang et al., 2024) improves the model's ability to recognize and disregard irrelevant information by introducing distractor documents and employing the Chain-ofThought (COT) method. In contrast, InstructRAG (Wei et al., 2024) achieves this by explicitly learning the denoising process. EvidenceRAG (Schimanski et al., 2024) introduces an indexing task to enhance the reliability and traceability of large language models (LLMs) in evidence-based question answering. However, the context is complex and variable, merely filtering out noise and finding relevant documents is insufficient. Our work, starting from complex context scenarios, proposes three progressive model capabilities and effectively enhances these capabilities using the "think before answering" stra

Query 2 (hybrid) top 20 for query: methods to reduce hallucination in RAG systems

[1] score=0.8650 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C41::251104142800
  text: 7 Challenges of RAG 7.5.4 Hallucination Reducing factual hallucinations remains a key focus. RAG inherently mitigates hallucinations by grounding outputs in retrieved evidence [82]. Training models to penalize ungrounded assertions and iterative retrieval within reasoning processes further enhance accuracy [90]. Self-check mechanisms (Self-RAG), where models critique and revise their outputs against retrieval results, significantly reduce hallucinated content [6]. External verification and fact-checking modules complement internal methods, collectively ensuring high factual reliability. For instance, RAG systems to cite sources significantly enhance their reliability by directly linking generated information to supporting evidence. This citation capability plays a crucial role in mitigating the common issue of hallucination, where generative models produce plausible yet inaccurate or fabricated information. By explicitly associating each factual statement with retrieved documents, RAG systems encourage transparency and verif

[2] score=0.5000 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S4::C01::251104174859
  text: 2 Related Works 2.1 Definitions of Hallucination The term hallucination has been used with varying scope across natural language generation tasks. Some studies emphasize factuality , describing hallucinations as outputs that contradict established facts, i.e., inconsistencies with world knowledge or external ground truth [31, 13]. Others highlight faithfulness , where hallucinations occur when generated responses deviate from the user instruction or a reference text, often producing plausible but ungrounded statements particularly in source-conditioned tasks such as summarization or question answering [26]. Beyond these two dimensions, researchers also note cases of incoherent or nonsensical text that cannot be clearly attributed to factuality or faithfulness criteria [14, 23].

[3] score=0.4216 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S4::C02::251104174859
  text: 2 Related Works 2.1 Definitions of Hallucination Alternative terms have also been introduced. Confabulation draws on psychology to describe fluent but fabricated content arising from model priors [9], while fabrication is preferred by some to avoid anthropomorphic connotations [1, 21]. More recently, Chakraborty et al. [3] propose a flexible definition tailored to deployment settings, defining a hallucination as a generated output that conflicts with constraints or deviates from desired behavior in actual deployment, while remaining syntactically plausible under the circumstance.

[4] score=0.3731 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S2::C06::251104174859
  text: 1 Introduction liable external sources and proprietary documents, into the prompt, RAG improves factuality and domain relevance. While effective against intrinsic hallucinations, RAG remains susceptible to extrinsic hallucinations, especially when retrieved evidence is ignored, misinterpreted, or insufficient [10].

[5] score=0.3285 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S10::C01::251104174859
  text: 7 Conclusion Hallucinations in RAG-based conversational agents remain a significant barrier to trustworthy deployment in real-world applications. We introduced MetaRAG , a metamorphic testing framework for hallucination detection in retrieval-augmented generation (RAG) that operates without requiring ground truth references or access to model internals. Our experiments show that MetaRAG achieves strong detection performance on a challenging proprietary dataset, aligning with prior benchmark studies. Beyond general reliability, MetaRAG's factoid-level localization also supports identity-aware deployment by surfacing unsupported claims in sensitive domains (e.g., healthcare, migration, labor). Looking ahead, we see MetaRAG as a step toward safer and fairer conversational AI, where hallucinations are not only detected but also connected to safeguards that protect users in identity-sensitive contexts. This connection to identity-aware AI ensures that hallucination detection does not treat all users as homogeneous but provides sa

[6] score=0.3149 doc=docs_ingestor/docs/arxiv/2509.09272.pdf chunk=S7::C02::251104174407
  text: LLMs in QA LLMs also face problems when it comes to domain specific QA or tasks where they are needed to recall factual information accurately instead of just probabilistically generating whatever comes next. Research has also explored different prompting techniques, like chain-of-thought prompting[24], and sampling based methods[23] to reduce hallucinations. Contemporary research increasingly explores strategies such as fine-tuning and retrieval augmentation to enhance LLM-based QA systems. Fine-tuning on domain-specific corpora (e.g., BioBERT for biomedical text [17], SciBERT for scientific text [18]) has been shown to sharpen model focus, reducing irrelevant or generic responses in specialized settings such as medical or legal QA. Retrieval-augmented architectures such as RAG [19] combine LLMs with external knowledge bases, to try to further mitigate issues of factual inaccuracy and enable real-time incorporation of new information. Building on RAG's ability to bridge parametric and non-parametric knowledge, many modern Q

[7] score=0.3054 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S2::C03::251104174859
  text: 1 Introduction adoption, raising concerns about user trust, regulatory compliance, and business risk [20]. Moreover, hallucinations are not uniformly risky: the same unsupported claim can differentially affect specific populations. In healthcare (e.g., pregnancy/trimester-specific contraindications), migration and asylum (e.g., protections for LGBTQ+ refugees), or labor rights (e.g., eligibility by status), ungrounded spans can cause disproportionate harm. Rather than treating users as homogeneous, hallucination detection methods should make such spans reviewable at the factoid level so downstream systems can apply identity-aware policies (e.g., stricter thresholds, forced citations, or escalation to a human) when the topic indicates elevated risk. This perspective connects hallucination detection to identity-aware deployment, where span-level evidence enables topic-conditioned safeguards that reduce disproportionate risk. Ji et al. [14] categorize hallucinations into two types: Intrinsic hallucination : fabricated or contra

[8] score=0.2441 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S9::C01::251104174859
  text: 6 Discussion 6.1 Practical Implications Integrating hallucination detection into enterprise RAG systems offers several advantages: Risk Mitigation : Early detection of unsupported answers mitigates the spread of misinformation in both customer-facing and internal applications.

[9] score=0.2385 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S9::C03::251104174859
  text: 6 Discussion 6.1 Practical Implications Operational Efficiency : Detecting hallucinations simultaneously with content delivery reduces the need for costly downstream human verification.

[10] score=0.2182 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=SDOC::SUM::251104174859
  text: This paper introduces MetaRAG, a framework designed to detect hallucinations in Retrieval-Augmented Generation (RAG) systems. By decomposing model outputs into factoids and applying metamorphic testing, MetaRAG assesses the factual consistency of claims using real-time, unsupervised methods. The framework addresses limitations of existing detection approaches by functioning in black-box settings, making it suitable for sensitive enterprise applications. The evaluation on a proprietary dataset demonstrates that MetaRAG effectively identifies hallucinations, enhancing the reliability of conversational agents and enabling identity-aware safeguards based on detected risks.

[11] score=0.2162 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S2::C07::251104174859
  text: 1 Introduction Detecting hallucinations is particularly challenging in real-world settings, where RAG-based chatbots must respond to queries about unseen, proprietary, or confidential content where gold-standard references are typically unavailable [19]. Many existing hallucination detection methods rely on gold-standard reference answers [23, 16], annotated datasets [33], or access to model internals such as hidden states or token log-probabilities [25, 6]. However, in enterprise settings, such internals are often inaccessible: many state-of-the-art LLMs (e.g., GPT-4, Claude) are proprietary and only accessible via APIs that expose the final output text but not intermediate computations, limiting the feasibility of these methods in practice [19].

[12] score=0.2027 doc=docs_ingestor/docs/arxiv/2507.20136.pdf chunk=S7::C05::251104151250
  text: 1 Introduction We demonstrate the effectiveness of our system on the KDD Cup 2025 Meta CRAG-MM challenge, and release our implementation to support reproducibility and future research on building reliable MM-RAG systems that reduce hallucination in real-world, egocentric scenarios.

[13] score=0.1773 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S5::C21::251104174859
  text: 3 MetaRAG: Methodology 3.6 Identity-Aware Safeguards for Deployment Escalation. If hallucinations persist above threshold in identitysensitive domains, the system may abstain, regenerate with a stricter prompt, or escalate to human review.

[14] score=0.1732 doc=docs_ingestor/docs/arxiv/2507.20136.pdf chunk=S13::C06::251104151250
  text: 4 Experiments 4.2 Single-source Augmentation missing rate but poor factual reliability. The RAG Agent baseline incorporates external retrieval via the official search API, enhancing access to relevant knowledge. This improves the accuracy to 27.88% and slightly reduces the missing rate to 9.62%. However, hallucination rate increases further to 62.50%, and the truthfulness score remains at -34.62%, showing that naive RAG without verification can introduce more misleading content. To further assess the role of verification, we remove key components from our pipeline to observe their individual effects. The w/o CoV & SelfConsistency variant disables both the self-consistency check and the Chain-of-Verification (CoV). As a result, the model becomes overly conservative, with a missing rate of 95.19% and very low accuracy (3.85%). While hallucination rate drops sharply to 0.96%, the overall coverage and utility of the model are severely degraded. A similar trend is observed in the w/o CoV variant, which enables self-consistency bu

[15] score=0.1608 doc=docs_ingestor/docs/arxiv/2509.09686.pdf chunk=S2::C03::251104180841
  text: 1. Introduction RAG enhances LLM performance by retrieving semantically relevant content from external sources using similarity-based search methods. By referencing this material during generation, RAG significantly reduces hallucinations and improves factual consistency. Its integration into modern LLM pipelines has made it a foundational technique for building robust AI systems, particularly in applications like chatbots, research assistants, and domain-specific tools.

[16] score=0.1368 doc=docs_ingestor/docs/arxiv/2508.09893.pdf chunk=S10::C01::251104161515
  text: 8 Discussion Throughout this work, we presented a multi-agent system that uses triplet-based knowledge graph construction and retrieval-augmented generation (RAG) to enable transparent, verifiable question-answering on a regulatory corpus. By delegating ingestion, triplet extraction, KG maintenance, and query orchestration to specialized agents, unstructured text becomes a structured data layer for precise retrieval. The synergy of KG and RAG provides high-confidence, explainable facts alongside fluent responses to the large language model, as Section 7 demonstrates through accurate section retrieval, factual correctness and navigational queries (Figure 3). Grounding answers with triplets reduces LLM hallucinations, and provenance links enable robust auditing.

[17] score=0.1233 doc=docs_ingestor/docs/arxiv/2507.20136.pdf chunk=S15::C01::251104151250
  text: 6 Conclusion In this work, we present our solution, a multi-stage, verificationcentric RAG framework tailored to the KDD Cup 2025 CRAG-MM challenge. Our system introduces several innovations, including lightweight query routing, dynamic retrieval filtering, dual-path generation, and a structured Chain-of-Verification process. These components work in concert to improve factual consistency, reduce hallucinations, and enhance the reliability of multimodal question answering. Despite encountering practical limitations related to fine-tuning and hardware constraints, our solution demonstrates strong performance in the competition. We believe our pipeline offers generalizable insights for deploying robust, egocentric RAG systems in real-world multimodal settings such as AR/XR environments and smart assistants.

[18] score=0.1142 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S4::C04::251104174859
  text: 2 Related Works 2.2 Hallucination Detection in LLMs More recent reference-free (unsupervised or zero-reference) methods aim to detect hallucinations without gold-standard labels by analyzing the model's own outputs. A prominent method is SelfCheckGPT [22], a zero-resource, black-box approach that queries the LLM multiple times with the same prompt and measures semantic consistency across responses. The intuition is that hallucinated content often leads to instability under stochastic re-generation; true facts remain stable, while fabricated ones diverge. Manakul et al. show that SelfCheckGPT achieves strong performance in sentence-level hallucination detection compared to gray-box methods, and emphasize that it requires no external database or access to model internals [22]. However, SelfCheckGPT may struggle when deterministic decoding or high model confidence leads to repeating the same incorrect output.

[19] score=0.0950 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S20::C09::251104142800
  text: 6 Evaluation of RAG Systems 6.3 Retrieval-Augmented Generation Assessment System RAGAS (Retrieval-Augmented Generation Assessment System) is an evaluation framework specifically designed for assessing and improving the factuality and grounding of RAG systems. Unlike conventional metrics that measure superficial linguistic overlap, RAGAS emphasizes the alignment between generated content and retrieved documents, providing explicit signals regarding factual correctness and attribution quality. By systematically measuring how well the generated outputs are supported by the retrieved evidence, RAGAS helps identify and penalize hallucinations-instances where the model generates plausible but unsupported statements. Consequently, employing RAGAS during model training or iterative fine-tuning guides RAG systems toward producing outputs firmly grounded in verifiable sources, substantially improving factual accuracy and reducing the incidence of hallucinated information.

[20] score=0.0840 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S5::C23::251104174859
  text: 3 MetaRAG: Methodology 3.6 Identity-Aware Safeguards for Deployment In this way, higher hallucination scores are systematically translated into stronger protective actions, with more conservative safeguards applied whenever queries touch on identity-sensitive contexts.

RRF Fusion top 40 for query: How are people trying to make RAG give more accurate and less made-up answers?

[1] score=0.0164 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S4::C03::251104134926
  text: 2 Related Work Upgrading of retrieval modules. From the perspective of retrieval methods, some studies have enhanced the quality of context by employing multistage retrieval reasoning (Asai et al., 2023b) (Gan et al., 2024), while others have designed adaptive retrieval modules that allow models to adjust retrieval behavior according to different tasks (Jeong et al., 2024). In terms of question understanding, some studies have improved search queries by rewriting, decomposing, and disambiguating (Chan et al., 2024). After retrieving articles, incorporating a ranking module can significantly enhance the final generation outcome (Glass et al., 2022)(Ram et al., 2023). RankRAG effectively integrates the ranking module with the generation module (Yu et al., 2024). These approaches have effectively improved the quality of retrieved articles in RAG systems. However, there is no such thing as a perfect context, and the generative model needs to be capable of handling contexts in various situations.

[2] score=0.0164 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C41::251104142800
  text: 7 Challenges of RAG 7.5.4 Hallucination Reducing factual hallucinations remains a key focus. RAG inherently mitigates hallucinations by grounding outputs in retrieved evidence [82]. Training models to penalize ungrounded assertions and iterative retrieval within reasoning processes further enhance accuracy [90]. Self-check mechanisms (Self-RAG), where models critique and revise their outputs against retrieval results, significantly reduce hallucinated content [6]. External verification and fact-checking modules complement internal methods, collectively ensuring high factual reliability. For instance, RAG systems to cite sources significantly enhance their reliability by directly linking generated information to supporting evidence. This citation capability plays a crucial role in mitigating the common issue of hallucination, where generative models produce plausible yet inaccurate or fabricated information. By explicitly associating each factual statement with retrieved documents, RAG systems encourage transparency and verif

[3] score=0.0161 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S6::C25::251104140038
  text: 4 Experiment 4.3.2 Confident RAG incorrect answers. Meanwhile, the combined effect of RAG and confidence filtering enhances the robustness, leading to significant improvements compared to vanilla LLMs. When employing the optimal confidence metric, all LLMs achieved an accuracy increase of nearly 10%, demonstrating the method's universality. In the experiments, when the number of embedding models N > 3, the accuracy improvement became limited, likely due to redundant or noisy retrievals introduced by additional models. At N=3, the method achieved an optimal balance between diversity and computational efficiency. Increasing the number of models further yields only marginal benefits.

[4] score=0.0161 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S4::C01::251104174859
  text: 2 Related Works 2.1 Definitions of Hallucination The term hallucination has been used with varying scope across natural language generation tasks. Some studies emphasize factuality , describing hallucinations as outputs that contradict established facts, i.e., inconsistencies with world knowledge or external ground truth [31, 13]. Others highlight faithfulness , where hallucinations occur when generated responses deviate from the user instruction or a reference text, often producing plausible but ungrounded statements particularly in source-conditioned tasks such as summarization or question answering [26]. Beyond these two dimensions, researchers also note cases of incoherent or nonsensical text that cannot be clearly attributed to factuality or faithfulness criteria [14, 23].

[5] score=0.0159 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S3::C02::251104134926
  text: 1 Introduction by incorporating an information retrieval component. While LLMs often use in-context learning (Gao et al., 2024) for generation, practical issues such as low-quality or poorly ranked retrieved documents can hinder RAG's effectiveness. These challenges emphasize the need for instruction-tuning tailored to RAG tasks. Fine-tuning generative models specifically for RAG improves their ability to integrate retrieved information (Zhang et al., 2024) (Yu et al., 2024), resulting in more accurate and contextually relevant responses compared to generalpurpose models.

[6] score=0.0159 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S4::C02::251104174859
  text: 2 Related Works 2.1 Definitions of Hallucination Alternative terms have also been introduced. Confabulation draws on psychology to describe fluent but fabricated content arising from model priors [9], while fabrication is preferred by some to avoid anthropomorphic connotations [1, 21]. More recently, Chakraborty et al. [3] propose a flexible definition tailored to deployment settings, defining a hallucination as a generated output that conflicts with constraints or deviates from desired behavior in actual deployment, while remaining syntactically plausible under the circumstance.

[7] score=0.0156 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S11::C04::251104164223
  text: 5 Generation-time feedback 5.2 Retrieval-on-Demand via Feedback Signals LLM's internal knowledge. However, the judgment is solely based on LLM, and without context, they try to be overconfident (Xiong et al., 2024). FLARE (Jiang et al., 2023) retrieves the documents only if the token probability is below a predefined threshold and uses the last generated sentence as a query for retrieval (excluding the uncertain tokens) and generates the response until the next uncertain token or completion is done. However, these uncertain tokens are not equally important to trigger a retrieval round. Based on this, DRAD (Su et al., 2024a) uses an external module for hallucination detection on entities in the generated answer; if the answer contains hallucination, the retrieval is triggered. The last generated sentence (without a hallucinated entity) is used as a query for retrieval. However, the choice of the new query for retrieval relies on heuristic strategies. Since the model's information needs may extend beyond the last sentence or C

[8] score=0.0156 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S2::C06::251104174859
  text: 1 Introduction liable external sources and proprietary documents, into the prompt, RAG improves factuality and domain relevance. While effective against intrinsic hallucinations, RAG remains susceptible to extrinsic hallucinations, especially when retrieved evidence is ignored, misinterpreted, or insufficient [10].

[9] score=0.0154 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S2::C01::251104140038
  text: Abstract Recently, as Large Language Models (LLMs) have fundamentally impacted various fields, the methods for incorporating up-to-date information into LLMs or adding external knowledge to construct domainspecific models have garnered wide attention. Retrieval-Augmented Generation (RAG), serving as an inference-time scaling method, is notable for its low cost and minimal effort for parameter tuning. However, due to heterogeneous training data and model architecture, the variant embedding models used in RAG exhibit different benefits across various areas, often leading to different similarity calculation results and, consequently, varying response quality from LLMs. To address this problem, we propose and examine two approaches to enhance RAG by combining the benefits of multiple embedding models, named Mixture-Embedding RAG and Confident RAG. Mixture-Embedding RAG simply sorts and selects retrievals from multiple embedding models based on standardized similarity; however, it does not outperform vanilla RAG. In contrast, Con

[10] score=0.0154 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S10::C01::251104174859
  text: 7 Conclusion Hallucinations in RAG-based conversational agents remain a significant barrier to trustworthy deployment in real-world applications. We introduced MetaRAG , a metamorphic testing framework for hallucination detection in retrieval-augmented generation (RAG) that operates without requiring ground truth references or access to model internals. Our experiments show that MetaRAG achieves strong detection performance on a challenging proprietary dataset, aligning with prior benchmark studies. Beyond general reliability, MetaRAG's factoid-level localization also supports identity-aware deployment by surfacing unsupported claims in sensitive domains (e.g., healthcare, migration, labor). Looking ahead, we see MetaRAG as a step toward safer and fairer conversational AI, where hallucinations are not only detected but also connected to safeguards that protect users in identity-sensitive contexts. This connection to identity-aware AI ensures that hallucination detection does not treat all users as homogeneous but provides sa

[11] score=0.0152 doc=docs_ingestor/docs/arxiv/2508.09874.pdf chunk=S8::C01::251104161352
  text: 7. Related Work Retrieval-Augmented Generation Retrieval-Augmented Generation (RAG) enhances language models by incorporating knowledge from external sources, with retrieval granularity ranging from documents [Chen et al., 2017] to passages [Guu et al., 2020, Izacard et al., 2023, Lewis et al., 2020] to tokens [He et al., 2021, Khandelwal et al., 2019, Min et al., 2022, Yogatama et al., 2021]. Tokenlevel retrieval achieves superior performance for rare patterns and out-of-domain scenarios but introduces substantial computation overhead during inference. While non-differentiable retrieval mechanisms prevent end-to-end optimization and memory token approaches [Chevalier et al., 2023] enable differentiable access but are limited to local contexts, Memory Decoder provides both differentiable optimization and full-dataset knowledge access without expensive retrieval operations or model-specific datastores.

[12] score=0.0152 doc=docs_ingestor/docs/arxiv/2509.09272.pdf chunk=S7::C02::251104174407
  text: LLMs in QA LLMs also face problems when it comes to domain specific QA or tasks where they are needed to recall factual information accurately instead of just probabilistically generating whatever comes next. Research has also explored different prompting techniques, like chain-of-thought prompting[24], and sampling based methods[23] to reduce hallucinations. Contemporary research increasingly explores strategies such as fine-tuning and retrieval augmentation to enhance LLM-based QA systems. Fine-tuning on domain-specific corpora (e.g., BioBERT for biomedical text [17], SciBERT for scientific text [18]) has been shown to sharpen model focus, reducing irrelevant or generic responses in specialized settings such as medical or legal QA. Retrieval-augmented architectures such as RAG [19] combine LLMs with external knowledge bases, to try to further mitigate issues of factual inaccuracy and enable real-time incorporation of new information. Building on RAG's ability to bridge parametric and non-parametric knowledge, many modern Q

[13] score=0.0149 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S6::C22::251104140038
  text: 4 Experiment 4.3.2 Confident RAG As shown in Figure 2, there exists a positive correlation between confidence and accuracy. Therefore, the Confident RAG method improves overall accuracy by integrating multiple embedding models to generate answers and selecting the highestconfidence results using the most effective metric. This process effectively filters out low-confidence

[14] score=0.0149 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S2::C03::251104174859
  text: 1 Introduction adoption, raising concerns about user trust, regulatory compliance, and business risk [20]. Moreover, hallucinations are not uniformly risky: the same unsupported claim can differentially affect specific populations. In healthcare (e.g., pregnancy/trimester-specific contraindications), migration and asylum (e.g., protections for LGBTQ+ refugees), or labor rights (e.g., eligibility by status), ungrounded spans can cause disproportionate harm. Rather than treating users as homogeneous, hallucination detection methods should make such spans reviewable at the factoid level so downstream systems can apply identity-aware policies (e.g., stricter thresholds, forced citations, or escalation to a human) when the topic indicates elevated risk. This perspective connects hallucination detection to identity-aware deployment, where span-level evidence enables topic-conditioned safeguards that reduce disproportionate risk. Ji et al. [14] categorize hallucinations into two types: Intrinsic hallucination : fabricated or contra

[15] score=0.0147 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S6::C19::251104140038
  text: 4 Experiment 4.3.1 Mixture-embedding RAG For general LLMs (e.g., Llama-3.1-8B and OLMo-2-1124-7B) without math fine-tuning, their internal math knowledge is limited, leading to lower accuracy in direct answer generation. In these cases, even noisy references retrieved by RAG are more reliable than the LLMs' own outputs, as RAG at least provides partially correct information. However, while the mixtureembedding RAG method may optimize the retrieval ranking process and improve the quality of the references, the general LLMs' capabilities prevent them from fully leveraging higher-quality references, resulting in performance similar to vanilla RAG. Additionally, if different embedding models return highly diverse references, directly combining the top-ranked documents may cause information overload or contextual confusion, negating the potential benefits of mixture-embedding method. Therefore, the performance of general LLMsmatches that of vanilla RAG rather than surpassing it.

[16] score=0.0147 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S9::C01::251104174859
  text: 6 Discussion 6.1 Practical Implications Integrating hallucination detection into enterprise RAG systems offers several advantages: Risk Mitigation : Early detection of unsupported answers mitigates the spread of misinformation in both customer-facing and internal applications.

[17] score=0.0145 doc=docs_ingestor/docs/arxiv/2509.14750.pdf chunk=S2::C05::251104184057
  text: 1. INTRODUCTION expert models can exhibit overconfidence in their own parametric knowledge, causing them to either bypass the retrieval process entirely or uncritically accept irrelevant search results during self-reflection phases. As shown in Figure 1, even when retrieval provides a low-quality document about 'respiratory distress syndrome' for a question on 'callus formation', a fine-tuned model might erroneously conclude its existing knowledge is sufficient and proceed to generate an incorrect answer. This overconfidence prevents the model from initiating further, more targeted searches, effectively trapping it in a state of hallucination supported by poor retrieval. This challenge is particularly acute in systems that utilize self-correction or reflection mechanisms but are hindered by an overconfident generator [3, 4].

[18] score=0.0145 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S9::C03::251104174859
  text: 6 Discussion 6.1 Practical Implications Operational Efficiency : Detecting hallucinations simultaneously with content delivery reduces the need for costly downstream human verification.

[19] score=0.0143 doc=docs_ingestor/docs/arxiv/2509.14750.pdf chunk=S2::C01::251104184057
  text: 1. INTRODUCTION Retrieval-Augmented Generation (RAG) seeks to enhance Large Language Models (LLMs) by integrating external knowledge bases, thereby mitigating factual inaccuracies and hallucinations [1, 2]. This approach has garnered significant interest for its potential in domain-specific applications. However, the performance of RAG systems is heavily dependent on the quality of retrieved information, and existing methods often face critical challenges that degrade their effectiveness, as illustrated in Figure 1.

[20] score=0.0143 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=SDOC::SUM::251104174859
  text: This paper introduces MetaRAG, a framework designed to detect hallucinations in Retrieval-Augmented Generation (RAG) systems. By decomposing model outputs into factoids and applying metamorphic testing, MetaRAG assesses the factual consistency of claims using real-time, unsupervised methods. The framework addresses limitations of existing detection approaches by functioning in black-box settings, making it suitable for sensitive enterprise applications. The evaluation on a proprietary dataset demonstrates that MetaRAG effectively identifies hallucinations, enhancing the reliability of conversational agents and enabling identity-aware safeguards based on detected risks.

[21] score=0.0141 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S13::C01::251104164223
  text: Key Insights The external feedback signals help in reducing retrieval rounds. These signals can come from different sources, at the LLM level (e.g., token generation confidence), at the answer level ( e.g., uncertainty or hallucination), and at the context level (e.g., relevancy or sufficiency). However, these methods may still retrieve noisy or irrelevant context, and complexity assessment remains a challenge.

[22] score=0.0141 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S2::C07::251104174859
  text: 1 Introduction Detecting hallucinations is particularly challenging in real-world settings, where RAG-based chatbots must respond to queries about unseen, proprietary, or confidential content where gold-standard references are typically unavailable
  
[19]. Many existing hallucination detection methods rely on gold-standard reference answers [23, 16], annotated datasets [33], or access to model internals such as hidden states or token log-probabilities [25, 6]. However, in enterprise settings, such internals are often inaccessible: many state-of-the-art LLMs (e.g., GPT-4, Claude) are proprietary and only accessible via APIs that expose the final output text but not intermediate computations, limiting the feasibility of these methods in practice [19].

[23] score=0.0139 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S5::C12::251104134926
  text: 3 HIRAG 3.2 Training Strategies model excel in selecting relevant information, integrating it, and reasoning about it within document contexts.

[24] score=0.0139 doc=docs_ingestor/docs/arxiv/2507.20136.pdf chunk=S7::C05::251104151250
  text: 1 Introduction We demonstrate the effectiveness of our system on the KDD Cup 2025 Meta CRAG-MM challenge, and release our implementation to support reproducibility and future research on building reliable MM-RAG systems that reduce hallucination in real-world, egocentric scenarios.

[25] score=0.0137 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S5::C15::251104134926
  text: 3 HIRAG 3.2 Training Strategies iii. Distractor Documents. In practical RAG scenarios, not every retrieved document is useful. Introducing noisy documents in training is crucial for helping the model learn to distinguish relevant from irrelevant information, thereby improving its ability to handle noise and generate accurate responses.

[26] score=0.0137 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S5::C21::251104174859
  text: 3 MetaRAG: Methodology 3.6 Identity-Aware Safeguards for Deployment Escalation. If hallucinations persist above threshold in identitysensitive domains, the system may abstain, regenerate with a stricter prompt, or escalate to human review.

[27] score=0.0135 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=SDOC::SUM::251104140038
  text: This paper presents two improved methods for Retrieval-Augmented Generation (RAG) in Large Language Models (LLMs): Mixture-Embedding RAG and Confident RAG. While the former does not outperform traditional RAG, the latter shows a notable performance increase, achieving around 10% improvement over vanilla LLMs. The research emphasizes how combining multiple embedding models can enhance response quality and reduce the issues of varying similarity calculations inherent in different model architectures. The findings highlight two main confidence metrics, self-certainty and Distributional Perplexity (DP), which are effective in improving LLM performance. The study concludes that while RAG is efficient, selecting optimal embedding models remains crucial for maximizing accuracy.

[28] score=0.0135 doc=docs_ingestor/docs/arxiv/2507.20136.pdf chunk=S13::C06::251104151250
  text: 4 Experiments 4.2 Single-source Augmentation missing rate but poor factual reliability. The RAG Agent baseline incorporates external retrieval via the official search API, enhancing access to relevant knowledge. This improves the accuracy to 27.88% and slightly reduces the missing rate to 9.62%. However, hallucination rate increases further to 62.50%, and the truthfulness score remains at -34.62%, showing that naive RAG without verification can introduce more misleading content. To further assess the role of verification, we remove key components from our pipeline to observe their individual effects. The w/o CoV & SelfConsistency variant disables both the self-consistency check and the Chain-of-Verification (CoV). As a result, the model becomes overly conservative, with a missing rate of 95.19% and very low accuracy (3.85%). While hallucination rate drops sharply to 0.96%, the overall coverage and utility of the model are severely degraded. A similar trend is observed in the w/o CoV variant, which enables self-consistency bu

[29] score=0.0133 doc=docs_ingestor/docs/arxiv/2507.07634.pdf chunk=S5::C05::251104135156
  text: 2 Related Work Traditional RAG approaches. Early work in grounding generation with real world documents focused on end-to-end differentiable encoder-decoder pipeline REALM [Guu et al., 2020], which augments Masked-Language Modeling (MLM) with a latent retriever model, backpropagating through retrieval to learn both retriever and generator jointly. However, this approach incurs significant computational cost and has only been shown to work with relatively smaller models like T5 [Raffel et al., 2020]. Building on this, Lewis et al. [2020] proposed a general finetuning strategy, RAG-Token which demonstrated that join-training outperforms fixed dense retrieval and BM25.

[30] score=0.0133 doc=docs_ingestor/docs/arxiv/2509.09686.pdf chunk=S2::C03::251104180841
  text: 1. Introduction RAG enhances LLM performance by retrieving semantically relevant content from external sources using similarity-based search methods. By referencing this material during generation, RAG significantly reduces hallucinations and improves factual consistency. Its integration into modern LLM pipelines has made it a foundational technique for building robust AI systems, particularly in applications like chatbots, research assistants, and domain-specific tools.

[31] score=0.0132 doc=docs_ingestor/docs/arxiv/2508.09874.pdf chunk=S2::C03::251104161352
  text: 1. Introduction Retrieval-Augmented Generation (RAG) offers an alternative approach by enhancing model outputs with relevant retrieved information [Izacard et al., 2023, Lewis et al., 2020]. While this method preserves the original model parameters, it introduces substantial computation overhead during inference due to expensive nearest neighbor ( kNN ) searches across large datastores and extended context [He et al., 2021].

[32] score=0.0132 doc=docs_ingestor/docs/arxiv/2508.09893.pdf chunk=S10::C01::251104161515
  text: 8 Discussion Throughout this work, we presented a multi-agent system that uses triplet-based knowledge graph construction and retrieval-augmented generation (RAG) to enable transparent, verifiable question-answering on a regulatory corpus. By delegating ingestion, triplet extraction, KG maintenance, and query orchestration to specialized agents, unstructured text becomes a structured data layer for precise retrieval. The synergy of KG and RAG provides high-confidence, explainable facts alongside fluent responses to the large language model, as Section 7 demonstrates through accurate section retrieval, factual correctness and navigational queries (Figure 3). Grounding answers with triplets reduces LLM hallucinations, and provenance links enable robust auditing.

[33] score=0.0130 doc=docs_ingestor/docs/arxiv/2508.15437.pdf chunk=S6::C08::251104164223
  text: 3 Query-level feedback 3.2 Generative Relevance Feedback (GRF) Feedback from Generated Answers. Beyond generating expansions, some methods use LLM-generated answers as implicit feedback. Generation-Augmented Retrieval (GAR) (Mao et al., 2021) generates answer-like contexts (titles, passages, summaries) using a model like BART (Lewis et al., 2020a), which are then concatenated to the query. However, this introduces risks of hallucination and irrelevant additions. To refine this idea, RRR (Arora et al., 2023) iteratively updates the query based on retrieval performance, using a feedback loop constrained by a document budget. LameR (Shen et al., 2024) first generates multiple answers, augments them with the query, and performs a second retrieval pass-effectively building a feedback loop from generation to retrieval. InteR (Feng et al., 2024) and Iter-RetGen (Shao et al., 2023) perform tighter integration between RAG and GAR by alternating between generation and retrieval for iterative refinement.

[34] score=0.0130 doc=docs_ingestor/docs/arxiv/2507.20136.pdf chunk=S15::C01::251104151250
  text: 6 Conclusion In this work, we present our solution, a multi-stage, verificationcentric RAG framework tailored to the KDD Cup 2025 CRAG-MM challenge. Our system introduces several innovations, including lightweight query routing, dynamic retrieval filtering, dual-path generation, and a structured Chain-of-Verification process. These components work in concert to improve factual consistency, reduce hallucinations, and enhance the reliability of multimodal question answering. Despite encountering practical limitations related to fine-tuning and hardware constraints, our solution demonstrates strong performance in the competition. We believe our pipeline offers generalizable insights for deploying robust, egocentric RAG systems in real-world multimodal settings such as AR/XR environments and smart assistants.

[35] score=0.0128 doc=docs_ingestor/docs/arxiv/2507.20059.pdf chunk=SDOC::SUM::251104151210
  text: This study evaluates the effectiveness of Retrieval-Augmented Generation (RAG) systems across diverse knowledge sources and tasks. Utilizing MASSIVEDS, a large-scale datastore, the authors found that retrieval mainly benefits smaller models while larger models exhibit diminishing returns. The analysis shows no single retrieval source consistently outperforms others, highlighting the necessity for adaptive retrieval strategies. Furthermore, efforts such as reranking and query routing have limited improvement on overall performance. Key challenges include inaccurate relevance estimation and training-inference mismatch, suggesting a need for improved integration of retrieval mechanisms with language models. The findings indicate a critical gap in deploying RAG systems effectively in real-world scenarios, particularly for larger models that increasingly internalize domain knowledge without retrieval support.

[36] score=0.0128 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S4::C04::251104174859
  text: 2 Related Works 2.2 Hallucination Detection in LLMs More recent reference-free (unsupervised or zero-reference) methods aim to detect hallucinations without gold-standard labels by analyzing the model's own outputs. A prominent method is SelfCheckGPT [22], a zero-resource, black-box approach that queries the LLM multiple times with the same prompt and measures semantic consistency across responses. The intuition is that hallucinated content often leads to instability under stochastic re-generation; true facts remain stable, while fabricated ones diverge. Manakul et al. show that SelfCheckGPT achieves strong performance in sentence-level hallucination detection compared to gray-box methods, and emphasize that it requires no external database or access to model internals [22]. However, SelfCheckGPT may struggle when deterministic decoding or high model confidence leads to repeating the same incorrect output.

[37] score=0.0127 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S3::C04::251104140038
  text: 1 Introduction To address this research gap, we propose two methods for improving RAG by combining the benefits of multiple embedding models. The first method is named Mixture-Embedding RAG, which sorts the retrieved materials from multiple embedding models based on normalized similarity and selects the top K materials as final references. The second method is named Confident RAG, where we first utilize vanilla RAG to generate answers multiple times, each time employing a different embedding model and recording the associated confidence metrics, and then select the answer with the highest confidence level as the final response. By validating our approach using multiple LLMs and embedding models, we illustrate the superior performance and generalization of Confident RAG, even though MixtureEmbedding RAG may lose to vanilla RAG. The main contributions of this paper can be summarized as follows: We first point out that in RAG, different embedding models operate within their own prior domains. To leverage the strengths of variou

[38] score=0.0127 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S20::C09::251104142800
  text: 6 Evaluation of RAG Systems 6.3 Retrieval-Augmented Generation Assessment System RAGAS (Retrieval-Augmented Generation Assessment System) is an evaluation framework specifically designed for assessing and improving the factuality and grounding of RAG systems. Unlike conventional metrics that measure superficial linguistic overlap, RAGAS emphasizes the alignment between generated content and retrieved documents, providing explicit signals regarding factual correctness and attribution quality. By systematically measuring how well the generated outputs are supported by the retrieved evidence, RAGAS helps identify and penalize hallucinations-instances where the model generates plausible but unsupported statements. Consequently, employing RAGAS during model training or iterative fine-tuning guides RAG systems toward producing outputs firmly grounded in verifiable sources, substantially improving factual accuracy and reducing the incidence of hallucinated information.

[39] score=0.0125 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S4::C04::251104134926
  text: 2 Related Work Training Methods for Generative Models. ChatQA (Liu et al., 2024) (Xu et al., 2024) enhances the model's zero-shot dialogue capabilities through synthetic data and a two-stage instruction fine-tuning approach. In terms of identifying noisy documents, RAFT (Zhang et al., 2024) improves the model's ability to recognize and disregard irrelevant information by introducing distractor documents and employing the Chain-ofThought (COT) method. In contrast, InstructRAG (Wei et al., 2024) achieves this by explicitly learning the denoising process. EvidenceRAG (Schimanski et al., 2024) introduces an indexing task to enhance the reliability and traceability of large language models (LLMs) in evidence-based question answering. However, the context is complex and variable, merely filtering out noise and finding relevant documents is insufficient. Our work, starting from complex context scenarios, proposes three progressive model capabilities and effectively enhances these capabilities using the "think before answering" stra

[40] score=0.0125 doc=docs_ingestor/docs/arxiv/2509.09360.pdf chunk=S5::C23::251104174859
  text: 3 MetaRAG: Methodology 3.6 Identity-Aware Safeguards for Deployment In this way, higher hallucination scores are systematically translated into stronger protective actions, with more conservative safeguards applied whenever queries touch on identity-sensitive contexts.

## Rerank (0.74s)

**Tech:** Cohere API

**Results**

Rerank summary:
- strategy=cohere
- model=rerank-english-v3.0
- candidates=40
- eligible_above_threshold=40
- kept=6 (threshold=0)

Reranked Relevant (6/40 kept ≥ 0) top 6 for query: How are people trying to make RAG give more accurate and less made-up answers?

[1] score=0.9657 doc=docs_ingestor/docs/arxiv/2509.09272.pdf chunk=S7::C02::251104174407
  text: LLMs in QA LLMs also face problems when it comes to domain specific QA or tasks where they are needed to recall factual information accurately instead of just probabilistically generating whatever comes next. Research has also explored different prompting techniques, like chain-of-thought prompting[24], and sampling based methods[23] to reduce hallucinations. Contemporary research increasingly explores strategies such as fine-tuning and retrieval augmentation to enhance LLM-based QA systems. Fine-tuning on domain-specific corpora (e.g., BioBERT for biomedical text [17], SciBERT for scientific text [18]) has been shown to sharpen model focus, reducing irrelevant or generic responses in specialized settings such as medical or legal QA. Retrieval-augmented architectures such as RAG [19] combine LLMs with external knowledge bases, to try to further mitigate issues of factual inaccuracy and enable real-time incorporation of new information. Building on RAG's ability to bridge parametric and non-parametric knowledge, many modern Q

[2] score=0.8953 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S3::C04::251104140038
  text: 1 Introduction To address this research gap, we propose two methods for improving RAG by combining the benefits of multiple embedding models. The first method is named Mixture-Embedding RAG, which sorts the retrieved materials from multiple embedding models based on normalized similarity and selects the top K materials as final references. The second method is named Confident RAG, where we first utilize vanilla RAG to generate answers multiple times, each time employing a different embedding model and recording the associated confidence metrics, and then select the answer with the highest confidence level as the final response. By validating our approach using multiple LLMs and embedding models, we illustrate the superior performance and generalization of Confident RAG, even though MixtureEmbedding RAG may lose to vanilla RAG. The main contributions of this paper can be summarized as follows: We first point out that in RAG, different embedding models operate within their own prior domains. To leverage the strengths of variou

[3] score=0.8864 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C41::251104142800
  text: 7 Challenges of RAG 7.5.4 Hallucination Reducing factual hallucinations remains a key focus. RAG inherently mitigates hallucinations by grounding outputs in retrieved evidence [82]. Training models to penalize ungrounded assertions and iterative retrieval within reasoning processes further enhance accuracy [90]. Self-check mechanisms (Self-RAG), where models critique and revise their outputs against retrieval results, significantly reduce hallucinated content [6]. External verification and fact-checking modules complement internal methods, collectively ensuring high factual reliability. For instance, RAG systems to cite sources significantly enhance their reliability by directly linking generated information to supporting evidence. This citation capability plays a crucial role in mitigating the common issue of hallucination, where generative models produce plausible yet inaccurate or fabricated information. By explicitly associating each factual statement with retrieved documents, RAG systems encourage transparency and verif
[4] score=0.8142 doc=docs_ingestor/docs/arxiv/2507.20136.pdf chunk=S13::C06::251104151250
  text: 4 Experiments 4.2 Single-source Augmentation missing rate but poor factual reliability. The RAG Agent baseline incorporates external retrieval via the official search API, enhancing access to relevant knowledge. This improves the accuracy to 27.88% and slightly reduces the missing rate to 9.62%. However, hallucination rate increases further to 62.50%, and the truthfulness score remains at -34.62%, showing that naive RAG without verification can introduce more misleading content. To further assess the role of verification, we remove key components from our pipeline to observe their individual effects. The w/o CoV & SelfConsistency variant disables both the self-consistency check and the Chain-of-Verification (CoV). As a result, the model becomes overly conservative, with a missing rate of 95.19% and very low accuracy (3.85%). While hallucination rate drops sharply to 0.96%, the overall coverage and utility of the model are severely degraded. A similar trend is observed in the w/o CoV variant, which enables self-consistency bu

[5] score=0.8041 doc=docs_ingestor/docs/arxiv/2507.05714.pdf chunk=S3::C02::251104134926
  text: 1 Introduction by incorporating an information retrieval component. While LLMs often use in-context learning (Gao et al., 2024) for generation, practical issues such as low-quality or poorly ranked retrieved documents can hinder RAG's effectiveness. These challenges emphasize the need for instruction-tuning tailored to RAG tasks. Fine-tuning generative models specifically for RAG improves their ability to integrate retrieved information (Zhang et al., 2024) (Yu et al., 2024), resulting in more accurate and contextually relevant responses compared to generalpurpose models.

[6] score=0.6704 doc=docs_ingestor/docs/arxiv/2507.17442.pdf chunk=S6::C22::251104140038
  text: 4 Experiment 4.3.2 Confident RAG As shown in Figure 2, there exists a positive correlation between confidence and accuracy. Therefore, the Confident RAG method improves overall accuracy by integrating multiple embedding models to generate answers and selecting the highestconfidence results using the most effective metric. This process effectively filters out low-confidence

  
## Context Expansion (1.04s)

**Tech:** Redis (to fetch neighbors and additional informtion)

**Results:**

### Document #1 — Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs
- `doc_id`: `doc::6371023da29b4bbe8242ffc5caf4a8cd`
- **Last Updated:** 2025-11-04T17:44:07.300967+00:00
- **Context:** Comparative study on methodologies for integrating knowledge graphs in QA systems using LLMs.
- **Content fetched inside document:**
```text
[start on page 4]
    LLMs in QA
    
    The advent of LLMs has steered in a transformative era in NLP, particularly within the domain of QA. These models, pre-trained on massive corpora of diverse text, exhibit sophisticated capabilities in both natural language understanding and generation. Their proficiency in producing coherent, contextually relevant, and human-like responses to a broad spectrum of prompts makes them exceptionally well-suited for QA tasks, where delivering precise and informative answers is paramount. Recent advancements by models such as BERT [57] and ChatGPT [58], have significantly propelled the field forward. LLMs have demonstrated strong performance in open-domain QA scenarios-such as commonsense reasoning[20]-owing to their extensive embedded knowledge of the world. Moreover, their ability to comprehend and articulate responses to abstract or contextually nuanced queries and reasoning tasks [22] underscores their utility in addressing complex QA challenges that require deep semantic understanding. Despite their strengths, LLMs also pose challenges: they can exhibit contextual ambiguity or overconfidence in their outputs ('hallucinations')[21], and their substantial computational and memory requirements complicate deployment in resource-constrained environments.
    
    RAG, fine tuning in QA

    ---------------------- this was the passage that we matched to the query -------------
    LLMs also face problems when it comes to domain specific QA or tasks where they are needed to recall factual information accurately instead of just probabilistically generating whatever comes next. Research has also explored different prompting techniques, like chain-of-thought prompting[24], and sampling based methods[23] to reduce hallucinations. Contemporary research increasingly explores strategies such as fine-tuning and retrieval augmentation to enhance LLM-based QA systems. Fine-tuning on domain-specific corpora (e.g., BioBERT for biomedical text [17], SciBERT for scientific text [18]) has been shown to sharpen model focus, reducing irrelevant or generic responses in specialized settings such as medical or legal QA. Retrieval-augmented architectures such as RAG [19] combine LLMs with external knowledge bases, to try to further mitigate issues of factual inaccuracy and enable real-time incorporation of new information. Building on RAG's ability to bridge parametric and non-parametric knowledge, many modern QA pipelines introduce a lightweight re-ranking step [25] to sift through the retrieved contexts and promote passages that are most relevant to the query. However, RAG still faces several challenges. One key issue lies in the retrieval step itself-if the retriever fails to fetch relevant documents, the generator is left to hallucinate or provide incomplete answers. Moreover, integrating noisy or loosely relevant contexts can degrade response quality rather than enhance it, especially in high-stakes domains where precision is critical. RAG pipelines are also sensitive to the quality and domain alignment of the underlying knowledge base, and they often require extensive tuning to balance recall and precision effectively.
    --------------------------------------------------------------------------------------
[end on page 5]
```

### Document #2 — Each to Their Own: Exploring the Optimal Embedding in RAG
- `doc_id`: `doc::3b9c43d010984d4cb11233b5de905555`
- **Last Updated:** 2025-11-04T14:00:38.215399+00:00
- **Context:** Enhancing Large Language Models using Retrieval-Augmented Generation techniques.
- **Content fetched inside document:**
```text
[start on page 1]

    1 Introduction
    Large language models (LLMs) have recently accelerated the pace of transformation across multiple fields, including transportation (Lyu et al., 2025), arts (Zhao et al., 2025), and education (Gao et al., 2024), through various paradigms such as direct answer generation, training from scratch on different types of data, and fine-tuning on target domains. However, the hallucination problem (Henkel et al., 2024) associated with LLMs has confused people for a long time, stemming from multiple factors such as a lack of knowledge on the given prompt (Huang et al., 2025b) and a biased training process (Zhao, 2025).

    Serving as a highly efficient solution, RetrievalAugmented Generation (RAG) has been widely employed in constructing foundation models (Chen et al., 2024) and practical agents (Arslan et al., 2024). Compared to training methods like fine-tuning and prompt-tuning, its plug-and-play feature makes RAG an efficient, simple, and costeffective approach. The main paradigm of RAG involves first calculating the similarities between a question and chunks in an external knowledge corpus, followed by incorporating the top K relevant chunks into the prompt to guide the LLMs (Lewis et al., 2020).

    Despite the advantages of RAG, selecting the appropriate embedding models remains a crucial concern, as the quality of retrieved references directly influences the generation results of the LLM (Tu et al., 2025). Variations in training data and model architecture lead to different embedding models providing benefits across various domains. The differing similarity calculations across embedding models often leave researchers uncertain about how to choose the optimal one. Consequently, improving the accuracy of RAG from the perspective of embedding models continues to be an ongoing area of research.
    ---------------------- this was the passage that we matched to the query -------------
    To address this research gap, we propose two methods for improving RAG by combining the benefits of multiple embedding models. The first method is named Mixture-Embedding RAG, which sorts the retrieved materials from multiple embedding models based on normalized similarity and selects the top K materials as final references. The second method is named Confident RAG, where we first utilize vanilla RAG to generate answers multiple times, each time employing a different embedding model and recording the associated confidence metrics, and then select the answer with the highest confidence level as the final response. By validating our approach using multiple LLMs and embedding models, we illustrate the superior performance and generalization of Confident RAG, even though MixtureEmbedding RAG may lose to vanilla RAG. The main contributions of this paper can be summarized as follows:
    
    We first point out that in RAG, different embedding models operate within their own prior domains. To leverage the strengths of various embedding models, we propose and test two novel RAG methods: MixtureEmbedding RAG and Confident RAG. These methods effectively utilize the retrieved results from different embedding models to their fullest extent.
    --------------------------------------------------------------------------------------
    While Mixture-Embedding RAG performs similarly to vanilla RAG, the Confident RAG method exhibits superior performance compared to both the vanilla LLM and vanilla RAG, with average improvements of 9.9% and 4.9%, respectively, when using the best confidence metric. Additionally, we discuss the optimal number of embedding models for the Confident RAG method based on the results.

    Our results reveal two outstanding confidence metrics: self-certainty and Distributional Perplexity (DP), both showing average improvements of approximately 10% compared to the vanilla LLM. Specifically, among the LLMs examined, self-certainty achieves a maximum increase of 10.4%, while DP demonstrates a maximum increase of 12.4% compared to the vanilla LLM. The reasons behind the better performance of these two metrics are discussed based on their formulas.

    4 Experiment
    4.2.3 Results of Confident RAG method
    | | LLMModel | Mix-Embedding RAG.2 Embs | Mix-Embedding RAG.3 Embs | Mix-Embedding RAG.4 Embs | Mix-Embedding RAG.Avg | Mix-Embedding RAG.v.s. VanillaLLM | Mix-Embedding RAG.v.s. Vanilla RAG |
    |---:|:----------------------------------------|:---------------------------|:---------------------------|:---------------------------|:------------------------|:------------------------------------|:-------------------------------------|
    | 0 | Qwen2.5-Math-7B (Yang et al., 2024) | 74.20% | 76.00% | 74.80% | 75.0% | -0.2% ↓ | -5.5% ↓ |
    | 1 | Llama-3.1-8B (Grattafiori et al., 2024) | 20.90% | 19.60% | 22.80% | 21.1% | 4.5% ↑ | -0.2% ↓ |
    | 2 | OLMo-2-1124-7B (OLMo et al., 2024) | 26.20% | 26.80% | 26.40% | 26.5% | 5.5% ↑ | 0.5% ↑ |
    | 3 | Average | 40.4% | 40.8% | 41.3% | 40.9% | 3.3% ↑ | -1.7% ↓ |

    Table 2: Performance of Mixture-Embedding RAG: This table illustrates the performance when using between 2 and 4 different embedding models randomly (denoted as 2 Embs to 4 Embs), comparing the results with those of the vanilla LLM and vanilla RAG.
    researchers should seek a suitable trade-off based on their own condition when choosing the optimal N.

    4.3.1 Mixture-embedding RAG
    For general LLMs (e.g., Llama-3.1-8B and OLMo-2-1124-7B) without math fine-tuning, their internal math knowledge is limited, leading to lower accuracy in direct answer generation. In these cases, even noisy references retrieved by RAG are more reliable than the LLMs' own outputs, as RAG at least provides partially correct information. However, while the mixtureembedding RAG method may optimize the retrieval ranking process and improve the quality of the references, the general LLMs' capabilities prevent them from fully leveraging higher-quality references, resulting in performance similar to vanilla RAG. Additionally, if different embedding models return highly diverse references, directly combining the top-ranked documents may cause information overload or contextual confusion, negating the potential benefits of mixture-embedding method. Therefore, the performance of general LLMsmatches that of vanilla RAG rather than surpassing it.

    On the other hand, for the LLMs that have been fine-tuned based on math corpora, vanilla RAG may result in smaller improvements, and lower-quality references can lead to poorer answer performance. For these types of LLMs, the mixture-embedding method may introduce additional noise in mathematical contexts, resulting in lower accuracy compared to vanilla RAG. The decline in accuracy may be caused by several factors: (1) Mathematical symbols and formulas may vary drastically across embedding models, making similarity calculations unstable. (2) Different models may encode math terms differently, causing the top-ranked reference to be suboptimal. (3) An embedding model might incorrectly rank an irrelevant math material highly, while better references from other models are ignored. If the LLM generates hallucinated answers based on incorrect references, its performance can degrade below that of the vanilla LLM. (4) Information overload or contextual confusion may also occur, similar to what happens with general LLMs.

    ---------------------- this was the passage that we matched to the query -------------
    4.3.2 Confident RAG
    As shown in Figure 2, there exists a positive correlation between confidence and accuracy. Therefore, the Confident RAG method improves overall accuracy by integrating multiple embedding models to generate answers and selecting the highestconfidence results using the most effective metric. This process effectively filters out low-confidence
    --------------------------------------------------------------------------------------

    | | LLM | Emb. Model | AvgLogP | Self-certainty | Gini | Entropy | DP |
    |---:|:----------------------------------------|:------------------------|:--------------|:-----------------|:-------------|:--------------|:------------------|
    | 0 | | 1,2 | 82.0% | 85.0% | 81.8% | 82.8% | 83.6% |
    | 1 | | 1,3 | 79.4% | 83.4% | 79.2% | 79.6% | 81.0% |
    | 2 | | 1,4 | 81.4% | 84.6% | 81.6% | 81.8% | 82.2% |
    | 3 | | 2,3 | 81.8% | 84.4% | 81.4% | 81.8% | 82.4% |
    | 4 | | 2,4 | 81.4% | 83.6% | 81.0% | 81.4% | 82.2% |
    | 5 | | 3,4 | 79.2% | 82.2% | 79.8% | 79.8% | 80.6% |
    | 6 | | Avg (n=2) | 80.9% | 83.9% | 80.8% | 81.2% | 82.0% |
    | 7 | | 1,2,3 | 79.4% | 85.0% | 79.0% | 79.8% | 81.6% |
    | 8 | Qwen2.5-Math-7B (Yang et al., 2024) | 1,2,4 | 79.6% | 85.0% | 79.6% | 80.6% | 82.0% |
    | 9 | | 1,3,4 | 79.0% | 84.8% | 79.6% | 80.0% | 80.8% |
    | 10 | | 2,3,4 | 79.2% | 84.4% | 79.0% | 79.8% | 81.0% |
    | 11 | | Avg (n=3) | 79.3% | 84.8% | 79.3% | 80.1% | 81.4% |
    | 12 | | 1,2,3,4 | 78.2% | 84.8% | 78.4% | 79.2% | 80.6% |
    | 13 | | Avg (n=2,3,4) | 80.1% | 84.3% | 80.0% | 80.6% | 81.6% |
    | 14 | | v.s. Vanilla RAG | 0.4% ↓ | 3.8% ↑ | 0.4% ↓ | 0.1% ↑ | 1.2% ↑ |
    | 15 | | v.s. Vanilla LLM | 4.9% ↑ | 9.1% ↑ | 4.8% ↑ | 5.4% ↑ | 6.4% ↑ |
    | 16 | | 1,2 | 27.2% | 27.2% | 27.0% | 27.0% | 27.2% |
    | 17 | | 1,3 | 26.8% | 26.8% | 26.6% | 27.0% | 27.2% |
    | 18 | | 1,4 | 26.8% | 26.4% | 26.4% | 26.4% | 26.4% |
    | 19 | | 2,3 | 23.2% | 24.0% | 24.0% | 24.0% | 23.6% |
    | 20 | | 2,4 | 26.6% | 26.8% | 26.4% | 26.0% | 26.6% |
    | 21 | | 3,4 | 25.8% | 26.4% | 26.6% | 26.6% | 26.4% |
    | 22 | | Avg (n=2) | 26.1% 28.6% | 26.3% 29.2% | 26.2% 28.8% | 26.2% | 26.2% 29.4% |
    | 23 | Llama-3.1-8B (Grattafiori et al., 2024) | 1,2,3 1,2,4 | 28.8% | | | 29.4% | |
    | 24 | | | | 28.6% | 28.0% | 28.0% | 28.2% 27.6% |
    | 25 | | 1,3,4 | 27.4% | 27.4% | 27.2% | 27.6% | |
    | 26 | | 2,3,4 | 25.8% | 26.4% | 26.0% | 26.0% | 26.4% |
    | 27 | | Avg (n=3) | 27.7% 27.8% | 27.9% 27.6% | 27.5% 27.0% | 27.8% 27.4% | 27.9% 27.6% |
    | 28 | | 1,2,3,4 Avg (n=2,3,4) | 26.8% | 27.0% | 26.7% | 26.9% | 27.0% |
    | 29 | | | | | 5.5% ↑ | | |
    | 30 | | v.s. Vanilla RAG | 5.6% ↑ | 5.7% ↑ | 10.1% ↑ | 5.6% ↑ | 5.7% ↑ |
    | 31 | | v.s. Vanilla LLM 1,2 | 10.2% ↑ 31.0% | 10.4% ↑ 31.2% | 30.4% | 10.3% ↑ 31.2% | 10.4% ↑ 32.2% |
    | 32 | | 1,3 | 29.8% | 29.6% | 29.8% | 30.0% | 30.0% |
    | 33 | | 1,4 | 29.4% | 29.2% | 28.6% | 29.6% | 31.0% |
    | 34 | | 2,3 | 31.6% | 30.8% | 30.2% | 31.4% | 32.8% |
    | 35 | | 2,4 | 32.6% | 32.0% | 31.0% | 33.0% | 33.8% |
    | 36 | | 3,4 | 29.6% | 30.2% | 29.6% | 30.6% | 31.4% |
    | 37 | | Avg (n=2) | 30.7% | 30.5% | 29.9% | 31.0% | 31.9% |
    | 38 | OLMo-2-1124-7B (OLMo et al., 2024) | 1,2,3 | 32.6% | 32.0% | 31.4% | 32.2% | 33.2% |
    | 39 | | 1,2,4 | 32.8% | 32.8% | 31.6% | 33.2% | 34.8% |
    | 40 | | 1,3,4 | 31.2% | 31.2% | 30.6% | 32.0% | 34.6% |
    | 41 | | 2,3,4 | 32.6% | 32.2% | 30.8% | 33.6% | 36.6% |
    | 42 | | Avg (n=3) | 32.3% | 32.1% | 31.1% | 32.8% | 34.8% |
    | 43 | | 1,2,3,4 | 32.8% | 32.4% | 31.0% | 33.2% | 35.8% |
    | 44 | | Avg (n=2,3,4) | 31.5% | 31.2% | 30.5% | 31.8% | 33.3% |
    | 45 | | v.s. Vanilla RAG | 5.5% ↑ | 5.2% ↑ | 4.5% ↑ | 5.8% ↑ | 7.3% ↑ |
    | 46 | | LLM | 10.5% ↑ | | ↑ | ↑ | 12.3% ↑ |
    | 47 | | v.s. Vanilla | | 10.2% ↑ | 9.5% | 10.8% | |
    | 48 | | Avg (n=2) Avg (n=3) | 45.9% 46.4% | 46.9% 48.3% | 45.6% 46.0% | 46.1% 46.9% | 46.7% 48.0% 48.0% |
    | 49 | Average | Avg (n=4) Avg (n=2,3,4) | 46.3% | 48.3% | 45.5% | 46.6% | 47.3% |
    | 50 | | v.s. Vanilla RAG | 46.1% 3.5% ↑ | 47.5% 4.9% ↑ | 45.7% 3.2% ↑ | 46.4% 3.9% ↑ | 4.7% ↑ |
    | 51 | | v.s. Vanilla LLM | 8.5% ↑ | 9.9% ↑ | 8.1% ↑ | 8.8% ↑ | 9.7% ↑ |

    Table 3: Accuracy Comparison Across Multi-RAG with Different Embedding Models: Avg( n ) denotes the average accuracy across different combinations of n embedding models. Each line uses underline to indicate the best embedding combination within each LLM. Each row uses bold to signify the best metric for confidence evaluation. For performance comparison with Vanilla RAG, we use the average accuracy of all single embedding models as the baseline.

    incorrect answers. Meanwhile, the combined effect of RAG and confidence filtering enhances the robustness, leading to significant improvements compared to vanilla LLMs. When employing the optimal confidence metric, all LLMs achieved an accuracy increase of nearly 10%, demonstrating the method's universality. In the experiments, when the number of embedding models N > 3, the accuracy improvement became limited, likely due to redundant or noisy retrievals introduced by additional models. At N=3, the method achieved an optimal balance between diversity and computational efficiency. Increasing the number of models further yields only marginal benefits.

    Self-Certainty and DP outperform other metrics since they directly measure the concentration and divergence of the probability distribution. Specifically, Self-Certainty measures how far the predicted distribution deviates from uniform. By scaling the probabilities by | v | and taking the negative logarithm, it heavily penalizes uniform-like distributions, favoring sharp peaks. This makes it highly discriminative for high-confidence answers. Additionally, DP is an exponential version of entropy. The exponentiation amplifies differences in entropy, making it more sensitive to the sharpness of the distribution. Low DP values indicate tightly clustered high-probability tokens, which strongly correlate with correct answers. In contrast, other metrics are less sensitive because they either average out uncertainties (AvgLogP) or lack normalization across vocabularies (Gini). While entropy can be useful, it is linear and less discriminative compared to DP's exponential scaling. Therefore, Self-Certainty and DP are more sensitive to subtle variations in model confidence.
[end on page 9]
```

### Document #3 — A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions
- `doc_id`: `doc::688cfbc0abdc4520a73e219ac26aff41`
- **Last Updated:** 2025-11-04T14:28:00.715095+00:00
- **Context:** A review of RAG's evolution and its evolving role in knowledge-intensive NLP.
- **Content fetched inside document:**
```text
[start on page 23]

    7 Challenges of RAG
    7.4.3 Customer Support and Knowledge Bases
    Metrics for Evaluation : Success in customer support is evaluated through resolution rates and customer satisfaction metrics, requiring responses that not only provide factual accuracy but practical utility.
    Ethical considerations include transparency, data security, job impact, and fairness. RAG implementations also provide valuable feedback for continual improvement of internal support documentation based on real-time user interactions, thereby enhancing both system performance and documentation quality.

    7.5.1 Retrieval Quality
    Maintaining high retrieval relevance is critical for effective RAG. Strategies to improve retrieval quality include domain-adaptive training, advanced encoders, and query reformulation methods to address vocabulary mismatches [85]. Employing reranking models further boosts relevance by re-scoring initial retrieval results with deeper contextual analysis, enhancing accuracy at the expense of additional computation [4]. Iterative retrieval and chain-of-thought reasoning represent future directions, breaking down complex queries into simpler sub-queries, thus ensuring relevant information retrieval at each reasoning step [90].

    7.5.2 Latency
    RAG systems introduce latency due to retrieval processes. Solutions include using efficient nearest-neighbor search structures, such as HNSW graphs, which significantly speed up similarity searches [57]. Caching mechanisms, including multi-level and approximate embedding caches (e.g., RAGCache and Proximity cache), enable reuse of previously retrieved information, drastically reducing retrieval time [40, 8]. Adaptive retrieval methods dynamically balance retrieval complexity based on query difficulty, optimizing overall throughput and reducing latency.

    7.5.3 Model Integration
    Effective integration between retrieval and generation models remains essential. Methods include joint end-to-end training of retrievers and generators, enhancing mutual compatibility and performance [52]. Architectural integration techniques, such as RETRO's cross-attention mechanism, dynamically incorporate retrieved facts during generation [10]. Alternatively, prompt-based integration treats LLMs as black-boxes, conditioning on retrieved documents without architectural modifications. Future hybrid approaches involving reinforcement learning and selective retrieval aim to optimize when and how external knowledge is incorporated into generation processes.

    ---------------------- this was the passage that we matched to the query -------------
    7.5.4 Hallucination
    Reducing factual hallucinations remains a key focus. RAG inherently mitigates hallucinations by grounding outputs in retrieved evidence [82]. Training models to penalize ungrounded assertions and iterative retrieval within reasoning processes further enhance accuracy [90]. Self-check mechanisms (Self-RAG), where models critique and revise their outputs against retrieval results, significantly reduce hallucinated content [6]. External verification and fact-checking modules complement internal methods, collectively ensuring high factual reliability. For instance, RAG systems to cite sources significantly enhance their reliability by directly linking generated information to supporting evidence. This citation capability plays a crucial role in mitigating the common issue of hallucination, where generative models produce plausible yet inaccurate or fabricated information. By explicitly associating each factual statement with retrieved documents, RAG systems encourage transparency and verifiability, enabling users and downstream processes to quickly assess the accuracy and provenance of claims. Moreover, requiring the model to cite sources during generation inherently promotes grounding outputs in verified data, further reducing the risk of generating unsupported statements [82]. Thus, citation functionality not only enhances user trust but also fosters more disciplined, factually accurate generation, substantially decreasing the likelihood of hallucinated outputs.
    --------------------------------------------------------------------------------------

    7.5.5 Scalability
    Scalability challenges arise as knowledge corpora expand. Advanced indexing, distributed retrieval, and approximate nearest neighbor techniques facilitate efficient handling of large-scale knowledge bases [57]. Selective indexing and corpus curation, combined with infrastructure improvements like caching and parallel retrieval, allow RAG systems to scale to massive knowledge repositories. Research indicates that moderate-sized models augmented with large external corpora can outperform significantly larger standalone models, suggesting parameter efficiency advantages [10].

    7.5.6 Knowledge Freshness
    Rapidly evolving information necessitates regularly updated knowledge bases. RAG systems can efficiently maintain knowledge freshness through incremental updates and selective retrieval methods without requiring frequent retraining [30]. Integrating live search APIs and hybrid retrieval methods ensure real-time information retrieval, addressing dynamic knowledge demands [21]. Continuous updates and user-feedback integration support lifelong learning and timely information access.

    7.5.7 Bias
    Addressing bias in RAG involves curating balanced knowledge sources, employing diversification techniques in retrieval, and adjusting retriever embeddings to counteract inherent biases [46]. Prompts and model training that encourage balanced representation, along with transparency in source attribution, further mitigate bias propagation. This multi-faceted approach helps minimize biases in RAG outputs.

    7.5.8 Misinformation
    Combating misinformation involves preventive measures like curating trustworthy knowledge sources and reactive verification through stance classifiers and credibility assessments [66]. Models employing vigilant prompting, cross-verification with multiple retrieved documents, and external fact-checking modules enhance reliability and truthfulness. Robustness against adversarial misinformation insertion through continuous monitoring and data validation further strengthens RAG systems, ensuring accurate information dissemination.
[end on page 24]
```

### Document #4 — Multi-Stage Verification-Centric Framework for Mitigating Hallucination in Multi-Modal RAG
- `doc_id`: `doc::9a2709238ca846b68a832069a85e77d6`
- **Last Updated:** 2025-11-04T15:12:50.927357+00:00
- **Context:** Paper on mitigating hallucination in Vision Language Models for the KDD Cup 2025 challenge.
- **Content fetched inside document:**
```text
[start on page 5]
    4 Experiments
    In this section, we present our experimental setup and evaluation results. To assess the effectiveness of our multimodal RAG pipeline under settings aligned with the CRAG-MM 2025 challenge. All local experiments were conducted on the same hardware as the challenge environment: a single NVIDIA L40S GPU with 48GB of memory. For the generated answer, we employed the GPT-4o-mini [7] model as the evaluation LLM.

    4.1 Metrics
    Weevaluate our system using the official metric suite defined by the CRAG-MM challenge. Specifically, we report the total number of queries, correct answers, missed answers, and hallucinations. From these, we derive key indicators such as accuracy, hallucination rate, and missing rate. Additionally, we compute the truthfulness score to assess factual grounding.

    4.2 Single-source Augmentation
    To evaluate the effectiveness of our method, we conduct an ablation study on Task 1 under local evaluation, as shown in Table 1. The goal is to assess the incremental impact of the key component in our verification-centric pipeline.
    The LLaMA Vision Only baseline refers to the vision-language model (VLM) directly answering questions from image and text input without any external retrieval or augmentation. This setting yields a relatively high accuracy of 25.00%, but suffers from severe hallucination issues (59.62%) and a negative truthfulness score of -34.62%, indicating that the answers often conflict with the ground truth. This is expected, as the VLM attempts to answer all questions regardless of its knowledge limitations, resulting in a low

    | | | Accuracy (%) | Missing Rate (%) | Hallucination Rate (%) | Truthfulness Score (%) |
    |---:|:---------------------------|---------------:|-------------------:|-------------------------:|-------------------------:|
    | 0 | LLaMA Vision Only | 25 | 15.38 | 59.62 | -34.62 |
    | 1 | RAG Agent | 27.88 | 9.62 | 62.5 | -34.62 |
    | 2 | w/o CoV & Self-Consistency | 3.85 | 95.19 | 0.96 | 2.88 |
    | 3 | w/o CoV | 4.81 | 95.19 | 0 | 4.81 |
    | 4 | Ours | 14.42 | 82.69 | 2.88 | 11.54 |
    Table 1: Ablation study on Task 1 single-source augmentation under local evaluation.

    ---------------------- this was the passage that we matched to the query -------------
    missing rate but poor factual reliability. The RAG Agent baseline incorporates external retrieval via the official search API, enhancing access to relevant knowledge. This improves the accuracy to 27.88% and slightly reduces the missing rate to 9.62%. However, hallucination rate increases further to 62.50%, and the truthfulness score remains at -34.62%, showing that naive RAG without verification can introduce more misleading content. To further assess the role of verification, we remove key components from our pipeline to observe their individual effects. The w/o CoV & SelfConsistency variant disables both the self-consistency check and the Chain-of-Verification (CoV). As a result, the model becomes overly conservative, with a missing rate of 95.19% and very low accuracy (3.85%). While hallucination rate drops sharply to 0.96%, the overall coverage and utility of the model are severely degraded. A similar trend is observed in the w/o CoV variant, which enables self-consistency but disables CoV. Although slightly more accurate (4.81%), it still fails to provide meaningful coverage. Our final agent, denoted as Ours , integrates all proposed components. This configuration strikes the best balance between caution and informativeness. It achieves a truthfulness score of 11.54% , the highest among all systems, with an acceptable hallucination rate of 2.88% and an improved accuracy of 14.42%. The results demonstrate that each verification stage contributes to reducing hallucinations while selectively increasing answer confidence, enabling the agent to provide more trustworthy responses.
    --------------------------------------------------------------------------------------
    Our method is developed specifically for the Single-source Augmentation (Task 1) setting. The same pipeline and implementation are directly applied to Task 2 and Task 3 during the CRAG-MM challenge phase. Thus, we focus our discussion on Task 1.
[end on page 6]
```

### Document #5 — HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation
- `doc_id`: `doc::b0610cc6134b401db0ea68a77096e883`
- **Last Updated:** 2025-11-04T13:49:26.359552+00:00
- **Context:** This document examines advancements in retrieval-augmented generation methods for language models.
- **Content fetched inside document:**
```text
[start on page 1]
    1 Introduction
    Retrieval Augmentation Generation (hereafter referred to as RAG) helps large language models (LLMs) (OpenAI et al., 2024) reduce hallucinations (Zhang et al., 2023) and access real-time data
    ---------------------- this was the passage that we matched to the query -------------
    by incorporating an information retrieval component. While LLMs often use in-context learning (Gao et al., 2024) for generation, practical issues such as low-quality or poorly ranked retrieved documents can hinder RAG's effectiveness. These challenges emphasize the need for instruction-tuning tailored to RAG tasks. Fine-tuning generative models specifically for RAG improves their ability to integrate retrieved information (Zhang et al., 2024) (Yu et al., 2024), resulting in more accurate and contextually relevant responses compared to generalpurpose models.
    --------------------------------------------------------------------------------------
    RAFT (Zhang et al., 2024) enhances model performance in domain-specific RAG tasks by introducing distractor documents during training. EvidenceRAG (Schimanski et al., 2024) improves large language models in evidence-based question answering by incorporating an indexing task, enhancing their ability to accurately cite and reflect source information. RankRAG (Yu et al., 2024) employs a two-stage training process to simultaneously optimize the context ranking and answer generation capabilities of large language models (LLMs) in RAG tasks.
    Despite significant research efforts on RAGspecific generative models, several issues remain.
    Lack of Granular RAG Task Focus : Researchers have primarily concentrated on finetuning RAG models without enhancing their capabilities through more granular RAG tasks, limiting the potential to strengthen RAG abilities effectively.
    Lack of Task-Specific CoT Paradigm Design in RAG : Although there have been proposals to integrate chain-of-thought (CoT) reasoning into the training process to enhance model accuracy (Wei et al., 2023), these methods are not specifically designed for RAG scenarios. Even in the rare cases where RAG models do incorporate CoT (Zhang et al.,

    Figure 1: Traditional RAG methods have primarily focused on retrieving relevant information, with less emphasis on its effective utilization. We propose a method that enhances model performance in complex RAG scenarios by developing three progressive capabilities.
    
    2024), there remains a lack of differentiated CoT paradigms designed to address the unique challenges posed by different tasks in RAG. Consequently, the full potential of CoT in enhancing RAG performance has yet to be realized.

    Thus, We introduce a new RAG Instruction Tuning method: Hierarchical-Thought InstructionTuning Retrieval-Augmented Generation ( HIRAG ) adapting to complex RAG scenarios and propose that when fine-tuning RAG generation models, we focus on three progressively hierarchical abilities shown in Figure 1: Filtering: The ability that LLM filters out noise and selects the direct information. Combination: The ability of LLMs to merge, integrate, and summarize multiple pieces of useful information. RAG-Specific Reasoning: The capability refers to the ability to answer a question by making implicit or explicit inferences based on the information in the documents when the relevant information is not directly provided.

    To better achieve these three capabilities, a "think before answering" approach based on progressively hierarchical thought has been introduced.
    
    The contributions of this work are summarized as follows:
    We propose three progressive hierarchical capabilities that a RAG model requires: filtering, combination, and RAG-specific reasoning to enhance the granularity and specificity of RAG tasks when dealing with complex scenarios.

    We introduce HIRAG , a fine-tuning strategy that employs task-specific reasoning patterns to construct a progressive chain of thought. This approach constructs a progressive chain of thought, enabling the model to learn from easier to more complex tasks, thereby significantly enhancing its performance in RAG scenarios.

    Extensive experiments were conducted on six datasets, including the RAG-specific benchmark, single-hop open-domain data, multihop open-domain data, and domain-specific data. Our model significantly outperforms the current state-of-the-art models. We also conducted experiments on Chinese datasets, confirming the robustness of our approach. Furthermore, ablation studies demonstrate that the training tasks for the three capabilities contribute to the performance of HIRAG, and we explored the optimal data ratio.
[end on page 2]
```

## LLM Answers (16.15s)
**Tech:** OpenAI's GPT-5 (reasoning low)

**Result:**

Researchers are improving RAG accuracy and reducing made-up answers with a mix of better retrieval, smarter generation, and verification:

- Make retrieval cleaner and more relevant: use domain-adaptive encoders, query reformulation, rerankers, and iterative retrieval that breaks complex questions into simpler sub-queries so the right passages are fetched first [3]. Many pipelines add a lightweight reranking step to promote the most relevant passages [1].

- Choose or combine embeddings wisely: retrieval quality hinges on embeddings, so some use multiple embedding models. Confident RAG runs RAG several times with different embeddings, then picks the answer with the highest confidence (metrics like self-certainty or Distributional Perplexity), yielding ~5–10% accuracy gains over vanilla RAG in studies [2].

- Fine-tune models for RAG: domain-specific fine-tuning (e.g., Bio/Sci variants) and RAG-specific instruction-tuning help models better use retrieved evidence and avoid generic guesses [1]. HIRAG trains three skills—filtering noise, combining sources, and RAG-specific reasoning with a "think before answering" approach—showing state-of-the-art gains across datasets [5]. Other methods like RAFT (training with distractors), EvidenceRAG (indexing for citation), and RankRAG (jointly optimizing ranking and generation) further improve grounding [5].

- Integrate retrieval and generation more tightly: joint retriever–generator training and architectures that inject retrieved facts during generation can reduce drift from evidence [3].

- Add verification and self-checks: self-critique (Self-RAG), penalizing ungrounded claims, external fact-checking, cross-document agreement checks, and requiring source citations reduce hallucinations and raise trust [3]. In multimodal RAG, multi-stage verification (self-consistency + Chain-of-Verification) cut hallucinations sharply while improving truthfulness compared to naïve RAG [4].

- Be aware of limits: RAG still depends on retrieval quality and corpus alignment; noisy or off-target passages can hurt answers without careful tuning [1].

**reasoning:** Synthesized techniques from reviews and empirical studies: retrieval/reranking and iterative methods [3,1]; multi-embedding confidence selection [2]; RAG-specific fine-tuning and instruction-tuning (HIRAG, RAFT, EvidenceRAG, RankRAG) [5,1]; integration and self-verification/citations [3]; verification pipeline evidence in multimodal setting [4]; caveats on retrieval quality [1].

**Citations:**

[1] doc::6371023da29b4bbe8242ffc5caf4a8cd — Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs

[2] doc::3b9c43d010984d4cb11233b5de905555 — Each to Their Own: Exploring the Optimal Embedding in RAG

[3] doc::688cfbc0abdc4520a73e219ac26aff41 — A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions

[4] doc::9a2709238ca846b68a832069a85e77d6 — Multi-Stage Verification-Centric Framework for Mitigating Hallucination in Multi-Modal RAG

[5] doc::b0610cc6134b401db0ea68a77096e883 — HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation


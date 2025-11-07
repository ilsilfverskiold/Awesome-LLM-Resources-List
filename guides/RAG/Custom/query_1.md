# Query 1

## Result 

**Original Query:** knowledge graphs vs plain rag, which is better?

**LLM answer:**

Neither is universally “better.” It depends on your needs.

When Knowledge Graphs help:
- Complex, relational, or multi-hop questions: Graph-based RAG represents entities and relations, giving more structured, semantically rich context than plain chunk retrieval [4]. This helps synthesize cross‑document insights and support verifiable claims with source links (e.g., GraphRAG) [1].
- Noisy corpora or multimodal tasks: Plain RAG can pull irrelevant text; KG-based methods reduce noise by retrieving structured facts. In VQA, multimodal KGs improved precision and outperformed strong baselines on E‑VQA and InfoSeek [3].
- Domains needing structure and traceability (e.g., customer support KGs) [4].

Trade-offs of KG-based RAG:
- Cost and complexity: Building and maintaining graphs (including multimodal) is hard; large KGs expand the search space and make retrieval slower [3]. Traditional GraphRAG can suffer from high construction cost and fixed, one-shot retrieval; agentic approaches (Graph‑R1) aim to fix this via lightweight hypergraphs and RL-driven multi‑turn retrieval [2]. Some methods also rely on long-context prompts and careful prompt design [2]. Additionally, some hierarchical graph systems still don’t tightly couple retrieval with the graph structure at query time, limiting gains [4].

When plain RAG is enough:
- Straightforward fact lookup or when you lack graph infrastructure. It’s simpler and cheaper to deploy, but can miss contextually precise, structured relationships and may retrieve irrelevant passages [1][3].

Rule of thumb:
- Start with plain RAG for simple FAQs and low-stakes tasks.
- Prefer KG‑RAG (or GraphRAG variants) for complex reasoning, cross‑document synthesis, and when you need evidence-linked, reliable answers; consider newer agentic/graph-guided retrieval to mitigate costs and rigidity [1][2][4].

**Citations:**

[1] doc::6371023da29b4bbe8242ffc5caf4a8cd — Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs

[2] doc::2d32ad4c9bce416ca4c233fab33bb7a9 — Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning

[3] doc::45dfa5e3daff4215b7e4c6c64b4497f5 — mKG-RAG: Multimodal Knowledge Graph-Enhanced RAG for Visual Question Answering

[4] doc::6c692b0a4ac04b1b9313ee01e24e7dd7 — LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval

--------------------------------------------------------------------------------------------------------------------------------------

# Pipeline

## Query Optimizer (3.07s)

**Tech:** GPT-4o-mini, structured responses

1. **Query:** comparison of knowledge graphs and retrieval-augmented generation
2. **Search Method:** hybrid

## Retrieve (1.14s)

**Tech:** Qdrant, LlamaIndex

Query 1 (hybrid) top 20 for query: comparison of knowledge graphs and retrieval-augmented generation (top_k = 20)

**Results**

[1] score=0.5018 doc=docs_ingestor/docs/arxiv/2508.05498.pdf chunk=SDOC::SUM::251104155932
  text: The document presents GRAIL, a novel framework integrating Large Language Models (LLMs) with Retrieval-Augmented Generation techniques to enhance reasoning over structured knowledge, specifically knowledge graphs. Unlike existing methods that primarily focus on unstructured data, GRAIL allows an agent to interactively explore graph structures, improving retrieval precision and reasoning performance significantly. Experiments demonstrate GRAIL's effectiveness, achieving an average accuracy improvement of 21.01% and F1 score enhancement of 22.43% across three knowledge graph question-answering datasets. The two-stage training paradigm combines supervised learning and reinforcement learning, refining the model's capabilities in graph comprehension and reasoning, ultimately addressing the scarcity of high-quality graph data in existing research.

[2] score=0.5000 doc=docs_ingestor/docs/arxiv/2508.10391.pdf chunk=S5::C01::251104161739
  text: Knowledge Graph Based Retrieval-Augmented Generation To better capture the relational nature of information, KGbased RAG has emerged as a prominent research direction. By representing knowledge as a graph of entities and relations, these methods aim to provide a more structured and semantically rich context for the LLM (Peng et al. 2024). Early approaches in this domain focused on leveraging graph structures for improved retrieval. For instance, GraphRAG (Edge et al. 2024) organizes documents into community-based KGs to preserve local context, while other methods like FastGraphRAG utilize graph-centrality metrics such as PageRank (Page et al. 1999) to prioritize more important nodes during retrieval. This subgraph retrieval approach has also proven effective in industrial applications like customer service, where KGs are constructed from historical support tickets to provide structured context (Xu et al. 2024). These methods marked a significant step forward by imposing a macro-structure onto the knowledge base, moving beyon

[3] score=0.5000 doc=docs_ingestor/docs/arxiv/2508.10391.pdf chunk=S3::C04::251104161739
  text: Introduction Figure 1: Comparison of typical LLM retrieval-augmented generation frameworks. Naïve LLM Naïve RAG Corpus Query: What is Apache Spark and what are its Response Chunk Response features? Query Retrieval RAG with Graph Graph Retrieval Graph Search Summary Knowledge Query Indexing Match in Complexity O(n all entities 0 O Corpus Origin Knowledge Response Reduce Redundancy LeanRAG Improve Efficiency Graph building Summary Knowledge Retrieval LCAI Search Query Indexing Match entities Complexity log(n) bottom-top Origin Knowledge Corpus Response Enriched Semantic Information Retrieval Entity key building tablish higher-order abstractions. This process transforms fragmented, isolated hierarchies into a unified, fully navigable semantic network, where both fine-grained details and abstracted knowledge are seamlessly interconnected.

[4] score=0.4320 doc=docs_ingestor/docs/arxiv/2508.10391.pdf chunk=S5::C02::251104161739
  text: Knowledge Graph Based Retrieval-Augmented Generation Recognizing the need for more fine-grained control and abstraction, subsequent works have explored more sophisticated hierarchical structures. HiRAG, the current stateof-the-art, clusters entities to form multi-level summaries (Huang et al. 2025a), while LightRAG (Guo et al. 2024) proposes a dual-level framework to balance global and local information retrieval. While these hierarchical methods have progressively improved retrieval quality, a critical gap persists in how the constructed graph structures are leveraged at query time. The retrieval process is often decoupled from the indexing structure; for instance, an initial search may be performed over a 'flattened' list of all nodes, rather than being directly guided by the indexed community or hierarchical relations. This decoupling means the rich structural information is primarily used for post-retrieval context expansion, rather than for guiding the initial, crucial step of identifying relevant information. This can 

[5] score=0.3150 doc=docs_ingestor/docs/arxiv/2507.19715.pdf chunk=S9::C01::251104150336
  text: Comparison to Our Work While prior research addresses individual components-fast retrieval, diversity, or graph-based augmentation-our work provides a unified framework for semantic compression and graph-augmented retrieval. We contribute theoretical foundations, objective formulations, and architectural implications for next-generation vector search systems.

[6] score=0.3116 doc=docs_ingestor/docs/arxiv/2509.09272.pdf chunk=S8::C01::251104174407
  text: RAG + Graphs in QA To address these limitations of traditional RAG, particularly in retrieving contextually precise and semantically rich information, research has tried to integrate structured knowledge sources and graphbased reasoning into the retrieval pipeline. One notable direction of work is LLM-based retrieval, which incorporates knowledge graphs information into the generation process of LLMs. The LLMs are augmented using the retrieved facts from the KG [26], leading to a clear dependence on the quality of extracted graph information, using it to generate responses. Research has also been done towards augmenting knowledge graph information, retrieved through some semantic or some other similarity index, in the prompts given to the LLM [27], to help the model do zero-shot question-answering. Some researchers have tried a different approach to fact retrieval, where the model tries different queries, using structured query languages, until the desired information comes through [28]. All of these approaches have used KGs

[7] score=0.2465 doc=docs_ingestor/docs/arxiv/2508.10391.pdf chunk=S4::C02::251104161739
  text: Related Work Retrieval-Augmented Generation Substantial research has been dedicated to overcoming this limitation. One line of work focuses on improving the retriever itself, evolving from sparse methods like BM25 (Robertson, Zaragoza et al. 2009) to dense retrieval models such as DPR (Karpukhin et al. 2020) and Contriever (Izacard et al. 2021), which learn to better capture semantic relevance. Another direction targets the indexing and organization of the source documents (Jiang et al. 2023). Recent advancements have explored creating hierarchical summaries of text chunks, allowing retrieval to occur at multiple levels of granularity. For instance, RAPTOR builds a tree of recursively summarized text clusters, enabling retrieval of both fine-grained details and high-level summaries (Sarthi et al. 2024). Despite these improvements, these methods still largely treat knowledge as a linear sequence or a simple tree of text. They do not explicitly model the complex, non-hierarchical relations that often exist between different en

[8] score=0.2375 doc=docs_ingestor/docs/arxiv/2508.05318.pdf chunk=S10::C01::251104155727
  text: 5 Conclusion Wepropose mKG-RAG, a novel retrieval-augmented generation framework that integrates multimodal knowledge graphs (KGs) to overcome the knowledge limitations of multimodal large language models (MLLMs). Our framework constructs structured, modality-aligned KGs using MLLM-driven keyword extraction and cross-modal alignment, and employs a dual-stage retrieval system, which combines vector-based and graph-based retrieval for precise knowledge augmentation. Extensive experiments show that mKG-RAG outperforms state-of-the-art methods, with ablation studies validating each component's contributions.

[9] score=0.2331 doc=docs_ingestor/docs/arxiv/2508.05498.pdf chunk=S16::C04::251104155932
  text: Main Results all retrieval-based methods demonstrate superior performance. This validates the fundamental challenge in graphaugmented generation - the inherent complexity of graph data exceeds the processing capacity of LLMs when handled without selective retrieval. (3) Compared to directly feeding k-hop information into the model, our approach achieves significant performance improvements, particularly on multi-hop reasoning benchmarks. This demonstrates that our proposed interactive retrieval mechanism effectively filters meaningful subgraph information, ensuring the input to the LLM remains both sufficient and concise for generating higher-quality answers. (4) Compared to training-free baselines (e.g., ToG, LightRAG), GRAIL-Retriever achieves substantial improvements, particularly in multi-hop reasoning tasks. This demonstrates the effectiveness of our proposed two-stage training strategy for retrieval, proving that enhancing the reasoning capability of the retrieval model can further boost retrieval performance.

[10] score=0.2274 doc=docs_ingestor/docs/arxiv/2507.20059.pdf chunk=SDOC::SUM::251104151210
  text: This study evaluates the effectiveness of Retrieval-Augmented Generation (RAG) systems across diverse knowledge sources and tasks. Utilizing MASSIVEDS, a large-scale datastore, the authors found that retrieval mainly benefits smaller models while larger models exhibit diminishing returns. The analysis shows no single retrieval source consistently outperforms others, highlighting the necessity for adaptive retrieval strategies. Furthermore, efforts such as reranking and query routing have limited improvement on overall performance. Key challenges include inaccurate relevance estimation and training-inference mismatch, suggesting a need for improved integration of retrieval mechanisms with language models. The findings indicate a critical gap in deploying RAG systems effectively in real-world scenarios, particularly for larger models that increasingly internalize domain knowledge without retrieval support.

[11] score=0.1616 doc=docs_ingestor/docs/arxiv/2508.05318.pdf chunk=S6::C03::251104155727
  text: 1 Introduction Recently, Retrieval-Augmented Generation ( RAG ) [16] has shown great potential in addressing these challenges by leveraging external knowledge databases to supplement the internal knowledge of MLLMs, thereby enabling more accurate answer generation [32, 4, 11]. Specifically, multiple query-relevant documents are retrieved from external knowledge databases and serve as in-context information to augment the generation process of MLLMs. Despite their success, vanilla RAGbased VQA methods that rely on unstructured documents or paragraphs often introduce irrelevant or even misleading information [38, 51], thereby compromising the accuracy and reliability of generated answers. Moreover, these approaches typically overlook the structural relationships among knowledge elements, limiting the reasoning capabilities of MLLMs. As illustrated in Figure 1 (b), the presence of noisy and unstructured context makes it difficult for MLLMs to identify and leverage relevant supporting evidence. To overcome these limitations, a p

[12] score=0.1480 doc=docs_ingestor/docs/arxiv/2507.21892.pdf chunk=SDOC::SUM::251104153106
  text: The document proposes Graph-R1, an innovative GraphRAG framework that utilizes end-to-end reinforcement learning (RL) to enhance retrieval-augmented generation for large language models (LLMs). It aims to overcome existing challenges in GraphRAG methods, such as high construction costs and reliance on long-context reasoning. The Graph-R1 framework introduces a lightweight knowledge hypergraph construction, enabling multi-turn agent-environment interaction. Experimental results demonstrate superior performance in reasoning accuracy, retrieval efficiency, and generation quality compared to traditional methods. The findings contribute to advancing knowledge-driven generation systems by addressing critical limitations of current models, thereby promoting better alignment between structured graph knowledge and language generation.

[13] score=0.1214 doc=docs_ingestor/docs/arxiv/2509.09272.pdf chunk=S3::C01::251104174407
  text: 1. Introduction Retrieval-Augmented Generation (RAG) has emerged as a powerful approach in natural language processing, particularly in question-answering systems. By combining the retrieval of relevant documents with the generative capabilities of large language models (LLMs), RAG models are able to generate contextually accurate and informative responses. Traditional RAG approaches typically rely on retrieving passages from vast corpora of unstructured text, which are then fed into a generative model to formulate an answer. While effective, this approach often faces challenges such as irrelevant retrieval, redundancy in answers, and limited capacity to understand structured relationships between concepts.

[14] score=0.1167 doc=docs_ingestor/docs/arxiv/2507.20804.pdf chunk=SDOC::SUM::251104151625
  text: The document introduces MMGraphRAG, a novel framework that enhances multimodal retrieval-augmented generation (RAG) by utilizing multimodal knowledge graphs (MMKGs). It addresses limitations of traditional methods in capturing multimodal relationships and requires less extensive training for better generalization. MMGraphRAG uses scene graphs to refine visual information and integrates this with textual knowledge graphs to form a robust MMKG, facilitating better entity linking and context retrieval. Experimental results demonstrate its state-of-the-art performance on the DocBench and MMLongBench datasets, showcasing improved accuracy in complex multimodal question answering tasks, especially in diverse domains such as academia, finance, government, law, and news, outpacing previous methods significantly.

[15] score=0.1020 doc=docs_ingestor/docs/arxiv/2508.05509.pdf chunk=S5::C01::251104160034
  text: Preliminaries. Retrieval-Augmented Generation (RAG) enhances language models by incorporating external knowledge retrieved from a large corpus. We denote the input as a natural language question q , which may involve latent constraints, or multihop reasoning. The system has access to a retrieval corpus C = { c 1 , c 2 , . . . , c N } , where each c i represents a passage, document chunk, or knowledge entry consisting of unstructured text. These entries may vary in granularity and source (e.g., Wikipedia, scientific papers, web documents), but are assumed to be independently indexable and retrievable. Given a query q or any intermediate sub-question q ′ , a retriever R returns a ranked list of relevant passages R ( q ′ ) ⊂ C to support downstream reasoning. Each retrieved item c ∈ C is treated as a semantically self-contained unit of information, which the system uses as external evidence during the generation or verification process.

[16] score=0.0856 doc=docs_ingestor/docs/arxiv/2509.09272.pdf chunk=S11::C01::251104174407
  text: GraphRAG GraphRAG, a Microsoft Research innovation, represents a significant advancement in retrievalaugmented generation (RAG) by integrating structured knowledge graphs to enhance large language models' (LLMs) contextual understanding of private datasets. Unlike traditional RAG systems that rely on vector similarity searches, GraphRAG constructs an LLM-generated entity-relationship graph from source documents, enabling holistic analysis and semantic connections across disparate data points. GraphRAG addresses critical limitations in baseline RAG systems by transforming unstructured text into structured knowledge representations, particularly in synthesizing crossdocument insights and supporting verifiable assertions through source-linked evidence. It extracts relevant data from the knowledge graph, identifies the appropriate community related to the question, and utilizes the generated graph, entities, relations, claims, summaries, etc., to generate accurate and concise responses.

[17] score=0.0844 doc=docs_ingestor/docs/arxiv/2508.05318.pdf chunk=S9::C08::251104155727
  text: 4 Experiments 4.2 Performance Comparison Results on E-VQA and InfoSeek. In this section, we compare mKG-RAG with Zero-shot MLLMs and RAG-based approaches on the benchmarks mentioned above. The results in Table 3 demonstrate that zero-shot MLLMs struggle with knowledge-based VQA tasks, particularly on the InfoSeek dataset. These limitations underscore the critical need for external knowledge integration. By augmenting LLaVA-More with mKG-RAG, we achieve substantial improvements, over 20.3% on E-VQA and 31.9% on InfoSeek, highlighting the value of retrieval augmentation.

[18] score=0.0635 doc=docs_ingestor/docs/arxiv/2507.21544.pdf chunk=SDOC::SUM::251104152346
  text: This document presents a framework for detecting knowledge conflicts in retrieval-augmented generation systems. It critiques existing benchmarks for limitations such as narrow focus and inadequate conflict types. Proposed is the MAGIC benchmark, which utilizes knowledge graphs to create varied and interpretive knowledge conflicts, enhancing LLMs' ability to identify contradictions in documents. Experimental results reveal that many LLMs struggle with conflict detection, particularly in complex multi-hop scenarios. Insights from MAGIC provide a basis for improving LLMs' integration of conflicting information, highlighting the need for more robust systems capable of identifying and addressing knowledge discrepancies.

[19] score=0.0608 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S3::C01::251104135247
  text: ABSTRACT Fine-tuning is an immensely resource expensive process when trying to retrain Large Language Models (LLMs) to have access to a larger bank of knowledge. To alleviate this issue there have been many different fine-tuning techniques proposed which have shown good progress in trying to reduce time and computational resources to achieve fine-tuning but with LLMs becoming more intelligent and larger, this issue continues to arise. Hence a new method of enabling knowledge expansion on LLMs had to be devised. Retrieval-Augment-Generate (RAG) is a class of techniques where information is stored in a database and appropriate chunks of information are retrieved to help answer the question. However there are many limitations to naive RAG implementations. This paper proposes the KeyKnowledgeRAG ( K 2 RAG ) framework to address the scalability and answer accuracy limitations associated with naive RAG implementations. This framework takes inspiration from divide-and-conquer ideology, and combines dense and sparse vector search, k

[20] score=0.0535 doc=docs_ingestor/docs/arxiv/2507.21892.pdf chunk=S2::C01::251104153106
  text: ABSTRACT Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose Graph-R1 , an agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality. Our code is publicly available 1 

## Rerank (0.62s)

**Tech:** Cohere API

**Results**

Rerank summary:
- strategy=cohere
- model=rerank-english-v3.0
- candidates=20
- eligible_above_threshold=20
- kept=6 (threshold=0)

Reranked Relevant (6/20 kept ≥ 0) top 6 for query: knowledge graphs vs plain rag, which is better?

[1] score=0.9565 doc=docs_ingestor/docs/arxiv/2509.09272.pdf chunk=S8::C01::251104174407
  text: RAG + Graphs in QA To address these limitations of traditional RAG, particularly in retrieving contextually precise and semantically rich information, research has tried to integrate structured knowledge sources and graphbased reasoning into the retrieval pipeline. One notable direction of work is LLM-based retrieval, which incorporates knowledge graphs information into the generation process of LLMs. The LLMs are augmented using the retrieved facts from the KG [26], leading to a clear dependence on the quality of extracted graph information, using it to generate responses. Research has also been done towards augmenting knowledge graph information, retrieved through some semantic or some other similarity index, in the prompts given to the LLM [27], to help the model do zero-shot question-answering. Some researchers have tried a different approach to fact retrieval, where the model tries different queries, using structured query languages, until the desired information comes through [28]. All of these approaches have used KGs

[2] score=0.9481 doc=docs_ingestor/docs/arxiv/2509.09272.pdf chunk=S11::C01::251104174407
  text: GraphRAG GraphRAG, a Microsoft Research innovation, represents a significant advancement in retrievalaugmented generation (RAG) by integrating structured knowledge graphs to enhance large language models' (LLMs) contextual understanding of private datasets. Unlike traditional RAG systems that rely on vector similarity searches, GraphRAG constructs an LLM-generated entity-relationship graph from source documents, enabling holistic analysis and semantic connections across disparate data points. GraphRAG addresses critical limitations in baseline RAG systems by transforming unstructured text into structured knowledge representations, particularly in synthesizing crossdocument insights and supporting verifiable assertions through source-linked evidence. It extracts relevant data from the knowledge graph, identifies the appropriate community related to the question, and utilizes the generated graph, entities, relations, claims, summaries, etc., to generate accurate and concise responses.

[3] score=0.9424 doc=docs_ingestor/docs/arxiv/2507.21892.pdf chunk=S2::C01::251104153106
  text: ABSTRACT Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose Graph-R1 , an agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality. Our code is publicly available 1 .

[4] score=0.8617 doc=docs_ingestor/docs/arxiv/2508.05318.pdf chunk=S10::C01::251104155727
  text: 5 Conclusion Wepropose mKG-RAG, a novel retrieval-augmented generation framework that integrates multimodal knowledge graphs (KGs) to overcome the knowledge limitations of multimodal large language models (MLLMs). Our framework constructs structured, modality-aligned KGs using MLLM-driven keyword extraction and cross-modal alignment, and employs a dual-stage retrieval system, which combines vector-based and graph-based retrieval for precise knowledge augmentation. Extensive experiments show that mKG-RAG outperforms state-of-the-art methods, with ablation studies validating each component's contributions.

[5] score=0.8239 doc=docs_ingestor/docs/arxiv/2508.05318.pdf chunk=S6::C03::251104155727
  text: 1 Introduction Recently, Retrieval-Augmented Generation ( RAG ) [16] has shown great potential in addressing these challenges by leveraging external knowledge databases to supplement the internal knowledge of MLLMs, thereby enabling more accurate answer generation [32, 4, 11]. Specifically, multiple query-relevant documents are retrieved from external knowledge databases and serve as in-context information to augment the generation process of MLLMs. Despite their success, vanilla RAGbased VQA methods that rely on unstructured documents or paragraphs often introduce irrelevant or even misleading information [38, 51], thereby compromising the accuracy and reliability of generated answers. Moreover, these approaches typically overlook the structural relationships among knowledge elements, limiting the reasoning capabilities of MLLMs. As illustrated in Figure 1 (b), the presence of noisy and unstructured context makes it difficult for MLLMs to identify and leverage relevant supporting evidence. To overcome these limitations, a p

[6] score=0.7551 doc=docs_ingestor/docs/arxiv/2508.10391.pdf chunk=S5::C01::251104161739
  text: Knowledge Graph Based Retrieval-Augmented Generation To better capture the relational nature of information, KGbased RAG has emerged as a prominent research direction. By representing knowledge as a graph of entities and relations, these methods aim to provide a more structured and semantically rich context for the LLM (Peng et al. 2024). Early approaches in this domain focused on leveraging graph structures for improved retrieval. For instance, GraphRAG (Edge et al. 2024) organizes documents into community-based KGs to preserve local context, while other methods like FastGraphRAG utilize graph-centrality metrics such as PageRank (Page et al. 1999) to prioritize more important nodes during retrieval. This subgraph retrieval approach has also proven effective in industrial applications like customer service, where KGs are constructed from historical support tickets to provide structured context (Xu et al. 2024). These methods marked a significant step forward by imposing a macro-structure onto the knowledge base, moving beyon
  
## Context Expansion (1.03s)

**Tech:** Redis (to fetch neighbors and additional informtion)

**Results:**

### Document #1 — Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs
- `doc_id`: `doc::6371023da29b4bbe8242ffc5caf4a8cd`
- **Last Updated:** 2025-11-04T17:44:07.300967+00:00
- **Context:** Comparative study on methodologies for integrating knowledge graphs in QA systems using LLMs.
- **Content fetched inside document:**
```text
[start on page 5]
    ---------------------- this was the passage that we matched to the query -------------
    RAG + Graphs in QA
    To address these limitations of traditional RAG, particularly in retrieving contextually precise and semantically rich information, research has tried to integrate structured knowledge sources and graphbased reasoning into the retrieval pipeline. One notable direction of work is LLM-based retrieval, which incorporates knowledge graphs information into the generation process of LLMs. The LLMs are augmented using the retrieved facts from the KG [26], leading to a clear dependence on the quality of extracted graph information, using it to generate responses. Research has also been done towards augmenting knowledge graph information, retrieved through some semantic or some other similarity index, in the prompts given to the LLM [27], to help the model do zero-shot question-answering. Some researchers have tried a different approach to fact retrieval, where the model tries different queries, using structured query languages, until the desired information comes through [28]. All of these approaches have used KGs as an external source to retrieve information from and answer the questions [29]. Then, aligning the retrieval process with LLM even more closely, some researchers have proposed methods which use LLMs in intermediate steps as well to plan the retrieval and judge whether the retrieved information is relevant or not [29,30], continuing the process until the desired output emerges.
    --------------------------------------------------------------------------------------
    Another interesting direction of work is integrating GNNs with LLMs, which leverages graph neural networks to enhance retrieval and re-ranking using learned graph representations, along with generation capabilities of LLMs. There have been approaches such as GNN-RAG [31], which have tried combining language understanding abilities of LLMs with the reasoning abilities of GNNs in a retrieval-augmented generation (RAG) style. Other methods of GNN-LLM alignment, have been classified into symmetric and asymmetric alignment [32]. Symmetric alignment refers to the equal treatment of the graph and text modalities during the alignment process[33, 34, 35]. Asymmetric alignment focuses on allowing one modality to assist or enhance the other, here leveraging the capabilities of GNNs to improve the LLMs [36, 37, 38].
    ---------------------- this was the passage that we matched to the query -------------
    GraphRAG
    GraphRAG, a Microsoft Research innovation, represents a significant advancement in retrievalaugmented generation (RAG) by integrating structured knowledge graphs to enhance large language models' (LLMs) contextual understanding of private datasets. Unlike traditional RAG systems that rely on vector similarity searches, GraphRAG constructs an LLM-generated entity-relationship graph from source documents, enabling holistic analysis and semantic connections across disparate data points. GraphRAG addresses critical limitations in baseline RAG systems by transforming unstructured text into structured knowledge representations, particularly in synthesizing crossdocument insights and supporting verifiable assertions through source-linked evidence. It extracts relevant data from the knowledge graph, identifies the appropriate community related to the question, and utilizes the generated graph, entities, relations, claims, summaries, etc., to generate accurate and concise responses.
    --------------------------------------------------------------------------------------
    The continual evolution of such hybrid architectures reflects the dynamic nature of the QA landscape and its responsiveness to complex information needs. Given the diversity of methods explored to enhance QA systems-from rule-based techniques to advanced neural and hybrid models-it becomes essential to establish robust mechanisms for evaluating their effectiveness. It is thus important to discuss the evaluation paradigms that compare and benchmark these systems. Extractive QA benchmarks almost universally use Exact Match (EM)-the strict proportion of predictions that character-for-character match a ground-truth answer-and F1 score, the harmonic mean of token-level precision and recall, to evaluate answer quality. On SQuAD v2.0 [39], human annotators achieve around 86.831 EM and 89.452 F1 on the test set, whereas state-of-the-art models now exceed 90 EM and 93 F1. Natural Questions [40] extends this paradigm to long-answer (paragraph) and short-answer (span or yes/no) annotations drawn from real Google search queries. Meanwhile, multiple-choice datasets like OpenBookQA [41]-designed to mimic open-book science exams-use simple accuracy. Complex reasoning benchmarks push beyond single-span extraction. HotpotQA [42], a multi-hop dataset built on pairs of Wikipedia articles, evaluates both answer-span EM/F1 and supporting-fact EM/F1, plus a joint metric requiring both to be correct; even top models achieve only ~72 joint F1[43] under the full-wiki setting, far below human performance of ~82 average F1 and ~96 upper bound F1. These core metrics and datasets underpin broader QA evaluation: multi-task suites like GLUE/MMLU [44, 45] include QA subtasks to probe general language understanding, while specialized frameworks such as MT-bench [46] ('LLM as judge') and automated platforms like Scale Evaluation layer on top to assess conversational and retrieval-augmented QA in real-world scenarios.
    This review has highlighted the broad spectrum of methodologies and evaluation strategies used in enhancing and assessing question answering systems. With this foundation in place, we now transition to the specific tools and frameworks employed in our research for graph-based question answering. In the following section, we briefly introduce and contextualize spaCy, Stanford CoreNLP, and GraphRAG-three diverse and widely-used tools we have utilized for knowledge graph construction and integration with LLMs.
[end on page 6]
```

### Document #2 — Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning
- `doc_id`: `doc::2d32ad4c9bce416ca4c233fab33bb7a9`
- **Last Updated:** 2025-11-04T15:31:06.753251+00:00
- **Context:** Advancing reinforcement learning applications in graph-based knowledge representation for LLMs.
- **Content fetched inside document:**
```text
[start on page 1]
    ---------------------- this was the passage that we matched to the query -------------
    ABSTRACT
    Retrieval-Augmented Generation (RAG) mitigates hallucination in LLMs by incorporating external knowledge, but relies on chunk-based retrieval that lacks structural semantics. GraphRAG methods improve RAG by modeling knowledge as entity-relation graphs, but still face challenges in high construction cost, fixed one-time retrieval, and reliance on long-context reasoning and prompt design. To address these challenges, we propose Graph-R1 , an agentic GraphRAG framework via end-to-end reinforcement learning (RL). It introduces lightweight knowledge hypergraph construction, models retrieval as a multi-turn agent-environment interaction, and optimizes the agent process via an end-to-end reward mechanism. Experiments on standard RAG datasets show that Graph-R1 outperforms traditional GraphRAG and RL-enhanced RAG methods in reasoning accuracy, retrieval efficiency, and generation quality. Our code is publicly available 1 .
    --------------------------------------------------------------------------------------
[end on page 1]
```

### Document #3 — mKG-RAG: Multimodal Knowledge Graph-Enhanced RAG for Visual Question Answering
- `doc_id`: `doc::45dfa5e3daff4215b7e4c6c64b4497f5`
- **Last Updated:** 2025-11-04T15:57:27.069651+00:00
- **Context:** This document presents a framework integrating multimodal knowledge graphs into VQA tasks.
- **Content fetched inside document:**
```text
[start on page 1]
    1 Introduction
    Visual Question Answering (VQA) [2, 19] is a challenging task at the intersection of vision and language understanding, requiring models to interpret images and answer related questions. This capability has enabled remarkable advances in various domains, including medical image diagnosis [33] and customer service support [12]. Recently, due to the powerful visual-linguistic understanding and reasoning capabilities, Multimodal Large Language Models ( MLLMs ) [35, 30, 50, 7] have provided a promising solution to conventional VQA tasks. For instance, LLaVA [35] demonstrates strong
    zero-shot performance on commonsense VQA by leveraging pre-trained visual encoders for image representation alongside the reasoning capabilities of large language models (LLMs). Despite notable advancements, MLLMs face critical limitations in knowledge-intensive VQA scenarios [40, 6] (termed knowledge-based VQA ), particularly those requiring encyclopedic knowledge, long-tail factual recall, or contextual reasoning beyond immediate visual inputs. As illustrated in Figure 1 (a), when queried about the latest renovation date of a stadium, typical MLLMs exhibit two characteristic failure modes: generating plausible but factually incorrect responses or refusing to answer altogether. These issues stem from the scarcity of relevant knowledge in MLLMs' training corpus and the inherent difficulty of memorizing low-frequency facts [6].
    ---------------------- this was the passage that we matched to the query -------------
    Recently, Retrieval-Augmented Generation ( RAG ) [16] has shown great potential in addressing these challenges by leveraging external knowledge databases to supplement the internal knowledge of MLLMs, thereby enabling more accurate answer generation [32, 4, 11]. Specifically, multiple query-relevant documents are retrieved from external knowledge databases and serve as in-context information to augment the generation process of MLLMs. Despite their success, vanilla RAGbased VQA methods that rely on unstructured documents or paragraphs often introduce irrelevant or even misleading information [38, 51], thereby compromising the accuracy and reliability of generated answers. Moreover, these approaches typically overlook the structural relationships among knowledge elements, limiting the reasoning capabilities of MLLMs. As illustrated in Figure 1 (b), the presence of noisy and unstructured context makes it difficult for MLLMs to identify and leverage relevant supporting evidence. To overcome these limitations, a promising direction is to retrieve structured knowledge, such as Knowledge Graphs (KGs) [23] for augmentation generation [22, 15, 59]. However, in the VQA setting, which inherently involves multimodal reasoning, relying solely on textual KGs is suboptimal, as both modalities are crucial for identifying relevant knowledge. Therefore, integrating Multimodal Knowledge Graphs into the retrieval-augmented VQA framework presents a more robust solution to generate reliable and precise responses in knowledge-intensive scenarios, as illustrated in Figure 1 (c).
    --------------------------------------------------------------------------------------
    However, retrieving relevant knowledge from multimodal knowledge graphs to enhance the generation of knowledge-based VQA tasks is exceptionally challenging. First, offthe-shelf multimodal KGs [36] are generally built around common entities, and often lack the encyclopedic or long-tail knowledge required by knowledge-intensive questions, rendering them ineffective for direct use in knowledge-based VQA. Moreover, current knowledge sources used in knowledge-based VQA [40, 6] are typically organized in unstructured documents containing substantial contextual noise, making it challenging to extract well-structured entities and relationships essential for constructing high-quality multimodal KGs. Furthermore, a large-scale knowledge graph constructed from millions of documents, each potentially containing hundreds of entities and relationships, significantly expands the search space. Consequently, performing direct retrieval over such a graph is computationally inefficient and adversely affects retrieval precision.
    To address the challenges above, this paper proposes mKG-RAG , a novel
    Figure 1: Illustration of issues in knowledge-based VQA. (b) Vanilla RA methods suffer from retrieving unstructured knowledge from external documents via unimodal retrievers. (c) Our mKG-RAG augments MLLMs with structural information from multimodal knowledge graphs.
    retrieval-augmented generation framework integrated with multimodal knowledge graphs designed to enhance the reasoning capabilities of MLLMs in knowledge-based VQA tasks. More specifically, a multimodal knowledge graph construction module is introduced to transform unstructured multimodal documents, such as Wikipedia articles, into structured knowledge representations. This module leverages MLLM-powered keyword extraction and vision-text alignment to extract semantically consistent and modality-aligned entities and relationships from external multimodal documents. To enable efficient retrieval, mKG-RAG develops a dual-stage search paradigm that combines a coarse-grained document recall and a fine-grained entity/relationship retrieval. The coarse stage efficiently narrows the search space by identifying candidate documents likely to contain relevant evidence, while the fine stage refines the results by retrieving query-relevant entities and relationships from multimodal KGs that are dynamically constructed from these potentially noisy documents. During retrieval, unlike previous methods that rely on isolated unimodal retrievers, we introduce a question-aware multimodal retriever trained on a high-quality question-evidence dataset to further enhance retrieval precision within the proposed search paradigm. Comprehensive evaluations on two frequently used benchmarks demonstrate the superior performance of mKG-RAG, achieving an accuracy of 36.3% on E-VQA and 40.5% on InfoSeek.
    The contributions of this work are summarized as follows:
    We propose mKG-RAG, a novel multimodal knowledge-augmented generation framework that integrates RAG with multimodal KGs to enhance the knowledge reasoning of MLLMs. To the best of our knowledge, this is the first work to investigate the potential of multimodal knowledge graphs in knowledge-intensive VQA tasks.
    Our framework develops a multimodal KG construction pipeline, allowing the extraction of image-text aligned entities and relations from multimodal documents. Additionally, a dual-stage retrieval schema with a question-aware multimodal retriever enables us to unleash the potential of RAG incorporated with multimodal KGs.
    Extensive experiments demonstrate that mKG-RAG significantly outperforms strong baselines, setting new state-of-the-art results on E-VQA and InfoSeek.
    ---------------------- this was the passage that we matched to the query -------------
    5 Conclusion
    Wepropose mKG-RAG, a novel retrieval-augmented generation framework that integrates multimodal knowledge graphs (KGs) to overcome the knowledge limitations of multimodal large language models (MLLMs). Our framework constructs structured, modality-aligned KGs using MLLM-driven keyword extraction and cross-modal alignment, and employs a dual-stage retrieval system, which combines vector-based and graph-based retrieval for precise knowledge augmentation. Extensive experiments show that mKG-RAG outperforms state-of-the-art methods, with ablation studies validating each component's contributions.
    --------------------------------------------------------------------------------------
[end on page 10]
```

### Document #4 — LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval
- `doc_id`: `doc::6c692b0a4ac04b1b9313ee01e24e7dd7`
- **Last Updated:** 2025-11-04T16:17:39.364403+00:00
- **Context:** LeanRAG advances RAG methodologies for improved semantic retrieval in AI language models.
- **Content fetched inside document:**
```text
[start on page 3]
    ---------------------- this was the passage that we matched to the query -------------
    Knowledge Graph Based Retrieval-Augmented Generation
    To better capture the relational nature of information, KGbased RAG has emerged as a prominent research direction. By representing knowledge as a graph of entities and relations, these methods aim to provide a more structured and semantically rich context for the LLM (Peng et al. 2024). Early approaches in this domain focused on leveraging graph structures for improved retrieval. For instance, GraphRAG (Edge et al. 2024) organizes documents into community-based KGs to preserve local context, while other methods like FastGraphRAG utilize graph-centrality metrics such as PageRank (Page et al. 1999) to prioritize more important nodes during retrieval. This subgraph retrieval approach has also proven effective in industrial applications like customer service, where KGs are constructed from historical support tickets to provide structured context (Xu et al. 2024). These methods marked a significant step forward by imposing a macro-structure onto the knowledge base, moving beyond disconnected text chunks.
    --------------------------------------------------------------------------------------
    Recognizing the need for more fine-grained control and abstraction, subsequent works have explored more sophisticated hierarchical structures. HiRAG, the current stateof-the-art, clusters entities to form multi-level summaries (Huang et al. 2025a), while LightRAG (Guo et al. 2024) proposes a dual-level framework to balance global and local information retrieval. While these hierarchical methods have progressively improved retrieval quality, a critical gap persists in how the constructed graph structures are leveraged at query time. The retrieval process is often decoupled from the indexing structure; for instance, an initial search may be performed over a 'flattened' list of all nodes, rather than being directly guided by the indexed community or hierarchical relations. This decoupling means the rich structural information is primarily used for post-retrieval context expansion, rather than for guiding the initial, crucial step of identifying relevant information. This can limit performance on complex queries where the relations between entities are paramount, highlighting the need for a new paradigm where the retrieval process is natively co-designed with the knowledge structure.
[end on page 3]
```
## LLM Answers (14.00s - 20.00s)
**Tech:** OpenAI's GPT-5 (reasoning low)

**Result:**

Neither is universally “better.” It depends on your needs.

When Knowledge Graphs help:
- Complex, relational, or multi-hop questions: Graph-based RAG represents entities and relations, giving more structured, semantically rich context than plain chunk retrieval [4]. This helps synthesize cross‑document insights and support verifiable claims with source links (e.g., GraphRAG) [1].
- Noisy corpora or multimodal tasks: Plain RAG can pull irrelevant text; KG-based methods reduce noise by retrieving structured facts. In VQA, multimodal KGs improved precision and outperformed strong baselines on E‑VQA and InfoSeek [3].
- Domains needing structure and traceability (e.g., customer support KGs) [4].

Trade-offs of KG-based RAG:
- Cost and complexity: Building and maintaining graphs (including multimodal) is hard; large KGs expand the search space and make retrieval slower [3]. Traditional GraphRAG can suffer from high construction cost and fixed, one-shot retrieval; agentic approaches (Graph‑R1) aim to fix this via lightweight hypergraphs and RL-driven multi‑turn retrieval [2]. Some methods also rely on long-context prompts and careful prompt design [2]. Additionally, some hierarchical graph systems still don’t tightly couple retrieval with the graph structure at query time, limiting gains [4].

When plain RAG is enough:
- Straightforward fact lookup or when you lack graph infrastructure. It’s simpler and cheaper to deploy, but can miss contextually precise, structured relationships and may retrieve irrelevant passages [1][3].

Rule of thumb:
- Start with plain RAG for simple FAQs and low-stakes tasks.
- Prefer KG‑RAG (or GraphRAG variants) for complex reasoning, cross‑document synthesis, and when you need evidence-linked, reliable answers; consider newer agentic/graph-guided retrieval to mitigate costs and rigidity [1][2][4].

**Reasoning:** Synthesized pros/cons and use cases from documents: KG benefits and GraphRAG description [1], RL-based improvements and challenges [2], multimodal KG advantages and plain RAG noise issues with benchmark gains [3], and broader KG-RAG advances, limitations, and industrial use [4].

**Citations:**

[1] doc::6371023da29b4bbe8242ffc5caf4a8cd — Fusing Knowledge and Language: A Comparative Study of Knowledge Graph-Based Question Answering with LLMs

[2] doc::2d32ad4c9bce416ca4c233fab33bb7a9 — Graph-R1: Towards Agentic GraphRAG Framework via End-to-end Reinforcement Learning

[3] doc::45dfa5e3daff4215b7e4c6c64b4497f5 — mKG-RAG: Multimodal Knowledge Graph-Enhanced RAG for Visual Question Answering

[4] doc::6c692b0a4ac04b1b9313ee01e24e7dd7 — LeanRAG: Knowledge-Graph-Based Generation with Semantic Aggregation and Hierarchical Retrieval




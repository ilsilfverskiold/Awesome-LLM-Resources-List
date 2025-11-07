# Query 3

## Result 
(latency is ~23.16s seconds end-to-end)
(retrieval is ~5.3 seconds)

**Original Query:** why is everyone saying RAG doesn’t scale? how are people fixing that?

**LLM answer:**

People say RAG “doesn’t scale” because several bottlenecks show up as data and users grow: (1) maintaining huge knowledge stores and fast indices for millions/billions of docs demands heavy infrastructure, distributed systems, and tight cost control [2]; (2) retrieval and long-context processing add latency compared to a standalone LLM, so responses slow down as k and corpora grow [2]; (3) LLMs can ignore or be confused by long, noisy contexts, and limited input windows force truncation, hurting accuracy at scale [2]; (4) naive setups often run full‑precision generator models, which spike memory and inference costs [1]; (5) updating corpora frequently (freshness) requires constant re-encoding and index sync, which is operationally hard [2]; (6) multi-part pipelines (retrievers, rankers, generators, graphs) are complex to maintain at production scale [2].

How people are fixing it:
- Infrastructure and retrieval: advanced/approximate nearest neighbor indexing (e.g., HNSW/FAISS), caching, cascaded/multi-tier and distributed retrieval, selective indexing, corpus curation, and parallel retrieval to keep throughput high with large corpora [2].
- Generation cost: use quantized LLMs so the model’s role is lightweight, cutting VRAM and inference time; K2RAG reports 3× lower VRAM and up to 40% faster execution vs naive pipelines [1].
- Data and pipeline design: summarize/condense training corpora to shrink build/update times (e.g., 93% reduction reported), and combine dense+spark/sparse search with knowledge graphs to retrieve more relevant passages with less noise [1].
- Long-context robustness: methods like BEE‑RAG re-balance attention so performance stays stable as context grows, improving scalability for long documents and multi-hop tasks [4].
- Freshness: incremental updates and hybrid/live retrieval to keep answers current without frequent full retraining [2].
- Security at scale: attribution/defense tools (e.g., RAGOrigin) that remain effective on very large knowledge bases (tested at 16.7M texts) help keep large deployments reliable [3].

**Citations:**

[1] doc::6d7a7d88cfc04636b20931fdf22f1e61 — KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities

[2] doc::688cfbc0abdc4520a73e219ac26aff41 — A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions

[3] doc::11e21d1e2c53420baf3c0c12575ab566 — Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation

[4] doc::b8d85d9737e544a98d7494a7811714ba — BEE-RAG: Balanced Entropy Engineering for Retrieval-Augmented Generation


--------------------------------------------------------------------------------------------------------------------------------------

# RAG Pipeline

**Latency Breakdown:**

- Query optimizer: 2.70s (boots up Qdrant client in parallel)
- Retrieve: 0.71s
- Rerank: 0.74s
- Context expansion: 1.11 s
- LLM answer: 14–20 s

## Query Optimizer (2.70s)

**Tech:** GPT-4o-mini, structured responses

Generated queries:
- hybrid: issues with RAG scalability
- hybrid: solutions for RAG scalability problems


## Retrieve (0.71s)

**Tech:** Qdrant, LlamaIndex

**Results:**

Query 1 (hybrid) top 20 for query: issues with RAG scalability

[1] score=0.5000 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C05::251104142800
  text: 7 Challenges of RAG 7.2.1 Scalability and Infrastructure Deploying RAG at scale requires substantial engineering to maintain large knowledge corpora and efficient retrieval indices. Systems must handle millions or billions of documents, demanding significant computational resources, efficient indexing, distributed computing infrastructure, and cost management strategies [21]. Efficient indexing methods, caching, and multi-tier retrieval approaches (such as cascaded retrieval) become essential at scale, especially in large deployments like web search engines.

[2] score=0.5000 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=SDOC::SUM::251104135247
  text: This paper proposes the KeyKnowledgeRAG (K2RAG) framework to enhance the efficiency and accuracy of Retrieval-Augment-Generate (RAG) systems. It addresses the high computational costs and scalability issues associated with naive RAG implementations by incorporating techniques such as knowledge graphs, a hybrid retrieval approach, and document summarization to reduce training times and improve answer accuracy. Evaluations show that K2RAG significantly outperforms traditional implementations, achieving greater answer similarity and faster execution times, thereby providing a scalable solution for companies seeking robust question-answering systems.

[3] score=0.4293 doc=docs_ingestor/docs/arxiv/2508.10701.pdf chunk=S5::C01::251104162516
  text: B. Issues with current approaches Scalability issue : Current vulnerability remediation mechanisms face significant scalability challenges. The sheer diversity of vulnerable devices and systems-encompassing complex infrastructures, numerous software applications, and millions of endpoints-makes broad protection difficult. A prime example is the Log4j vulnerability, which impacted devices ranging from Apache servers to consumer appliances like Siemens refrigerators. Manually generating and validating patches or filtering rules for each unique vulnerabilitydevice combination demands immense domain expertise and is impractical at scale. Furthermore, patch deployment itself consumes substantial time, manpower, and technical resources, a burden particularly acute for organizations with limited budgets. This scalability gap is exacerbated by the emergence of LLM-empowered exploitation tools (e.g., HackerGPT [31], WormGPT [29]), which dramatically enhance attackers' ability to launch large-scale exploits.

[4] score=0.4204 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C42::251104142800
  text: 7 Challenges of RAG 7.5.5 Scalability Scalability challenges arise as knowledge corpora expand. Advanced indexing, distributed retrieval, and approximate nearest neighbor techniques facilitate efficient handling of large-scale knowledge bases [57]. Selective indexing and corpus curation, combined with infrastructure improvements like caching and parallel retrieval, allow RAG systems to scale to massive knowledge repositories. Research indicates that moderate-sized models augmented with large external corpora can outperform significantly larger standalone models, suggesting parameter efficiency advantages [10].

[5] score=0.4120 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S4::C10::251104135247
  text: 1 Introduction Solution: We propose the K 2 RAG framework which addresses the following 4 research goals based on the aforementioned answer accuracy and scalability issues characteristic of naive RAG implementations: Goal 1. Reduce information store such as Knowledge Graph and Spare and Dense vector database creation times.

[6] score=0.3951 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S3::C01::251104135247
  text: ABSTRACT Fine-tuning is an immensely resource expensive process when trying to retrain Large Language Models (LLMs) to have access to a larger bank of knowledge. To alleviate this issue there have been many different fine-tuning techniques proposed which have shown good progress in trying to reduce time and computational resources to achieve fine-tuning but with LLMs becoming more intelligent and larger, this issue continues to arise. Hence a new method of enabling knowledge expansion on LLMs had to be devised. Retrieval-Augment-Generate (RAG) is a class of techniques where information is stored in a database and appropriate chunks of information are retrieved to help answer the question. However there are many limitations to naive RAG implementations. This paper proposes the KeyKnowledgeRAG ( K 2 RAG ) framework to address the scalability and answer accuracy limitations associated with naive RAG implementations. This framework takes inspiration from divide-and-conquer ideology, and combines dense and sparse vector search, k

[7] score=0.3861 doc=docs_ingestor/docs/arxiv/2508.10701.pdf chunk=S1::C02::251104162516
  text: REFN: A Reinforcement-Learning-From-Network Framework against 1-day/n-day Exploitations Abstract -The exploitation of 1-day/n-day vulnerabilities poses severe threats to networked devices due to massive deployment scales and delayed patching (average Mean-Time-To-Patch exceeds 60 days). Existing defenses, including host-based patching and network-based filtering, are inadequate due to limited scalability across diverse devices, compatibility issues especially with embedded/legacy systems, and error-prone deployment process (e.g., manual patch validation). To address these issues, we introduce REFN (Reinforcement-Learning-From-Network), a novel framework that trains Large Language Models (LLMs) to autonomously generate network filters to prevent 1-day/nday exploitations. REFN ensures scalability by uniquely employs Reinforcement Learning (RL) driven by online network rewards instead of traditional Human Feedback (RLHF). REFN guarantees compatibility via unified deployment on edge security gateways (e.g., Amazon Eero). REFN pr

[8] score=0.3157 doc=docs_ingestor/docs/arxiv/2509.03934.pdf chunk=S8::C01::251104172307
  text: Limitations While SelfAug is designed as a plug-and-play approach that integrates seamlessly with both LoRA and full-parameter fine-tuning, we did not conduct extensive experiments on full-parameter settings due to computational constraints. For extremely long input contexts exceeding 32,000 tokens, our method may face scalability issues, as aligning logits across the entire sequence can be costly. We recommend using attention-based or importance-sampling mechanisms to focus only on the most critical tokens. Future work will explore the effectiveness and scalability of SelfAug in fullparameter fine-tuning settings, potentially revealing additional insights into its broader applicability across different training paradigms.

[9] score=0.3009 doc=docs_ingestor/docs/arxiv/2507.21753.pdf chunk=S1::C01::251104152703
  text: Towards a rigorous evaluation of RAG systems : the challenge of due diligence The rise of generative AI, has driven significant advancements in high-risk sectors like healthcare and finance. The Retrieval-Augmented Generation (RAG) architecture, combining language models (LLMs) with search engines, is particularly notable for its ability to generate responses from document corpora. Despite its potential, the reliability of RAG systems in critical contexts remains a concern, with issues such as hallucinations persisting. This study evaluates a RAG system used in due diligence for an investment fund. We propose a robust evaluation protocol combining human annotations and LLM-Judge annotations to identify system failures, like hallucinations, off-topic, failed citations, and abstentions. Inspired by the Prediction Powered Inference (PPI) method, we achieve precise performance measurements with statistical guarantees. We provide a comprehensive dataset for further analysis. Our contributions aim to enhance the reliability and sc

[10] score=0.2721 doc=docs_ingestor/docs/arxiv/2507.10382.pdf chunk=S14::C01::251104135349
  text: VI. CONCLUSION AND FUTURE WORKS This paper presents a comprehensive shared e-mobility platform that integrates cloud-based simulation, real-time data processing, and RAG-enhanced decision-making to address the challenges in sustainable urban transportation. The platform supports dynamic multi-modal routing and flexible docking-based systems through a scalable and usercentric design. The schema-level RAG evaluation highlights XiYanSQL's high execution accuracy of 0.81 on system operator queries and 0.98 on user queries. An error analysis reveals that logic errors are the predominant failure type for most models, while granularity issues are more common in user-level queries. Although the LLM-based RAG framework performs robustly, the generated responses are not always perfect. Users and system operators should treat LLM outputs as supportive decision-making aids rather than definitive conclusions and should cross-check critical decisions against simulation data or dashboard outputs when high reliability is required.

[11] score=0.2678 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C38::251104142800
  text: 7 Challenges of RAG 7.5.2 Latency RAG systems introduce latency due to retrieval processes. Solutions include using efficient nearest-neighbor search structures, such as HNSW graphs, which significantly speed up similarity searches [57]. Caching mechanisms, including multi-level and approximate embedding caches (e.g., RAGCache and Proximity cache), enable reuse of previously retrieved information, drastically reducing retrieval time [40, 8]. Adaptive retrieval methods dynamically balance retrieval complexity based on query difficulty, optimizing overall throughput and reducing latency.

[12] score=0.2487 doc=docs_ingestor/docs/arxiv/2507.10382.pdf chunk=S4::C01::251104135349
  text: A. Transportation Simulation Platform Classical traffic simulation platforms, such as AIMSUN [4] and VISSIM [5], primarily address issues like congestion and scheduling through microscopic modeling [12]. SUMO [6] further introduced a hybrid approach combining different mathematical models for algorithm evaluation. While traditional platforms focus on congestion and accident analysis, recent efforts have begun integrating e-mobility to assess impacts on energy, urban economies, and communities. Examples include digital twin-based platforms linking e-mobility and smart grids [13], sustainable EV adoption tools [14], and data-driven EV impact simulations [15]. Although studies have explored shared e-mobility impacts [16], [17], few have developed scalable simulation platforms with native support for shared e-mobility and eHub distribution, which is often oversimplified in smart city contexts [18]. [7] proposed a multi-agent platform using deep reinforcement learning for EV fleet management, but research on scalable, multi-modal

[13] score=0.2404 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C03::251104142800
  text: 7 Challenges of RAG 7.1.2 Latency and Efficiency RAG inherently increases computational complexity and latency compared to standalone LLMs due to retrieval overhead, vector searches, and expanded context processing. Techniques like approximate nearest neighbor indices (e.g., FAISS, HNSW), caching, model distillation, or lightweight retrievers can reduce latency at the expense of accuracy. Integrating retrieval efficiently with large language models (LLMs) and ensuring rapid responses in real-time scenarios (e.g., customer support) remains a significant challenge [3]. Interestingly, using retrieval can allow smaller models to match the performance of larger models without retrieval (e.g., RETRO, Atlas), reducing model size requirements but shifting complexity to maintaining external knowledge bases and infrastructure.

[14] score=0.2044 doc=docs_ingestor/docs/arxiv/2509.20707.pdf chunk=S7::C06::251104185442
  text: 1 Introduction To address the need for accurate, interpretable and scalable plan evaluation, we introduce an automated RAG system powered by LLaMA-4 109B for radiotherapy quality assessment. Our framework integrates (1) a scoring module that computes normalized dose metrics and population-based percentiles, (2) a retrieval module that identifies similar historical plans based on numerical and textual features, and (3) a constraint-checking tool that flags clinical violations using protocol-defined thresholds. These components are leveraged through an LLM-driven reasoning process that issues explicit tool calls to retrieve context, perform checks, and synthesize a structured, protocol-aware summary of plan quality. Unlike prior end-to-end systems, our method separates data, logic, and generation, reducing hallucinations while supporting traceability and flexible protocol integration. The output includes a quantitative plan score and a list of failed constraints, enabling transparent decision support aligned with clinical prac

[15] score=0.1850 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S21::C03::251104142800
  text: Generation Quality. Evaluated using: Memory and compute requirements -Especially important for deploying RAG systems at scale. Scalability. As corpus size grows, the system's ability to maintain retrieval quality and generation fidelity is tested. Evaluation considers: Index size vs. retrieval accuracy . Adaptability to new or evolving data without retraining .

[16] score=0.1780 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C12::251104142800
  text: 7 Challenges of RAG 7.4 Application Domains and Case Studies RAGsystems comprise multiple components-retrievers, rerankers, indexes, and LLMs-resulting in increased complexity and potential points of failure. Maintenance includes synchronizing knowledge updates, managing access controls, orchestrating prompts, and handling multi-turn dialogues. Robust evaluation methods must assess end-to-end performance, retrieval quality, and faithfulness of model outputs to the evidence [3].

[17] score=0.1565 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C37::251104142800
  text: 7 Challenges of RAG 7.5.1 Retrieval Quality : Maintaining high retrieval relevance is critical for effective RAG. Strategies to improve retrieval quality include domain-adaptive training, advanced encoders, and query reformulation methods to address vocabulary mismatches [85]. Employing reranking models further boosts relevance by re-scoring initial retrieval results with deeper contextual analysis, enhancing accuracy at the expense of additional computation [4]. Iterative retrieval and chain-of-thought reasoning represent future directions, breaking down complex queries into simpler sub-queries, thus ensuring relevant information retrieval at each reasoning step [90].

[18] score=0.1474 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C01::251104142800
  text: 7 Challenges of RAG This section discusses the challenges of RAG, cases of manifestation of such challenges in the selected domain of RAG application, and outlines existing solutions and the way forward.

[19] score=0.1270 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C06::251104142800
  text: 7 Challenges of RAG 7.2.2 Freshness and Knowledge Updates One motivation for RAG is providing current information. However, continuously updating external knowledge bases and retrieval indices is challenging. Domains requiring real-time updates (e.g., finance, healthcare) demand sophisticated data pipelines for incremental updates, possibly frequent re-encoding of documents, and synchronization of retrieval indices. Delays in updates or inconsistencies between the LLM's internal knowledge and newly retrieved data can produce outdated or contradictory answers [61].

[20] score=0.1201 doc=docs_ingestor/docs/arxiv/2508.10701.pdf chunk=S6::C01::251104162516
  text: C. New vantage point to prevent 1-day/n-day vulnerabilities To tackle the compatibility and scalability issue, security vendors are shifting the vulnerability fixing function from host-side to Edge Security Gateways (ESG) , including Amazon eero [2], Cisco Meraki [3], Netgear Orbi [6] and Linksys Velop [5]. In such network-fix paradigm, the vulnerability fixing is enforced as network filtering on the edge security gateways. The remote cloud services is responsible for generating the filtering rules and installing them on the edge security gateways. For example, the Cisco Talos Intelligence cloud service can generate a network filtering for Log4j, and deploy it on Meraki MX edge routers to detect and block Log4j exploits [4], [7]. The gateways hosting the vulnerability fixes are unified platforms such as Cisco IOS [13], Rasberry PI [24] or OpenWRT [22]. The network-based patches only need to adapt to several unified edge platforms instead of heterogeneous vulnerable devices. Unlike current host-based patching mechanisms - whi

Query 2 (hybrid) top 20 for query: solutions for RAG scalability problems

[1] score=0.5000 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C05::251104142800
  text: 7 Challenges of RAG 7.2.1 Scalability and Infrastructure Deploying RAG at scale requires substantial engineering to maintain large knowledge corpora and efficient retrieval indices. Systems must handle millions or billions of documents, demanding significant computational resources, efficient indexing, distributed computing infrastructure, and cost management strategies [21]. Efficient indexing methods, caching, and multi-tier retrieval approaches (such as cascaded retrieval) become essential at scale, especially in large deployments like web search engines.

[2] score=0.5000 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=SDOC::SUM::251104135247
  text: This paper proposes the KeyKnowledgeRAG (K2RAG) framework to enhance the efficiency and accuracy of Retrieval-Augment-Generate (RAG) systems. It addresses the high computational costs and scalability issues associated with naive RAG implementations by incorporating techniques such as knowledge graphs, a hybrid retrieval approach, and document summarization to reduce training times and improve answer accuracy. Evaluations show that K2RAG significantly outperforms traditional implementations, achieving greater answer similarity and faster execution times, thereby providing a scalable solution for companies seeking robust question-answering systems.

[3] score=0.4368 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C42::251104142800
  text: 7 Challenges of RAG 7.5.5 Scalability Scalability challenges arise as knowledge corpora expand. Advanced indexing, distributed retrieval, and approximate nearest neighbor techniques facilitate efficient handling of large-scale knowledge bases [57]. Selective indexing and corpus curation, combined with infrastructure improvements like caching and parallel retrieval, allow RAG systems to scale to massive knowledge repositories. Research indicates that moderate-sized models augmented with large external corpora can outperform significantly larger standalone models, suggesting parameter efficiency advantages [10].

[4] score=0.4321 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S4::C10::251104135247
  text: 1 Introduction Solution: We propose the K 2 RAG framework which addresses the following 4 research goals based on the aforementioned answer accuracy and scalability issues characteristic of naive RAG implementations: Goal 1. Reduce information store such as Knowledge Graph and Spare and Dense vector database creation times.

[5] score=0.3647 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S4::C08::251104135247
  text: 1 Introduction Scalability: Naive implementations of Retrieval-Augmented Generation (RAG) often rely on 16-bit floating-point large language models (LLMs) for the generation component. However, this approach introduces significant scalability challenges due to the increased memory demands required to host the LLM as well as longer inference times due to using a higher precision number type. To enable more efficient scaling, it is crucial to integrate methods or techniques that reduce the memory footprint and inference times of generator models. Quantized models offer more scalable solutions due to less computational requirements, hence when developing RAG systems we should aim to use quantized LLMs for more cost effective deployment as compared to a full fine-tuned LLM whose performance might be good but is more expensive to deploy due to higher memory requirements. A quantized LLM's role in the RAG pipeline itself should be minimal and for means of rewriting retrieved information into a presentable fashion for the end users

[6] score=0.3314 doc=docs_ingestor/docs/arxiv/2508.12682.pdf chunk=S10::C01::251104163000
  text: RAG pipeline We construct our RAG pipeline using RAGFlow, an opensource framework chosen for its scalability and flexibility. To maximize retrieval accuracy, we process terminology knowledge and factual knowledge differently.

[7] score=0.3055 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C38::251104142800
  text: 7 Challenges of RAG 7.5.2 Latency RAG systems introduce latency due to retrieval processes. Solutions include using efficient nearest-neighbor search structures, such as HNSW graphs, which significantly speed up similarity searches [57]. Caching mechanisms, including multi-level and approximate embedding caches (e.g., RAGCache and Proximity cache), enable reuse of previously retrieved information, drastically reducing retrieval time [40, 8]. Adaptive retrieval methods dynamically balance retrieval complexity based on query difficulty, optimizing overall throughput and reducing latency.

[8] score=0.2965 doc=docs_ingestor/docs/arxiv/2507.10382.pdf chunk=S3::C01::251104135349
  text: II. LITERATURE REVIEW Recent developments in e-mobility platforms aim to tackle urban congestion, environmental concerns, and transport optimization. However, current solutions often lack integration of energy management, real-time cloud simulation, and user-centric features. This section reviews classical traffic simulation platforms, e-mobility platforms, RAG-powered transportation systems, and commercial and public shared emobility platforms, highlighting their strengths, limitations, and gaps. It also emphasizes our platform's unique contributions in providing a scalable, multi-modal, and user-focused approach to shared e-mobility simulation and optimization.

[9] score=0.2402 doc=docs_ingestor/docs/arxiv/2508.12682.pdf chunk=S3::C05::251104163000
  text: Introduction Recently, retrieval-augmented generation (RAG) has emerged as a cost-effective alternative to fine-tuning, offering higher accuracy and more flexible deployment for enterprise applications (Lewis et al. 2021; Fan et al. 2024). By enabling large language models (LLMs) to seamlessly access external knowledge bases during inference, RAG reduces hallucinations and improves factual consistency without retraining. Constructing such knowledge bases is straightforward: raw documents are split into chunks, embedded, and stored in vector databases-an approach that can be fully deployed locally without extensive data cleaning. As a result, RAGprovides a private, scalable, and low-overhead solution for adapting LLMs to industrial use cases (Balaguer et al. 2024).

[10] score=0.1865 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C03::251104142800
  text: 7 Challenges of RAG 7.1.2 Latency and Efficiency RAG inherently increases computational complexity and latency compared to standalone LLMs due to retrieval overhead, vector searches, and expanded context processing. Techniques like approximate nearest neighbor indices (e.g., FAISS, HNSW), caching, model distillation, or lightweight retrievers can reduce latency at the expense of accuracy. Integrating retrieval efficiently with large language models (LLMs) and ensuring rapid responses in real-time scenarios (e.g., customer support) remains a significant challenge [3]. Interestingly, using retrieval can allow smaller models to match the performance of larger models without retrieval (e.g., RETRO, Atlas), reducing model size requirements but shifting complexity to maintaining external knowledge bases and infrastructure.

[11] score=0.1772 doc=docs_ingestor/docs/arxiv/2508.05100.pdf chunk=S3::C06::251104155301
  text: Introduction Empirical analyses across multiple real-world benchmarks reveal that BEE-RAG fundamentally alters the entropy scaling laws governing conventional RAG systems, which provides a robust and scalable solution for RAG systems dealing with long-context scenarios. Our main contributions are summarized as follows: We introduce the concept of balanced context entropy, a novel attention reformulation that ensures entropy invariance across varying context lengths, and allocates attention to important segments. It addresses the critical challenge of context expansion in RAG.

[12] score=0.1705 doc=docs_ingestor/docs/arxiv/2507.19562.pdf chunk=S12::C04::251104143937
  text: B. Retrieval-Augmented Generation (RAG) for Code Generalization PennyCoder combines the strengths of instruction fine-tuning and lightweight parameter-efficient adaptation techniques to deliver a robust, locally deployable LLM solution for PennyLane quantum programming. By addressing both data scarcity and compute constraints, PennyCoder provides a practical path forward for scalable, privacypreserving quantum code generation.

[13] score=0.1486 doc=docs_ingestor/docs/arxiv/2507.19666.pdf chunk=S4::C03::251104145914
  text: 2 Related Work However , these studies evaluate out-of-the-box LLM or VLM performance; they do not aim to optimize model accuracy and do not incorporate retrieval augmentation techniques. While GPT models have likely encountered many laws and question pairs during pretraining, reframing this task as a Retrieval-Augmented Generation (RAG) problem could offer a more principled and scalable approach to improving legal and regulation-specific reasoning.

[14] score=0.1305 doc=docs_ingestor/docs/arxiv/2507.10382.pdf chunk=S2::C05::251104135349
  text: I. INTRODUCTION To address these challenges, there is a growing need for cloud-based SUMO solutions that leverage the scalability of cloud platforms. Recent advancements in cloud-based SUMO focus on enhancing large-scale simulations and realtime data processing, but most studies concentrate on vehicular network optimization rather than shared e-mobility. [8] offers a high-level cloud-based SUMO framework, but it may be limited to simple queries or purely dashboardbased designs, which are not user-friendly. Meanwhile, the emergence of Large Language Models (LLMs) has opened new opportunities for enhancing intelligent decision-making in traffic-related domains, including delivery scheduling, location prediction, and secure communications. Building on this potential, we propose an LLM-powered, cloudbased platform that integrates dynamic traffic simulation with natural language interaction, aiming to bridge the gap between traditional simulation tools and user-friendly, adaptive shared mobility solutions. In particular, our plat

[15] score=0.1128 doc=docs_ingestor/docs/arxiv/2509.13772.pdf chunk=S11::C02::251104182521
  text: 7. Discussion and Limitations Scalability of RAGOrigin: We extend our evaluation by scaling the NQ dataset's knowledge database to 16.7 million texts, combining entries from the knowledge database of NQ, HotpotQA, and MS-MARCO. Using the same user questions from NQ, we assess RAGOrigin's performance under larger data volumes. As shown in Table 16, RAGOrigin maintains consistent effectiveness and performance even on this significantly expanded database. These results demonstrate that RAGOrigin remains robust at scale, making it suitable for enterprise-level applications requiring large

[16] score=0.1049 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S21::C03::251104142800
  text: Generation Quality. Evaluated using: Memory and compute requirements -Especially important for deploying RAG systems at scale. Scalability. As corpus size grows, the system's ability to maintain retrieval quality and generation fidelity is tested. Evaluation considers: Index size vs. retrieval accuracy . Adaptability to new or evolving data without retraining .

[17] score=0.1035 doc=docs_ingestor/docs/arxiv/2508.04442.pdf chunk=S7::C06::251104155058
  text: 5 Discussion 5.3 Linguistic and Curricular Nuances in a Malaysian Context The study confirmed GPT-4o's proficiency in handling specific Bahasa Melayu mathematical terminology like integer, pecahan (fraction), perpuluhan (decimal), and tertib menurun (descending order) as used in the source documents.[1] The RAG methods were particularly effective at adopting the precise phrasing and style of the teacher-prepared notes. This, however, reveals a deeper nuance in the concept of 'curriculum alignment.' The automated metrics used in this study excel at verifying topical and factual alignment. For example, the STS score confirms a question is about integers, and the RAG-QA check confirms it is answerable based on the curriculum's scope. Yet, the RPT specifies not just topics but also cognitive skills. Learning standard 1.2.6, 'Menyelesaikan masalah yang melibatkan integer' (Solving problems involving integers), requires a higher cognitive level than standard 1.1.2, 'Mengenal dan memerihalkan integer' (Recognizing and describing in

[18] score=0.0861 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C37::251104142800
  text: 7 Challenges of RAG 7.5.1 Retrieval Quality : Maintaining high retrieval relevance is critical for effective RAG. Strategies to improve retrieval quality include domain-adaptive training, advanced encoders, and query reformulation methods to address vocabulary mismatches [85]. Employing reranking models further boosts relevance by re-scoring initial retrieval results with deeper contextual analysis, enhancing accuracy at the expense of additional computation [4]. Iterative retrieval and chain-of-thought reasoning represent future directions, breaking down complex queries into simpler sub-queries, thus ensuring relevant information retrieval at each reasoning step [90].

[19] score=0.0810 doc=docs_ingestor/docs/arxiv/2507.23242.pdf chunk=S12::C07::251104154436
  text: RAG System Implementation | | Query Rewriter | Document Recognizer | Document Retriever | Train Data | |---:|:------------------|:----------------------|:---------------------|:---------------| | 0 | RL-QR multi-modal | ColQwen2.5-v0.2 | ColQwen2.5-v0.2 | D mm D tm D tm | | 1 | RL-QR lexical | AI Parser | ixi-RAG lexical | | | 2 | RL-QR semantic | AI Parser | ixi-RAG semantic | | | 3 | RL-QR hybrid | AI Parser | ixi-RAG hybrid | D tm | Table 2: RL-QR Training Scheme rocal Rank Fusion (RRF) algorithm (Cormack, Clarke, and Buettcher 2009). RRF calculates a new score for each document based on its position in the individual rankings, producing a single, more robustly ordered list that balances keyword relevance with semantic similarity. This method effectively captures both the precision of lexical search and the contextual understanding of semantic search to deliver a highly refined final ranking.

[20] score=0.0810 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C12::251104142800
  text: 7 Challenges of RAG 7.4 Application Domains and Case Studies RAGsystems comprise multiple components-retrievers, rerankers, indexes, and LLMs-resulting in increased complexity and potential points of failure. Maintenance includes synchronizing knowledge updates, managing access controls, orchestrating prompts, and handling multi-turn dialogues. Robust evaluation methods must assess end-to-end performance, retrieval quality, and faithfulness of model outputs to the evidence [3].

RRF Fusion top 31 for query: why is everyone saying RAG doesn’t scale? how are people fixing that?

[1] score=0.0328 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C05::251104142800
  text: 7 Challenges of RAG 7.2.1 Scalability and Infrastructure Deploying RAG at scale requires substantial engineering to maintain large knowledge corpora and efficient retrieval indices. Systems must handle millions or billions of documents, demanding significant computational resources, efficient indexing, distributed computing infrastructure, and cost management strategies [21]. Efficient indexing methods, caching, and multi-tier retrieval approaches (such as cascaded retrieval) become essential at scale, especially in large deployments like web search engines.

[2] score=0.0323 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=SDOC::SUM::251104135247
  text: This paper proposes the KeyKnowledgeRAG (K2RAG) framework to enhance the efficiency and accuracy of Retrieval-Augment-Generate (RAG) systems. It addresses the high computational costs and scalability issues associated with naive RAG implementations by incorporating techniques such as knowledge graphs, a hybrid retrieval approach, and document summarization to reduce training times and improve answer accuracy. Evaluations show that K2RAG significantly outperforms traditional implementations, achieving greater answer similarity and faster execution times, thereby providing a scalable solution for companies seeking robust question-answering systems.

[3] score=0.0315 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C42::251104142800
  text: 7 Challenges of RAG 7.5.5 Scalability Scalability challenges arise as knowledge corpora expand. Advanced indexing, distributed retrieval, and approximate nearest neighbor techniques facilitate efficient handling of large-scale knowledge bases [57]. Selective indexing and corpus curation, combined with infrastructure improvements like caching and parallel retrieval, allow RAG systems to scale to massive knowledge repositories. Research indicates that moderate-sized models augmented with large external corpora can outperform significantly larger standalone models, suggesting parameter efficiency advantages [10].

[4] score=0.0310 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S4::C10::251104135247
  text: 1 Introduction Solution: We propose the K 2 RAG framework which addresses the following 4 research goals based on the aforementioned answer accuracy and scalability issues characteristic of naive RAG implementations: Goal 1. Reduce information store such as Knowledge Graph and Spare and Dense vector database creation times.

[5] score=0.0290 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C38::251104142800
  text: 7 Challenges of RAG 7.5.2 Latency RAG systems introduce latency due to retrieval processes. Solutions include using efficient nearest-neighbor search structures, such as HNSW graphs, which significantly speed up similarity searches [57]. Caching mechanisms, including multi-level and approximate embedding caches (e.g., RAGCache and Proximity cache), enable reuse of previously retrieved information, drastically reducing retrieval time [40, 8]. Adaptive retrieval methods dynamically balance retrieval complexity based on query difficulty, optimizing overall throughput and reducing latency.

[6] score=0.0280 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C03::251104142800
  text: 7 Challenges of RAG 7.1.2 Latency and Efficiency RAG inherently increases computational complexity and latency compared to standalone LLMs due to retrieval overhead, vector searches, and expanded context processing. Techniques like approximate nearest neighbor indices (e.g., FAISS, HNSW), caching, model distillation, or lightweight retrievers can reduce latency at the expense of accuracy. Integrating retrieval efficiently with large language models (LLMs) and ensuring rapid responses in real-time scenarios (e.g., customer support) remains a significant challenge [3]. Interestingly, using retrieval can allow smaller models to match the performance of larger models without retrieval (e.g., RETRO, Atlas), reducing model size requirements but shifting complexity to maintaining external knowledge bases and infrastructure.

[7] score=0.0265 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S21::C03::251104142800
  text: Generation Quality. Evaluated using: Memory and compute requirements -Especially important for deploying RAG systems at scale. Scalability. As corpus size grows, the system's ability to maintain retrieval quality and generation fidelity is tested. Evaluation considers: Index size vs. retrieval accuracy . Adaptability to new or evolving data without retraining .

[8] score=0.0258 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C37::251104142800
  text: 7 Challenges of RAG 7.5.1 Retrieval Quality : Maintaining high retrieval relevance is critical for effective RAG. Strategies to improve retrieval quality include domain-adaptive training, advanced encoders, and query reformulation methods to address vocabulary mismatches [85]. Employing reranking models further boosts relevance by re-scoring initial retrieval results with deeper contextual analysis, enhancing accuracy at the expense of additional computation [4]. Iterative retrieval and chain-of-thought reasoning represent future directions, breaking down complex queries into simpler sub-queries, thus ensuring relevant information retrieval at each reasoning step [90].

[9] score=0.0257 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C12::251104142800
  text: 7 Challenges of RAG 7.4 Application Domains and Case Studies RAGsystems comprise multiple components-retrievers, rerankers, indexes, and LLMs-resulting in increased complexity and potential points of failure. Maintenance includes synchronizing knowledge updates, managing access controls, orchestrating prompts, and handling multi-turn dialogues. Robust evaluation methods must assess end-to-end performance, retrieval quality, and faithfulness of model outputs to the evidence [3].

[10] score=0.0159 doc=docs_ingestor/docs/arxiv/2508.10701.pdf chunk=S5::C01::251104162516
  text: B. Issues with current approaches Scalability issue : Current vulnerability remediation mechanisms face significant scalability challenges. The sheer diversity of vulnerable devices and systems-encompassing complex infrastructures, numerous software applications, and millions of endpoints-makes broad protection difficult. A prime example is the Log4j vulnerability, which impacted devices ranging from Apache servers to consumer appliances like Siemens refrigerators. Manually generating and validating patches or filtering rules for each unique vulnerabilitydevice combination demands immense domain expertise and is impractical at scale. Furthermore, patch deployment itself consumes substantial time, manpower, and technical resources, a burden particularly acute for organizations with limited budgets. This scalability gap is exacerbated by the emergence of LLM-empowered exploitation tools (e.g., HackerGPT [31], WormGPT [29]), which dramatically enhance attackers' ability to launch large-scale exploits.

[11] score=0.0154 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S4::C08::251104135247
  text: 1 Introduction Scalability: Naive implementations of Retrieval-Augmented Generation (RAG) often rely on 16-bit floating-point large language models (LLMs) for the generation component. However, this approach introduces significant scalability challenges due to the increased memory demands required to host the LLM as well as longer inference times due to using a higher precision number type. To enable more efficient scaling, it is crucial to integrate methods or techniques that reduce the memory footprint and inference times of generator models. Quantized models offer more scalable solutions due to less computational requirements, hence when developing RAG systems we should aim to use quantized LLMs for more cost effective deployment as compared to a full fine-tuned LLM whose performance might be good but is more expensive to deploy due to higher memory requirements. A quantized LLM's role in the RAG pipeline itself should be minimal and for means of rewriting retrieved information into a presentable fashion for the end users

[12] score=0.0152 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S3::C01::251104135247
  text: ABSTRACT Fine-tuning is an immensely resource expensive process when trying to retrain Large Language Models (LLMs) to have access to a larger bank of knowledge. To alleviate this issue there have been many different fine-tuning techniques proposed which have shown good progress in trying to reduce time and computational resources to achieve fine-tuning but with LLMs becoming more intelligent and larger, this issue continues to arise. Hence a new method of enabling knowledge expansion on LLMs had to be devised. Retrieval-Augment-Generate (RAG) is a class of techniques where information is stored in a database and appropriate chunks of information are retrieved to help answer the question. However there are many limitations to naive RAG implementations. This paper proposes the KeyKnowledgeRAG ( K 2 RAG ) framework to address the scalability and answer accuracy limitations associated with naive RAG implementations. This framework takes inspiration from divide-and-conquer ideology, and combines dense and sparse vector search, k

[13] score=0.0152 doc=docs_ingestor/docs/arxiv/2508.12682.pdf chunk=S10::C01::251104163000
  text: RAG pipeline We construct our RAG pipeline using RAGFlow, an opensource framework chosen for its scalability and flexibility. To maximize retrieval accuracy, we process terminology knowledge and factual knowledge differently.

[14] score=0.0149 doc=docs_ingestor/docs/arxiv/2508.10701.pdf chunk=S1::C02::251104162516
  text: REFN: A Reinforcement-Learning-From-Network Framework against 1-day/n-day Exploitations Abstract -The exploitation of 1-day/n-day vulnerabilities poses severe threats to networked devices due to massive deployment scales and delayed patching (average Mean-Time-To-Patch exceeds 60 days). Existing defenses, including host-based patching and network-based filtering, are inadequate due to limited scalability across diverse devices, compatibility issues especially with embedded/legacy systems, and error-prone deployment process (e.g., manual patch validation). To address these issues, we introduce REFN (Reinforcement-Learning-From-Network), a novel framework that trains Large Language Models (LLMs) to autonomously generate network filters to prevent 1-day/nday exploitations. REFN ensures scalability by uniquely employs Reinforcement Learning (RL) driven by online network rewards instead of traditional Human Feedback (RLHF). REFN guarantees compatibility via unified deployment on edge security gateways (e.g., Amazon Eero). REFN pr

[15] score=0.0147 doc=docs_ingestor/docs/arxiv/2509.03934.pdf chunk=S8::C01::251104172307
  text: Limitations While SelfAug is designed as a plug-and-play approach that integrates seamlessly with both LoRA and full-parameter fine-tuning, we did not conduct extensive experiments on full-parameter settings due to computational constraints. For extremely long input contexts exceeding 32,000 tokens, our method may face scalability issues, as aligning logits across the entire sequence can be costly. We recommend using attention-based or importance-sampling mechanisms to focus only on the most critical tokens. Future work will explore the effectiveness and scalability of SelfAug in fullparameter fine-tuning settings, potentially revealing additional insights into its broader applicability across different training paradigms.

[16] score=0.0147 doc=docs_ingestor/docs/arxiv/2507.10382.pdf chunk=S3::C01::251104135349
  text: II. LITERATURE REVIEW Recent developments in e-mobility platforms aim to tackle urban congestion, environmental concerns, and transport optimization. However, current solutions often lack integration of energy management, real-time cloud simulation, and user-centric features. This section reviews classical traffic simulation platforms, e-mobility platforms, RAG-powered transportation systems, and commercial and public shared emobility platforms, highlighting their strengths, limitations, and gaps. It also emphasizes our platform's unique contributions in providing a scalable, multi-modal, and user-focused approach to shared e-mobility simulation and optimization.

[17] score=0.0145 doc=docs_ingestor/docs/arxiv/2507.21753.pdf chunk=S1::C01::251104152703
  text: Towards a rigorous evaluation of RAG systems : the challenge of due diligence The rise of generative AI, has driven significant advancements in high-risk sectors like healthcare and finance. The Retrieval-Augmented Generation (RAG) architecture, combining language models (LLMs) with search engines, is particularly notable for its ability to generate responses from document corpora. Despite its potential, the reliability of RAG systems in critical contexts remains a concern, with issues such as hallucinations persisting. This study evaluates a RAG system used in due diligence for an investment fund. We propose a robust evaluation protocol combining human annotations and LLM-Judge annotations to identify system failures, like hallucinations, off-topic, failed citations, and abstentions. Inspired by the Prediction Powered Inference (PPI) method, we achieve precise performance measurements with statistical guarantees. We provide a comprehensive dataset for further analysis. Our contributions aim to enhance the reliability and sc

[18] score=0.0145 doc=docs_ingestor/docs/arxiv/2508.12682.pdf chunk=S3::C05::251104163000
  text: Introduction Recently, retrieval-augmented generation (RAG) has emerged as a cost-effective alternative to fine-tuning, offering higher accuracy and more flexible deployment for enterprise applications (Lewis et al. 2021; Fan et al. 2024). By enabling large language models (LLMs) to seamlessly access external knowledge bases during inference, RAG reduces hallucinations and improves factual consistency without retraining. Constructing such knowledge bases is straightforward: raw documents are split into chunks, embedded, and stored in vector databases-an approach that can be fully deployed locally without extensive data cleaning. As a result, RAGprovides a private, scalable, and low-overhead solution for adapting LLMs to industrial use cases (Balaguer et al. 2024).

[19] score=0.0143 doc=docs_ingestor/docs/arxiv/2507.10382.pdf chunk=S14::C01::251104135349
  text: VI. CONCLUSION AND FUTURE WORKS This paper presents a comprehensive shared e-mobility platform that integrates cloud-based simulation, real-time data processing, and RAG-enhanced decision-making to address the challenges in sustainable urban transportation. The platform supports dynamic multi-modal routing and flexible docking-based systems through a scalable and usercentric design. The schema-level RAG evaluation highlights XiYanSQL's high execution accuracy of 0.81 on system operator queries and 0.98 on user queries. An error analysis reveals that logic errors are the predominant failure type for most models, while granularity issues are more common in user-level queries. Although the LLM-based RAG framework performs robustly, the generated responses are not always perfect. Users and system operators should treat LLM outputs as supportive decision-making aids rather than definitive conclusions and should cross-check critical decisions against simulation data or dashboard outputs when high reliability is required.

[20] score=0.0141 doc=docs_ingestor/docs/arxiv/2508.05100.pdf chunk=S3::C06::251104155301
  text: Introduction Empirical analyses across multiple real-world benchmarks reveal that BEE-RAG fundamentally alters the entropy scaling laws governing conventional RAG systems, which provides a robust and scalable solution for RAG systems dealing with long-context scenarios. Our main contributions are summarized as follows: We introduce the concept of balanced context entropy, a novel attention reformulation that ensures entropy invariance across varying context lengths, and allocates attention to important segments. It addresses the critical challenge of context expansion in RAG.

[21] score=0.0139 doc=docs_ingestor/docs/arxiv/2507.10382.pdf chunk=S4::C01::251104135349
  text: A. Transportation Simulation Platform Classical traffic simulation platforms, such as AIMSUN [4] and VISSIM [5], primarily address issues like congestion and scheduling through microscopic modeling [12]. SUMO [6] further introduced a hybrid approach combining different mathematical models for algorithm evaluation. While traditional platforms focus on congestion and accident analysis, recent efforts have begun integrating e-mobility to assess impacts on energy, urban economies, and communities. Examples include digital twin-based platforms linking e-mobility and smart grids [13], sustainable EV adoption tools [14], and data-driven EV impact simulations [15]. Although studies have explored shared e-mobility impacts [16], [17], few have developed scalable simulation platforms with native support for shared e-mobility and eHub distribution, which is often oversimplified in smart city contexts [18]. [7] proposed a multi-agent platform using deep reinforcement learning for EV fleet management, but research on scalable, multi-modal

[22] score=0.0139 doc=docs_ingestor/docs/arxiv/2507.19562.pdf chunk=S12::C04::251104143937
  text: B. Retrieval-Augmented Generation (RAG) for Code Generalization PennyCoder combines the strengths of instruction fine-tuning and lightweight parameter-efficient adaptation techniques to deliver a robust, locally deployable LLM solution for PennyLane quantum programming. By addressing both data scarcity and compute constraints, PennyCoder provides a practical path forward for scalable, privacypreserving quantum code generation.

[23] score=0.0137 doc=docs_ingestor/docs/arxiv/2507.19666.pdf chunk=S4::C03::251104145914
  text: 2 Related Work However , these studies evaluate out-of-the-box LLM or VLM performance; they do not aim to optimize model accuracy and do not incorporate retrieval augmentation techniques. While GPT models have likely encountered many laws and question pairs during pretraining, reframing this task as a Retrieval-Augmented Generation (RAG) problem could offer a more principled and scalable approach to improving legal and regulation-specific reasoning.

[24] score=0.0135 doc=docs_ingestor/docs/arxiv/2509.20707.pdf chunk=S7::C06::251104185442
  text: 1 Introduction To address the need for accurate, interpretable and scalable plan evaluation, we introduce an automated RAG system powered by LLaMA-4 109B for radiotherapy quality assessment. Our framework integrates (1) a scoring module that computes normalized dose metrics and population-based percentiles, (2) a retrieval module that identifies similar historical plans based on numerical and textual features, and (3) a constraint-checking tool that flags clinical violations using protocol-defined thresholds. These components are leveraged through an LLM-driven reasoning process that issues explicit tool calls to retrieve context, perform checks, and synthesize a structured, protocol-aware summary of plan quality. Unlike prior end-to-end systems, our method separates data, logic, and generation, reducing hallucinations while supporting traceability and flexible protocol integration. The output includes a quantitative plan score and a list of failed constraints, enabling transparent decision support aligned with clinical prac

[25] score=0.0135 doc=docs_ingestor/docs/arxiv/2507.10382.pdf chunk=S2::C05::251104135349
  text: I. INTRODUCTION To address these challenges, there is a growing need for cloud-based SUMO solutions that leverage the scalability of cloud platforms. Recent advancements in cloud-based SUMO focus on enhancing large-scale simulations and realtime data processing, but most studies concentrate on vehicular network optimization rather than shared e-mobility. [8] offers a high-level cloud-based SUMO framework, but it may be limited to simple queries or purely dashboardbased designs, which are not user-friendly. Meanwhile, the emergence of Large Language Models (LLMs) has opened new opportunities for enhancing intelligent decision-making in traffic-related domains, including delivery scheduling, location prediction, and secure communications. Building on this potential, we propose an LLM-powered, cloudbased platform that integrates dynamic traffic simulation with natural language interaction, aiming to bridge the gap between traditional simulation tools and user-friendly, adaptive shared mobility solutions. In particular, our plat

[26] score=0.0133 doc=docs_ingestor/docs/arxiv/2509.13772.pdf chunk=S11::C02::251104182521
  text: 7. Discussion and Limitations Scalability of RAGOrigin: We extend our evaluation by scaling the NQ dataset's knowledge database to 16.7 million texts, combining entries from the knowledge database of NQ, HotpotQA, and MS-MARCO. Using the same user questions from NQ, we assess RAGOrigin's performance under larger data volumes. As shown in Table 16, RAGOrigin maintains consistent effectiveness and performance even on this significantly expanded database. These results demonstrate that RAGOrigin remains robust at scale, making it suitable for enterprise-level applications requiring large

[27] score=0.0130 doc=docs_ingestor/docs/arxiv/2508.04442.pdf chunk=S7::C06::251104155058
  text: 5 Discussion 5.3 Linguistic and Curricular Nuances in a Malaysian Context The study confirmed GPT-4o's proficiency in handling specific Bahasa Melayu mathematical terminology like integer, pecahan (fraction), perpuluhan (decimal), and tertib menurun (descending order) as used in the source documents.[1] The RAG methods were particularly effective at adopting the precise phrasing and style of the teacher-prepared notes. This, however, reveals a deeper nuance in the concept of 'curriculum alignment.' The automated metrics used in this study excel at verifying topical and factual alignment. For example, the STS score confirms a question is about integers, and the RAG-QA check confirms it is answerable based on the curriculum's scope. Yet, the RPT specifies not just topics but also cognitive skills. Learning standard 1.2.6, 'Menyelesaikan masalah yang melibatkan integer' (Solving problems involving integers), requires a higher cognitive level than standard 1.1.2, 'Mengenal dan memerihalkan integer' (Recognizing and describing in

[28] score=0.0128 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C01::251104142800
  text: 7 Challenges of RAG This section discusses the challenges of RAG, cases of manifestation of such challenges in the selected domain of RAG application, and outlines existing solutions and the way forward.

[29] score=0.0127 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C06::251104142800
  text: 7 Challenges of RAG 7.2.2 Freshness and Knowledge Updates One motivation for RAG is providing current information. However, continuously updating external knowledge bases and retrieval indices is challenging. Domains requiring real-time updates (e.g., finance, healthcare) demand sophisticated data pipelines for incremental updates, possibly frequent re-encoding of documents, and synchronization of retrieval indices. Delays in updates or inconsistencies between the LLM's internal knowledge and newly retrieved data can produce outdated or contradictory answers [61].

[30] score=0.0127 doc=docs_ingestor/docs/arxiv/2507.23242.pdf chunk=S12::C07::251104154436
  text: RAG System Implementation | | Query Rewriter | Document Recognizer | Document Retriever | Train Data | |---:|:------------------|:----------------------|:---------------------|:---------------| | 0 | RL-QR multi-modal | ColQwen2.5-v0.2 | ColQwen2.5-v0.2 | D mm D tm D tm | | 1 | RL-QR lexical | AI Parser | ixi-RAG lexical | | | 2 | RL-QR semantic | AI Parser | ixi-RAG semantic | | | 3 | RL-QR hybrid | AI Parser | ixi-RAG hybrid | D tm | Table 2: RL-QR Training Scheme rocal Rank Fusion (RRF) algorithm (Cormack, Clarke, and Buettcher 2009). RRF calculates a new score for each document based on its position in the individual rankings, producing a single, more robustly ordered list that balances keyword relevance with semantic similarity. This method effectively captures both the precision of lexical search and the contextual understanding of semantic search to deliver a highly refined final ranking.

[31] score=0.0125 doc=docs_ingestor/docs/arxiv/2508.10701.pdf chunk=S6::C01::251104162516
  text: C. New vantage point to prevent 1-day/n-day vulnerabilities To tackle the compatibility and scalability issue, security vendors are shifting the vulnerability fixing function from host-side to Edge Security Gateways (ESG) , including Amazon eero [2], Cisco Meraki [3], Netgear Orbi [6] and Linksys Velop [5]. In such network-fix paradigm, the vulnerability fixing is enforced as network filtering on the edge security gateways. The remote cloud services is responsible for generating the filtering rules and installing them on the edge security gateways. For example, the Cisco Talos Intelligence cloud service can generate a network filtering for Log4j, and deploy it on Meraki MX edge routers to detect and block Log4j exploits [4], [7]. The gateways hosting the vulnerability fixes are unified platforms such as Cisco IOS [13], Rasberry PI [24] or OpenWRT [22]. The network-based patches only need to adapt to several unified edge platforms instead of heterogeneous vulnerable devices. Unlike current host-based patching mechanisms - whi



## Rerank (0.74s)

**Tech:** Cohere API

**Results**

Rerank summary:
- strategy=cohere
- model=rerank-english-v3.0
- candidates=31
- eligible_above_threshold=31
- kept=6 (threshold=0)

Reranked Relevant (6/31 kept ≥ 0) top 6 for query: why is everyone saying RAG doesn’t scale? how are people fixing that?

[1] score=0.7920 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S4::C08::251104135247
  text: 1 Introduction Scalability: Naive implementations of Retrieval-Augmented Generation (RAG) often rely on 16-bit floating-point large language models (LLMs) for the generation component. However, this approach introduces significant scalability challenges due to the increased memory demands required to host the LLM as well as longer inference times due to using a higher precision number type. To enable more efficient scaling, it is crucial to integrate methods or techniques that reduce the memory footprint and inference times of generator models. Quantized models offer more scalable solutions due to less computational requirements, hence when developing RAG systems we should aim to use quantized LLMs for more cost effective deployment as compared to a full fine-tuned LLM whose performance might be good but is more expensive to deploy due to higher memory requirements. A quantized LLM's role in the RAG pipeline itself should be minimal and for means of rewriting retrieved information into a presentable fashion for the end users

[2] score=0.4749 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C42::251104142800
  text: 7 Challenges of RAG 7.5.5 Scalability Scalability challenges arise as knowledge corpora expand. Advanced indexing, distributed retrieval, and approximate nearest neighbor techniques facilitate efficient handling of large-scale knowledge bases [57]. Selective indexing and corpus curation, combined with infrastructure improvements like caching and parallel retrieval, allow RAG systems to scale to massive knowledge repositories. Research indicates that moderate-sized models augmented with large external corpora can outperform significantly larger standalone models, suggesting parameter efficiency advantages [10].

[3] score=0.4304 doc=docs_ingestor/docs/arxiv/2507.18910.pdf chunk=S22::C05::251104142800
  text: 7 Challenges of RAG 7.2.1 Scalability and Infrastructure Deploying RAG at scale requires substantial engineering to maintain large knowledge corpora and efficient retrieval indices. Systems must handle millions or billions of documents, demanding significant computational resources, efficient indexing, distributed computing infrastructure, and cost management strategies [21]. Efficient indexing methods, caching, and multi-tier retrieval approaches (such as cascaded retrieval) become essential at scale, especially in large deployments like web search engines.

[4] score=0.3556 doc=docs_ingestor/docs/arxiv/2509.13772.pdf chunk=S11::C02::251104182521
  text: 7. Discussion and Limitations Scalability of RAGOrigin: We extend our evaluation by scaling the NQ dataset's knowledge database to 16.7 million texts, combining entries from the knowledge database of NQ, HotpotQA, and MS-MARCO. Using the same user questions from NQ, we assess RAGOrigin's performance under larger data volumes. As shown in Table 16, RAGOrigin maintains consistent effectiveness and performance even on this significantly expanded database. These results demonstrate that RAGOrigin remains robust at scale, making it suitable for enterprise-level applications requiring large

[5] score=0.2235 doc=docs_ingestor/docs/arxiv/2507.07695.pdf chunk=S3::C01::251104135247
  text: ABSTRACT Fine-tuning is an immensely resource expensive process when trying to retrain Large Language Models (LLMs) to have access to a larger bank of knowledge. To alleviate this issue there have been many different fine-tuning techniques proposed which have shown good progress in trying to reduce time and computational resources to achieve fine-tuning but with LLMs becoming more intelligent and larger, this issue continues to arise. Hence a new method of enabling knowledge expansion on LLMs had to be devised. Retrieval-Augment-Generate (RAG) is a class of techniques where information is stored in a database and appropriate chunks of information are retrieved to help answer the question. However there are many limitations to naive RAG implementations. This paper proposes the KeyKnowledgeRAG ( K 2 RAG ) framework to address the scalability and answer accuracy limitations associated with naive RAG implementations. This framework takes inspiration from divide-and-conquer ideology, and combines dense and sparse vector search, k

[6] score=0.1277 doc=docs_ingestor/docs/arxiv/2508.05100.pdf chunk=S3::C06::251104155301
  text: Introduction Empirical analyses across multiple real-world benchmarks reveal that BEE-RAG fundamentally alters the entropy scaling laws governing conventional RAG systems, which provides a robust and scalable solution for RAG systems dealing with long-context scenarios. Our main contributions are summarized as follows: We introduce the concept of balanced context entropy, a novel attention reformulation that ensures entropy invariance across varying context lengths, and allocates attention to important segments. It addresses the critical challenge of context expansion in RAG.
  
## Context Expansion (1.18s)

**Tech:** Redis (to fetch neighbors and additional informtion)

**Results:**

### Document #1 — KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities
- `doc_id`: `doc::6d7a7d88cfc04636b20931fdf22f1e61`
- **Last Updated:** 2025-11-04T13:52:47.359001+00:00
- **Context:** Enhances RAG systems for efficient knowledge retrieval in LLMs.
- **Content fetched inside document:**
```text
[start on page 1]
    ---------------------- this was the passage that we matched to the query -------------
    ABSTRACT
    
    Fine-tuning is an immensely resource expensive process when trying to retrain Large Language Models (LLMs) to have access to a larger bank of knowledge. To alleviate this issue there have been many different fine-tuning techniques proposed which have shown good progress in trying to reduce time and computational resources to achieve fine-tuning but with LLMs becoming more intelligent and larger, this issue continues to arise. Hence a new method of enabling knowledge expansion on LLMs had to be devised. Retrieval-Augment-Generate (RAG) is a class of techniques where information is stored in a database and appropriate chunks of information are retrieved to help answer the question. However there are many limitations to naive RAG implementations. This paper proposes the KeyKnowledgeRAG ( K 2 RAG ) framework to address the scalability and answer accuracy limitations associated with naive RAG implementations. This framework takes inspiration from divide-and-conquer ideology, and combines dense and sparse vector search, knowledge graphs and text summarization to help address these limitations. The framework also involves a data preprocessing step to reduce training times. The MultiHopRAG dataset was used for evaluation where the our implemented K 2 RAG pipeline was trained with the document corpus and then evaluated with the test data. Results have shown that there is an improvement in answering questions compared to common Naive RAG implementations where K 2 RAG achieved the highest mean answer similarity at 0.57 and also was able to answer more questions with more ground truth similarity with a highest Q 3 quartile at 0.82 . Additionally our proposed framework is trainable much faster due to the inclusion of a train data corpus summarization step which reduces training times of the individual components by 93% on average. Furthermore K 2 RAG has shown to operate even faster than a traditional knowledge graph based naive RAG implementation with a mean execution times reduced by up to 40% . In addition to superior question-answering capabilities, K 2 RAG offers being a more scalable solution with VRAM requirements reduced by 3 times in comparison to implementations of several naive RAG pipelines evaluated in this paper. Hence K 2 RAG can help companies in sophisticated decision-making through implementing a more lightweight and robust question-answering systems built on internal documents.
    --------------------------------------------------------------------------------------

    1 Introduction
    
    different setbacks associated with the LLM fine-tuning process such as improving the final efficacy as well as reducing computational resources.
    Fine-tuning LLMs is usually done on full LLMs and is generally not advised to fine-tune on or convert pre-finetuned models into quantized models due to potential loss of knowledge retention and accuracy which can be attributed to lower precision of the model's weights. Even though existing techniques like QLoRA have shown results which indicate minimal performance degradation upon the resulting fine-tuned quantized models which are based on their Guanaco models, this is not applicable to all LLMs [3]. For example an empirical study [4] evaluated different LLAMA3 quantizations and fine-tune quantizations has shown a significant reduction in LLAMA3-8B scores on the MMLU benchmarks when compared to the standard full 16-bit model and a QLoRA fine-tune-quantization to 4bits. This is because although all LLMs follow the basic transformer architecture there are still differences in model designs which can impact the quality of fine-tunes and hence even previously considered efficient methods such as QLoRA face their limitations with newer models [4].

    Since 2020 the concept of RAG has since evolved to include many different implementations and approaches to performing information retrieval to enhance LLM answering capabilities on new datasets [Figure 1]. This highlights how the benefits gained from RAG are being researched in depth to further optimize this process.

    Answer Accuracy: Naive RAG implementations often struggle with the "Needle-in-a-Haystack" problem [8], where the generator model fails to provide accurate answers due to an inability to extract relevant information from irrelevant or excessively long contexts. This issue typically arises from poorly optimized chunk sizes and retrieval processes [9]. Optimizing chunk size is critical and challenging, as smaller chunks risk losing essential context, while larger chunks may introduce irrelevant information and unnecessarily increase context length [Figure 2]. Optimizing the retrieval process to maximize the relevance of retrieved information blocks is crucial. This ensures that irrelevant data, which could act as noise and obscure the correct information, does not compromise the generator LLM's ability to accurately

    Figure 2: Graph showing how LLMs prefer extracting information from top or bottom of context to answer a question [7].
 
    answer the question. Another limitation of naive RAG systems is that embeddings which are similar but not actually relevant to answer the question might be retrieved [10] as the original sentences might be semantically similar or carry a similar meaning but might not be right in helping answer the question in that context.

    ---------------------- this was the passage that we matched to the query -------------
    Scalability: Naive implementations of Retrieval-Augmented Generation (RAG) often rely on 16-bit floating-point large language models (LLMs) for the generation component. However, this approach introduces significant scalability challenges due to the increased memory demands required to host the LLM as well as longer inference times due to using a higher precision number type. To enable more efficient scaling, it is crucial to integrate methods or techniques that reduce the memory footprint and inference times of generator models. Quantized models offer more scalable solutions due to less computational requirements, hence when developing RAG systems we should aim to use quantized LLMs for more cost effective deployment as compared to a full fine-tuned LLM whose performance might be good but is more expensive to deploy due to higher memory requirements. A quantized LLM's role in the RAG pipeline itself should be minimal and for means of rewriting retrieved information into a presentable fashion for the end users to interpret.
    --------------------------------------------------------------------------------------

    Additionally, in naive RAG implementations, information store components, such as Knowledge Graphs, face lengthy training times for their creation and updating. This creates scalability challenges, as it increases the downtime required to update the information store with new knowledge in a production environment. Hence an effective data pre-processing step should be introduced to reduce the training corpus size and hence training times while preserving as much information as possible.

    Solution: We propose the K 2 RAG framework which addresses the following 4 research goals based on the aforementioned answer accuracy and scalability issues characteristic of naive RAG implementations:
    
    Goal 1. Reduce information store such as Knowledge Graph and Spare and Dense vector database creation times.
    Goal 2. Reduce chances of LLMs suffering from "Needle in Haystack" Problem.
    Goal 3. Increase rate of retrieving relevant passages for answering the question.
    Goal 4. Alleviate time and computational cost associated with full LLMs.
    
    K 2 RAG incorporates various different concepts covered in the literature in a novel way to improve the performance and offer a more resource efficient and accurate RAG pipeline compared to naive RAG methods outlined in the literature.
[end on page 3]
```

### Document #2 — A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions
- `doc_id`: `doc::688cfbc0abdc4520a73e219ac26aff41`
- **Last Updated:** 2025-11-04T14:28:00.715095+00:00
- **Context:** A review of RAG's evolution and its evolving role in knowledge-intensive NLP.
- **Content fetched inside document:**
```text
[start on page 20]
    7 Challenges of RAG
    This section discusses the challenges of RAG, cases of manifestation of such challenges in the selected domain of RAG application, and outlines existing solutions and the way forward.

    7.1.1 Retrieval Quality and Relevance
    The quality of retrieved documents significantly impacts the accuracy of RAG-generated answers. High recall and precision are critical since poor retrieval leads directly to incorrect or irrelevant answers [61]. Traditional methods like BM25 are limited, often missing relevant texts or returning noisy results [12]. Modern neural retrievers with dense embeddings improve performance but still face issues like vocabulary mismatches, ambiguous queries, and domain-specific terminology. Specialized domain tuning, such as using legal embeddings or medical synonym expansion, can help, but maintenance of these tailored retrievers remains challenging. Determining the optimal number of retrieved passages ( k ) is also complex. Too few passages limit evidence; too many overwhelm the model and introduce irrelevant context. Approaches like ranking retrieved passages or iterative query reformulation can improve retrieval precision, but add complexity and latency [34, 21].

    7.1.2 Latency and Efficiency
    RAG inherently increases computational complexity and latency compared to standalone LLMs due to retrieval overhead, vector searches, and expanded context processing. Techniques like approximate nearest neighbor indices (e.g., FAISS, HNSW), caching, model distillation, or lightweight retrievers can reduce latency at the expense of accuracy. Integrating retrieval efficiently with large language models (LLMs) and ensuring rapid responses in real-time scenarios (e.g., customer support) remains a significant challenge [3]. Interestingly, using retrieval can allow smaller models to match the performance of larger models without retrieval (e.g., RETRO, Atlas), reducing model size requirements but shifting complexity to maintaining external knowledge bases and infrastructure.

    7.1.3 Integration with Large Language Models
    Integrating retrieved evidence effectively with LLMs is subtle. Models may ignore retrieved evidence, especially when internal model knowledge conflicts with external retrieved information, leading to a "tug-of-war" effect [41]. Multiple retrieved documents might create confusion or confirmation bias if they contradict each other. Limited input lengths in transformer-based LLMs exacerbate these integration challenges by forcing truncation or summarization, potentially omitting essential context. Fine-tuning models specifically for retrieval-augmented tasks often yields better integration than simple zero-shot prompting but introduces complexity, especially when using non-differentiable or API-based models that do not support custom training.

    ---------------------- this was the passage that we matched to the query -------------
    7.2.1 Scalability and Infrastructure
    Deploying RAG at scale requires substantial engineering to maintain large knowledge corpora and efficient retrieval indices. Systems must handle millions or billions of documents, demanding significant computational resources, efficient indexing, distributed computing infrastructure, and cost management strategies [21]. Efficient indexing methods, caching, and multi-tier retrieval approaches (such as cascaded retrieval) become essential at scale, especially in large deployments like web search engines.
    --------------------------------------------------------------------------------------

    7.2.2 Freshness and Knowledge Updates
    One motivation for RAG is providing current information. However, continuously updating external knowledge bases and retrieval indices is challenging. Domains requiring real-time updates (e.g., finance, healthcare) demand sophisticated data pipelines for incremental updates, possibly frequent re-encoding of documents, and synchronization of retrieval indices. Delays in updates or inconsistencies between the LLM's internal knowledge and newly retrieved data can produce outdated or contradictory answers [61].

    7.2.3 Hallucination and Reliability
    While RAG reduces LLM hallucinations, it does not eliminate them completely. Models may fabricate or misattribute information if retrieval provides incomplete or partially contradictory context.
    Legal domain studies found that RAG significantly reduces hallucinations, but still generates errors at concerning rates [56]. Hallucinations also occur in citation generation, with models occasionally inventing nonexistent references. Strategies such as verifying outputs against retrieved sources or calibrating model confidence are needed, but no approach completely prevents hallucination.
    HIPAA), transparent data usage policies, and security testing are essential to protect privacy and prevent breaches.

    7.3.4 Accountability and Transparency
    RAG's use of sourced retrieval provides an

    7.2.4 Complex Pipeline and Maintenance
    advantage for accountability, allowing users to trace AI-generated responses back to evidence. However, inaccurate citations or improper synthesis can mislead users. Ethical deployment involves clearly attributing evidence, providing explanations on request, managing user expectations, and clearly delineating accountability-especially when RAG informs critical decisions. Transparency and accountability require ongoing evaluation, oversight, and mechanisms for user feedback and correction [56].

    7.4.3 Customer Support and Knowledge Bases
    Metrics for Evaluation : Success in customer support is evaluated through resolution rates and customer satisfaction metrics, requiring responses that not only provide factual accuracy but practical utility.
    Ethical considerations include transparency, data security, job impact, and fairness. RAG
    implementations also provide valuable feedback for continual improvement of internal support documentation based on real-time user interactions, thereby enhancing both system performance and documentation quality.

    7.5.1 Retrieval Quality
    : Maintaining high retrieval relevance is critical for effective RAG. Strategies to improve retrieval quality include domain-adaptive training, advanced encoders, and query reformulation methods to address vocabulary mismatches [85]. Employing reranking models further boosts relevance by re-scoring initial retrieval results with deeper contextual analysis, enhancing accuracy at the expense of additional computation [4]. Iterative retrieval and chain-of-thought reasoning represent future directions, breaking down complex queries into simpler sub-queries, thus ensuring relevant information retrieval at each reasoning step [90].

    7.5.2 Latency
    RAG systems introduce latency due to retrieval processes. Solutions include using efficient nearest-neighbor search structures, such as HNSW graphs, which significantly speed up similarity searches [57]. Caching mechanisms, including multi-level and approximate embedding caches (e.g., RAGCache and Proximity cache), enable reuse of previously retrieved information, drastically reducing retrieval time [40, 8]. Adaptive retrieval methods dynamically balance retrieval complexity based on query difficulty, optimizing overall throughput and reducing latency.

    7.5.3 Model Integration
    Effective integration between retrieval and generation models remains essential. Methods include joint end-to-end training of retrievers and generators, enhancing mutual compatibility and performance [52]. Architectural integration techniques, such as RETRO's cross-attention mechanism, dynamically incorporate retrieved facts during generation [10]. Alternatively, prompt-based integration treats LLMs as black-boxes, conditioning on retrieved documents without architectural modifications. Future hybrid approaches involving reinforcement learning and selective retrieval aim to optimize when and how external knowledge is incorporated into generation processes.

    7.5.4 Hallucination
    Reducing factual hallucinations remains a key focus. RAG inherently mitigates hallucinations by grounding outputs in retrieved evidence [82]. Training models to penalize ungrounded assertions and iterative retrieval within reasoning processes further enhance accuracy [90]. Self-check mechanisms (Self-RAG), where models critique and revise their outputs against retrieval results, significantly reduce hallucinated content [6]. External verification and fact-checking modules complement internal methods, collectively ensuring high factual reliability. For instance, RAG systems to cite sources significantly enhance their reliability by directly linking generated information to supporting evidence. This citation capability plays a crucial role in mitigating the common issue of hallucination, where generative models produce plausible yet inaccurate or fabricated information. By explicitly associating each factual statement with retrieved documents, RAG systems encourage transparency and verifiability, enabling users and downstream processes to quickly assess the accuracy and provenance of claims. Moreover, requiring the model to cite sources during generation inherently promotes grounding outputs in verified data, further reducing the risk of generating unsupported statements [82]. Thus, citation functionality not only enhances user trust but also fosters more disciplined, factually accurate generation, substantially decreasing the likelihood of hallucinated outputs.

    ---------------------- this was the passage that we matched to the query -------------
    7.5.5 Scalability
    Scalability challenges arise as knowledge corpora expand. Advanced indexing, distributed retrieval, and approximate nearest neighbor techniques facilitate efficient handling of large-scale knowledge bases [57]. Selective indexing and corpus curation, combined with infrastructure improvements like caching and parallel retrieval, allow RAG systems to scale to massive knowledge repositories. Research indicates that moderate-sized models augmented with large external corpora can outperform significantly larger standalone models, suggesting parameter efficiency advantages [10].
    --------------------------------------------------------------------------------------

    7.5.6 Knowledge Freshness
    Rapidly evolving information necessitates regularly updated knowledge bases. RAG systems can efficiently maintain knowledge freshness through incremental updates and selective retrieval methods without requiring frequent retraining [30]. Integrating live search APIs and hybrid retrieval methods ensure real-time information retrieval, addressing dynamic knowledge demands [21]. Continuous updates and user-feedback integration support lifelong learning and timely information access.

    7.5.7 Bias
    Addressing bias in RAG involves curating balanced knowledge sources, employing diversification techniques in retrieval, and adjusting retriever embeddings to counteract inherent biases [46]. Prompts and model training that encourage balanced representation, along with transparency in source attribution, further mitigate bias propagation. This multi-faceted approach helps minimize biases in RAG outputs.

    7.5.8 Misinformation
    Combating misinformation involves preventive measures like curating trustworthy knowledge sources and reactive verification through stance classifiers and credibility assessments [66]. Models employing vigilant prompting, cross-verification with multiple retrieved documents, and external fact-checking modules enhance reliability and truthfulness. Robustness against adversarial misinformation insertion through continuous monitoring and data validation further strengthens RAG systems, ensuring accurate information dissemination.
[end on page 24]
```

### Document #3 — Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation
- `doc_id`: `doc::11e21d1e2c53420baf3c0c12575ab566`
- **Last Updated:** 2025-11-04T18:25:21.220682+00:00
- **Context:** RAGOrigin enhances RAG systems' security against text poisoning.
- **Content fetched inside document:**
```text
[start on page 12]
    7. Discussion and Limitations
    Performance of more attribution methods: In this part, we examine two additional attribution methods, ContextCite [62] and AttriBoT [63]. Table 15 reports their performance under different attacks on NQ dataset. We observe that ContextCite attains a DACC below 0.77, while both FPR and FNR exceed 0.23 under all attack settings. AttriBoT performs even worse, with DACC falling below 0.70 and FPR and FNR above 0.29. These poor results are largely due to their reliance on a one-dimensional metric for measuring a text's contribution.
    ---------------------- this was the passage that we matched to the query -------------
    Scalability of RAGOrigin: We extend our evaluation by scaling the NQ dataset's knowledge database to 16.7 million texts, combining entries from the knowledge database of NQ, HotpotQA, and MS-MARCO. Using the same user questions from NQ, we assess RAGOrigin's performance under larger data volumes. As shown in Table 16, RAGOrigin maintains consistent effectiveness and performance even on this significantly expanded database. These results demonstrate that RAGOrigin remains robust at scale, making it suitable for enterprise-level applications requiring large knowledge databases. Note that in this section, we present only the results for RAGOrigin due to space limitations.
    --------------------------------------------------------------------------------------

    | | Metric | PRAGB | PRAGW | ProInject | HijackRAG | LIAR | Jamming | BadRAG | Phantom | AgentPoison |
    |---:|:---------|--------:|--------:|------------:|------------:|-------:|----------:|---------:|----------:|--------------:|
    | 0 | DACC | 0.99 | 0.99 | 1 | 1 | 0.98 | 1 | 1 | 1 | 1 |
    | 1 | FPR | 0 | 0 | 0.01 | 0 | 0.01 | 0 | 0 | 0 | 0 |
    | 2 | FNR | 0.02 | 0.03 | 0 | 0 | 0.03 | 0 | 0 | 0 | 0 |
    
    TABLE 18: Results of RAGOrigin with dynamic number of poisoned texts per target question on NQ dataset.

    | | Metric | PRAGB+PRAGW+ProInject+HijackRAG+LIAR | Jamming+BadRAG+Phantom+AgentPoison |
    |---:|:---------|---------------------------------------:|-------------------------------------:|
    | 0 | DACC | 0.99 | 1 |
    | 1 | FPR | 0.01 | 0 |
    | 2 | FNR | 0 | 0 |
    
    TABLE 19: Results of RAGOrigin on NQ dataset under the scenario that a target question is poisoned by multiple attack methods simultaneously.

    Effectiveness of RAGOrigin with advanced RAG frameworks: We evaluate RAGOrigin across multiple sophisticated RAG frameworks, including AAR [64], SuRe [65], Adaptive-RAG [66], SELF-RAG-arch [22], and IRCoT [67]. As demonstrated in Table 17, RAGOrigin consistently maintains high DACC and low FPR/FNR. These findings confirm that RAGOrigin remains robust even when integrated with advanced RAG frameworks that employ complex retrieval strategies and reasoning mechanisms.

    Effectiveness of RAGOrigin with dynamic number of poisoned texts per target question: By default, we assume the number of poisoned texts is the same across different target questions. We examine a scenario where the attacker injects a varying number of poisoned texts across target questions. In our setup, this number ranges from 5 to 50 per question, sampled from a truncated Gaussian distribution. As shown in Table 18, RAGOrigin maintains consistent effectiveness regardless of the poisoning scale. These results confirm RAGOrigin's reliability in production environments, where the extent of poisoning may be unpredictable or deliberately amplified.

    Effectiveness of RAGOrigin with multi-attacker scenarios: We evaluate RAGOrigin across three challenging multi-attacker scenarios. First, we consider the case where a single target question is simultaneously poisoned by multiple attack methods. These methods may target specific answers (PRAGB, PRAGW, ProInject, HijackRAG, LIAR) or denial-of-service answers (Jamming, BadRAG, Phantom, AgentPoison). As shown in Table 19, 'PRAGB+PRAGW+ProInject+HijackRAG+LIAR' indicates that, for each target question, multiple independent attackers inject different poisoned texts generated by their respective methods into the knowledge database. RAGOrigin maintains high DACC and low FPR/FNR in both targeted and denial-of-service cases. Second, we evaluate the scenario where multiple attackers simultaneously poison different target questions. Although each question has a different target answer, the misgeneration event reported by users is associated with only one of them. As demonstrated in Table 20, RAGOrigin can still accurately attribute the poisoned texts responsible for the reported misgeneration event. Third, we study the case where multiple independent attackers target the same question but with different target answers. We exclude Jamming, BadRAG, Phantom, and

    | | Metric | PRAGB | PRAGW | ProInject | HijackRAG | LIAR | Jamming | BadRAG | Phantom | AgentPoison |
    |---:|:---------|--------:|--------:|------------:|------------:|-------:|----------:|---------:|----------:|--------------:|
    | 0 | DACC | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 1 | 1 | 1 | 1 |
    | 1 | FPR | 0.01 | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0 | 0 | 0 |
    | 2 | FNR | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
    
    TABLE 20: Results of RAGOrigin on NQ dataset under the scenario that multiple attackers inject poisoned texts for different target questions simultaneously.

    | | Metric | PRAGB | PRAGW | ProInject | HijackRAG | LIAR |
    |---:|:---------|--------:|--------:|------------:|------------:|-------:|
    | 0 | DACC | 1 | 0.99 | 0.99 | 1 | 0.98 |
    | 1 | FPR | 0 | 0.01 | 0.01 | 0 | 0.03 |
    | 2 | FNR | 0 | 0 | 0 | 0 | 0 |
    
    TABLE 21: Results of RAGOrigin on NQ dataset under the scenario that multiple attackers target the same question but with different target answers.

    AgentPoison here, as they all share the same denial-ofservice target. Again, only one attacker's misgeneration is observed by the user. Table 21 confirms the robustness of RAGOrigin against such attack strategy.
[end on page 13]
```

### Document #4 — BEE-RAG: Balanced Entropy Engineering for Retrieval-Augmented Generation
- `doc_id`: `doc::b8d85d9737e544a98d7494a7811714ba`
- **Last Updated:** 2025-11-04T15:53:01.146617+00:00
- **Context:** The document presents BEE-RAG, a framework enhancing RAG performance in LLMs by optimizing context management.
- **Content fetched inside document:**
```text
[start on page 1]
    Introduction
    Retrieval-augmented generation (RAG) has emerged as a transformative paradigm for augmenting large language models (LLMs) with dynamically integrated external knowledge (Gao et al. 2023; Lewis et al. 2020). While conventional RAG systems demonstrate remarkable capabilities in knowledge-intensive tasks, their performance exhibits critical fragility when processing extended contextual inputs (Ren et al. 2023). This vulnerability arises from the inherent requirement of RAG frameworks to incorporate external retrieved documents, which typically results in substantially longer input sequences compared to standard generation tasks (Zhang et al. 2024a). This limitation manifests acutely in scenarios demanding long-context comprehension, such as scientific literature analysis, multi-hop reasoning, and cross-document synthesis, where the growing complexity of modern knowledge systems necessitates robust processing of extended text spans (Liu et al. 2023).
    
    Figure 1: In vanilla RAG, increasing input context length raises the information entropy of attention scores (left), potentially harming performance, while the LLM is less focused on important segments (right). This paper proposes a balanced entropy engineering strategy, which maintains entropy stability for longer contexts and guide the LLM to focus on critical segments.

    Typically, existing efforts to mitigate this challenge focus predominantly on heuristic document filtering, extrapolative training, or architectural modifications, introducing worth noting trade-offs. Threshold-based retrieval document truncation or selection reduces context length at the cost of discarding potentially relevant information (Jeong et al. 2024; Wang et al. 2024). While extended context training may enhance RAG performance (Lin et al. 2024; Zhang et al. 2024a), it substantially increases computational demands and risks compromising model generalization. The introduction of auxiliary modules, such as specialized encoders (Yen, Gao, and Chen 2024), further escalates the complexity of the system.

    As shown in Figure 1, we identify a fundamental oversight in existing explorations: the failure to address the core tension between context entropy growth and attention allocation dynamics. As retrieved document length increases, unconstrained context entropy expansion progressively dilutes attention distributions, undermining LLMs' capacity to prioritize semantically critical content. Such a phenomenon substantially degrades RAG's ability to extract and utilize salient information.

    In this work, we fundamentally rethink RAG's context length adaptability through the perspective of entropy engineering, and propose Balanced Entropy-Engineered RAG ( BEE-RAG ), a principled framework that establishes entropy-invariant dynamics to decouple attention sharpness from context length. First, we introduce balanced context entropy, a novel attention reformulation incorporating document-specific balancing entropy factor β to enforce entropy invariance across variable context lengths. Theoretical analysis demonstrates this mechanism maintains entropy levels within a stable regime for various context scales, contrasting with conventional entropy scaling. Building on this foundation, we develop intrinsic multi-importance inference, a zero-shot strategy that derives balancing entropy factors through context aggregation while preserving inter-document independence via parallel context scoring and generation. To ensure broad applicability, we further propose adaptive balancing factor learning, which is a parameter-efficient tuning method that learns task-specific balancing factor through lightweight linear projections for domain adaptation with minimal parameter updates (0.014% of total parameters).

    ---------------------- this was the passage that we matched to the query -------------
    Empirical analyses across multiple real-world benchmarks reveal that BEE-RAG fundamentally alters the entropy scaling laws governing conventional RAG systems, which provides a robust and scalable solution for RAG systems dealing with long-context scenarios. Our main contributions are summarized as follows:
    
    We introduce the concept of balanced context entropy, a novel attention reformulation that ensures entropy invariance across varying context lengths, and allocates attention to important segments. It addresses the critical challenge of context expansion in RAG.
    --------------------------------------------------------------------------------------
    We propose a zero-shot strategy called intrinsic multiimportance inference and parametric-efficient finetuning strategy adaptive balancing entropy factor learning to obtain the balancing entropy factor in different scenarios while maintaining efficiency.
    Extensive empirical validation across multiple benchmarks demonstrates the effectiveness and efficiency of BEE-RAG.
[end on page 2]
```


## LLM Answers (17.83s)
**Tech:** OpenAI's GPT-5 (reasoning low)

**Result:**

People say RAG “doesn’t scale” because several bottlenecks show up as data and users grow: (1) maintaining huge knowledge stores and fast indices for millions/billions of docs demands heavy infrastructure, distributed systems, and tight cost control [2]; (2) retrieval and long-context processing add latency compared to a standalone LLM, so responses slow down as k and corpora grow [2]; (3) LLMs can ignore or be confused by long, noisy contexts, and limited input windows force truncation, hurting accuracy at scale [2]; (4) naive setups often run full‑precision generator models, which spike memory and inference costs [1]; (5) updating corpora frequently (freshness) requires constant re-encoding and index sync, which is operationally hard [2]; (6) multi-part pipelines (retrievers, rankers, generators, graphs) are complex to maintain at production scale [2].

How people are fixing it:
- Infrastructure and retrieval: advanced/approximate nearest neighbor indexing (e.g., HNSW/FAISS), caching, cascaded/multi-tier and distributed retrieval, selective indexing, corpus curation, and parallel retrieval to keep throughput high with large corpora [2].
- Generation cost: use quantized LLMs so the model’s role is lightweight, cutting VRAM and inference time; K2RAG reports 3× lower VRAM and up to 40% faster execution vs naive pipelines [1].
- Data and pipeline design: summarize/condense training corpora to shrink build/update times (e.g., 93% reduction reported), and combine dense+spark/sparse search with knowledge graphs to retrieve more relevant passages with less noise [1].
- Long-context robustness: methods like BEE‑RAG re-balance attention so performance stays stable as context grows, improving scalability for long documents and multi-hop tasks [4].
- Freshness: incremental updates and hybrid/live retrieval to keep answers current without frequent full retraining [2].
- Security at scale: attribution/defense tools (e.g., RAGOrigin) that remain effective on very large knowledge bases (tested at 16.7M texts) help keep large deployments reliable [3].


**Reasoning:** Synthesized from documents: root causes (infrastructure, latency, integration, model cost, freshness, pipeline complexity) mainly from [2]; naive LLM precision and K2RAG improvements from [1]; long-context scaling via BEE-RAG from [4]; large-scale security robustness via RAGOrigin from [3].

**Citations:**

[1] doc::6d7a7d88cfc04636b20931fdf22f1e61 — KeyKnowledgeRAG (K^2RAG): An Enhanced RAG method for improved LLM question-answering capabilities

[2] doc::688cfbc0abdc4520a73e219ac26aff41 — A Systematic Review of Key Retrieval-Augmented Generation (RAG) Systems: Progress, Gaps, and Future Directions

[3] doc::11e21d1e2c53420baf3c0c12575ab566 — Who Taught the Lie? Responsibility Attribution for Poisoned Knowledge in Retrieval-Augmented Generation

[4] doc::b8d85d9737e544a98d7494a7811714ba — BEE-RAG: Balanced Entropy Engineering for Retrieval-Augmented Generation

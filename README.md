인공지능과 관련한 다양한 논문들을 리뷰하는 블로그입니다.

## 0. AI 기초

#### 기호논리 강의자료
- [기호논리 기초](./0.%20Basics%20of%20AI/기호논리%20기초.md)
- [Fundamentals of Symbolic Logic](./0.%20Basics%20of%20AI/Fundamentals%20of%20Symbolic%20Logic.md)
#### AI 기초 자료
- [Laplace approximation beginner guide](./0.%20Basics%20of%20AI/Laplace%20approximation%20beginner%20guide.md)
	- 테일러 전개에서 Fisher 정보까지: 라플라스 근사를 이해하는 가장 쉬운 길
- [A Generalized Representer Theorem](./0.%20Basics%20of%20AI/A%20Generalized%20Representer%20Theorem.md)
	- 무한차원 함수 최적화를 유한차원 문제로 바꾸는 Representer Theorem
## 1. LLM 기초 

#### 논문 리뷰
- [Why think step by step - Reasoning emerges from the locality of experience](./1.%20LLM%20Fundamentals/Why%20think%20step%20by%20step%20-%20Reasoning%20emerges%20from%20the%20locality%20of%20experience.md)
	- 왜 AI는 단계적으로 생각할 때 더 잘 풀까?
	- 2023년, 인용수 196
- [LightRAG- Simple and Fast Retrieval-Augmented Generation](./1.%20LLM%20Fundamentals/LightRAG-%20Simple%20and%20Fast%20Retrieval-Augmented%20Generation.md)
	- LightRAG는 왜 "커뮤니티"를 만들지 않고도 GraphRAG를 이기는가
	- 2025년, 인용수 793
- [Measuring AI Ability to Complete Long Software Tasks](./1.%20LLM%20Fundamentals/Measuring%20AI%20Ability%20to%20Complete%20Long%20Software%20Tasks.md)
	- AI는 "얼마나 긴 일"을 해낼 수 있을까 — 시간 지평(Time Horizon)으로 AI 능력을 재다
	- 2026년, 인용수 14

## 2. Neuro-symbolic AI 관련 연구

- [1. Neurosymbolic AI 연구 정리](./2.%20Neurosymbolic%20AI/1.%20Neurosymbolic%20AI%20연구%20정리.md)
#### 0) Survey 논문
- [The third AI summer- AAAI Robert S. Engelmore Memorial Lecture](./2.%20Neurosymbolic%20AI/The%20third%20AI%20summer-%20AAAI%20Robert%20S.%20Engelmore%20Memorial%20Lecture.md)
	- AI의 계절 순환사와 뉴로심볼릭 결합의 여섯 가지 설계
	- 뉴로심볼릭 연구의 분류체계를 제시 -> 표준적 분류체계로 사용 중임
	- 2022년, 인용수 470
- Towards data-and knowledge-driven AI- a survey on neuro-symbolic computing 

#### 1) 초기 연구: 지식을 신경망으로 표현 및 주입
- [Knowledge-Based Artificial Neural Networks](./2.%20Neurosymbolic%20AI/Knowledge-Based%20Artificial%20Neural%20Networks.md)
	- 논리 규칙을 신경망의 구조와 초기 가중치로 컴파일하기
	- 1994년, 인용수 1303
- [The Connectionist Inductive Learning and Logic Programming System](./2.%20Neurosymbolic%20AI/The%20Connectionist%20Inductive%20Learning%20and%20Logic%20Programming%20System.md)
	- 논리 프로그램을 단일 은닉층 신경망으로 정확히 옮기고 예제로 정제하기
	- 1999년, 인용수 280
#### 2) 술어, 변수, 논리식 등을 벡터 공간으로 매핑 - 지식표현
- [Semantic Compositionality through Recursive Matrix-Vector Spaces](./2.%20Neurosymbolic%20AI/Semantic%20Compositionality%20through%20Recursive%20Matrix-Vector%20Spaces.md)
	- 단어를 벡터이자 함수로 보기 
	- 2012년, 인용수 1896
- [Lifted Rule Injection for Relation Embeddings](./2.%20Neurosymbolic%20AI/Lifted%20Rule%20Injection%20for%20Relation%20Embeddings.md)
	- 논리 규칙을 벡터 공간의 부등식으로 바꾸기
	- 2016년, 인용수 201
- [Learning Continuous Semantic Representations of Symbolic Expressions](./2.%20Neurosymbolic%20AI/Learning%20Continuous%20Semantic%20Representations%20of%20Symbolic%20Expressions.md)
	- 기호 수식에 좌표를 찍는 법 — Learning Continuous Semantic Representations of Symbolic Expressions
	- 2017년, 인용수 135

#### 3) 논리 추론이 가능한 일반화된 신경망 프레임워크 제안
- [Logic Tensor Networks](./2.%20Neurosymbolic%20AI/Logic%20Tensor%20Networks.md)
	- 1차 논리를 미분 가능한 계산 그래프로 바꿔서 신경망으로 구현
	- **Real Logic**: 도메인, 상수, 변수를 텐서로, 술어, 함수를 텐서 간의 연산을 수행하는 신경망 조각으로 정의
	- 추론이 아닌 논리식 평가
	- 2016년 첫 게시 후 2022년 수정본 게시, 인용수: 633
- [Semantic-based regularization for learning and inference](./2.%20Neurosymbolic%20AI/Semantic-based%20regularization%20for%20learning%20and%20inference.md)
	- 데이터로부터의 학습과 논리적 추론을 하나의 최적화 문제로 통합
	- 2017년, 인용수 289
- [DeepProbLog : Neural Probabilistic Logic Programming](./2.%20Neurosymbolic%20AI/DeepProbLog%20-%20Neural%20Probabilistic%20Logic%20Programming.md)
	- LTN이 신경망으로 논리식을 구현하려 한 반면, 
	- 2018년, 인용수: 1189
- [Neural Logic Machines](./2.%20Neurosymbolic%20AI/Neural%20Logic%20Machines.md)
	- Neural Logic Machines: 논리 규칙을 학습하는 신경망
	- 2019년, 인용수 450
- [Logical Neural Networks](./2.%20Neurosymbolic%20AI/Logical%20Neural%20Networks.md)
	- 논리식의 구문 트리를 그대로 신경망으로 만들기
	- 2020년, 인용수 357
#### 4) 프로그램, 기호식 변환규칙, 연산 규칙, 물리법칙 등 학습
- [Making Neural Programming Architectures Generalize via Recursion](./2.%20Neurosymbolic%20AI/Making%20Neural%20Programming%20Architectures%20Generalize%20via%20Recursion.md)
	- 재귀(Recursion)로 신경망 프로그램을 완벽하게 일반화시키기
	- 2017년, 인용수 167
- [Bridging Machine Learning and Logical Reasoning by Abductive Learning](./2.%20Neurosymbolic%20AI/Bridging%20Machine%20Learning%20and%20Logical%20Reasoning%20by%20Abductive%20Learning.md)
	- 지각과 추론을 다리 놓다: Abductive Learning (ABL) 파헤치기
	- 2019년, 인용수 272
- [Deep Learning for Symbolic Mathematics](./2.%20Neurosymbolic%20AI/Deep%20Learning%20for%20Symbolic%20Mathematics.md)
	- 신경망으로 적분을 풀다 — Deep Learning for Symbolic Mathematics 읽기
	- 2020년, 인용수 737
- [Discovering Symbolic Models from Deep Learning with Inductive Biases](./2.%20Neurosymbolic%20AI/Discovering%20Symbolic%20Models%20from%20Deep%20Learning%20with%20Inductive%20Biases.md)
	- 신경망이 주고받는 메시지에서 물리 법칙을 꺼내기
	- 2020년, 인용수 1054
- [DreamCoder: growing generalizable interpretable knowledge with wake–sleep Bayesian program learning](DreamCoder%20-%20growing%20generalizable%20interpretable%20knowledge%20with%20wake–sleep%20Bayesian%20program%20learning.md)
	- 깨어서 풀고 잠들어 추상화하며 꿈꾸어 익히는 프로그램 학습기
	- 2023년, 인용수 400
#### 5) 논리규칙 학습
- [OptNet - Differentiable Optimization as a Layer in Neural Networks](./2.%20Neurosymbolic%20AI/OptNet%20-%20Differentiable%20Optimization%20as%20a%20Layer%20in%20Neural%20Networks.md)
	- 제약 최적화 문제를 신경망의 한 층으로 넣고 규칙을 제약으로 학습하기
	- 2017년, 인용수 1923
- Learning explanatory rules from noisy data
	- 
	- 2018년, 인용수 844
- [SATNet : Bridging deep learning and logical reasoning using a differentiable satisfiability solver](./2.%20Neurosymbolic%20AI/SATNet%20-%20Bridging%20deep%20learning%20and%20logical%20reasoning%20using%20a%20differentiable%20satisfiability%20solver.md)
	- SATNet: 딥러닝에 논리 추론을 심는 법
	- 2019년, 인용수 417
- [Inductive Relation Prediction by Subgraph Reasoning](./2.%20Neurosymbolic%20AI/Inductive%20Relation%20Prediction%20by%20Subgraph%20Reasoning.md)
	- 서브그래프 추론으로 관계를 예측하다 — GraIL 논문 읽기
	- 2020년, 인용수 738
- 
#### 6) 제약식을 만족하는 출력이 나오도록 신경망 학습 및 추론 (신뢰성)
- [Incorporating Prior Domain Knowledge into Deep Neural Networks](./2.%20Neurosymbolic%20AI/Incorporating%20Prior%20Domain%20Knowledge%20into%20Deep%20Neural%20Networks.md)
	- 도메인 지식을 손실 함수에 넣는다는 것 — DANN 논문 읽기
	- 2018년, 인용수 275
- [A Semantic Loss Function for Deep Learning with Symbolic Knowledge](./2.%20Neurosymbolic%20AI/A%20Semantic%20Loss%20Function%20for%20Deep%20Learning%20with%20Symbolic%20Knowledge.md)
	- 신경망에 논리를 붙이는 법: A Semantic Loss Function for Deep Learning with Symbolic Knowledge 읽기
	- 2018년, 인용수 788
- [Augmenting Neural Networks with First-order Logic](./2.%20Neurosymbolic%20AI/Augmenting%20Neural%20Networks%20with%20First-order%20Logic.md)
	- 논리식을 신경망 층으로 컴파일하기 — Augmenting Neural Networks with First-order Logic
	- 2019년, 인용수 182
- [PICARD- Parsing Incrementally for Constrained Auto-Regressive Decoding from Language Models](./2.%20Neurosymbolic%20AI/PICARD-%20Parsing%20Incrementally%20for%20Constrained%20Auto-Regressive%20Decoding%20from%20Language%20Models.md)
	- 디코딩 단계마다 문법에 어긋나는 토큰을 지워 언어모델이 유효한 SQL만 만들게 하기
	- 2021년, 인용수 931
- [Semantic Probabilistic Layers for Neuro-Symbolic Learning](./2.%20Neurosymbolic%20AI/Semantic%20Probabilistic%20Layers%20for%20Neuro-Symbolic%20Learning.md)
	- 논리 제약을 100% 보장하는 예측 레이어
	- 2022년, 인용수 198
- [Tractable Control for Autoregressive Language Generation](./2.%20Neurosymbolic%20AI/Tractable%20Control%20for%20Autoregressive%20Language%20Generation.md)
	- 다루기 쉬운 확률모형으로 언어모델에 어휘 제약을 보장하기
	- 2023년, 인용수 106
- [CCN+ : A neuro-symbolic framework for deep learning with requirements](./2.%20Neurosymbolic%20AI/CCN+%20-%20A%20neuro-symbolic%20framework%20for%20deep%20learning%20with%20requirements.md)
	- 규칙을 "반드시" 지키는 신경망: CCN+ 뜯어보기
	- 2024년, 인용수 45(2026.8)

#### 7) 온톨로지 학습
- [End-to-End Ontology Learning with Large Language Models](./2.%20Neurosymbolic%20AI/End-to-End%20Ontology%20Learning%20with%20Large%20Language%20Models.md)
	- LLM으로 온톨로지를 처음부터 끝까지 학습하기 — OLLM 논문 읽기
	- 2024년, 인용수 51

#### 8) LLM과 뉴로심볼릭의 결합
- [PAL: Program-aided Language Models](./2.%20Neurosymbolic%20AI/PAL%20Program-aided%20Language%20Models.md)
	- 문제 이해는 언어모델에게, 계산은 인터프리터에게
	- 2023년, 인용수 2223
- [Logic-LM: Empowering Large Language Models with Symbolic Solvers for Faithful Logical Reasoning](./2.%20Neurosymbolic%20AI/Logic-LM-%20Empowering%20Large%20Language%20Models%20with%20Symbolic%20Solvers%20for%20Faithful%20Logical%20Reasoning.md)
	- 논리 추론을 LLM이 아닌 기호 솔버에 맡기기
	- 2023년, 인용수 883
## 3. Ontology and Knowledge Graph

#### 논문 리뷰
- [Semi-Supervised Classification with Graph Convolutional Networks](./3.%20Ontology%20and%20Knowldeg%20Graph/Semi-Supervised%20Classification%20with%20Graph%20Convolutional%20Networks.md)
	- 소수의 라벨로 그래프 전체를 분류하기: GCN 논문 뜯어보기
	- 2016년, 인용수 61210
- [Modeling Relational Data with Graph Convolutional Networks](./3.%20Ontology%20and%20Knowldeg%20Graph/Modeling%20Relational%20Data%20with%20Graph%20Convolutional%20Networks.md)
	- 지식 그래프를 위한 그래프 신경망: R-GCN 뜯어보기
	- 2018년, 인용수 8833
- [KagNet - Knowledge-Aware Graph Networks for Commonsense Reasoning](./3.%20Ontology%20and%20Knowldeg%20Graph/KagNet%20-%20Knowledge-Aware%20Graph%20Networks%20for%20Commonsense%20Reasoning.md)
	- KagNet: 외부 지식그래프로 상식 추론을 풀고, 그 근거까지 보여주기
	- 2019년, 인용수 705
- [Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering](./3.%20Ontology%20and%20Knowldeg%20Graph/Ontology-Guided%20Reverse%20Thinking%20Makes%20Large%20Language%20Models%20Stronger%20on%20Knowledge%20Graph%20Question%20Answering.md)
	- 목적에서 조건으로 거꾸로 걷기 — ORT 논문 읽기
	- 2025년, 인용수 31
- [Construct, Align, and Reason - Large Ontology Models for Enterprise Knowledge Management](./3.%20Ontology%20and%20Knowldeg%20Graph/Construct,%20Align,%20and%20Reason%20-%20Large%20Ontology%20Models%20for%20Enterprise%20Knowledge%20Management.md)
	- 온톨로지를 LLM에 "모달리티"로 붙이기: Large Ontology Model 읽기
	- 2026년, 인용수 2

## 4. VQA (Visual Question Answering) 관련 연구

#### 논문 리뷰
- Graph-structured representations for visual question answering
	- scene graph를 제안
	- 2017년, 인용수 598
- [CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning](./4.%20Visual%20Question%20Answering/CLEVR%20A%20Diagnostic%20Dataset%20for%20Compositional%20Language%20and%20Elementary%20Visual%20Reasoning.md)
	- 정답을 먼저 계산하고 이미지를 나중에 그리는 데이터셋
	- 2017년, 인용수 3686
- [Inferring and Executing Programs for Visual Reasoning](./4.%20Visual%20Question%20Answering/Inferring%20and%20Executing%20Programs%20for%20Visual%20Reasoning.md)
	- 질문을 프로그램으로 번역하고, 그 프로그램을 신경망으로 실행하기
	- 2017년, 인용수 787
- [Neural-Symbolic VQA - Disentangling Reasoning from Vision and Language Understanding](./4.%20Visual%20Question%20Answering/Neural-Symbolic%20VQA%20-%20Disentangling%20Reasoning%20from%20Vision%20and%20Language%20Understanding.md)
	- 지각과 추론을 분리하여 기호 실행기 위에서 시각 질문에 답하기
	- 2019년, 인용수 1021
- Explainable and Explicit Visual Reasoning over Scene Graphs
	- CLEVR 방법론에 scene graph를 명시적으로 결합
	- 2019년, 인용수 319
- [Probabilistic Neural-symbolic Models for Interpretable Visual Question Answering](./4.%20Visual%20Question%20Answering/Probabilistic%20Neural-symbolic%20Models%20for%20Interpretable%20Visual%20Question%20Answering.md)
	- 프로그램을 확률변수로: Prob-NMN 읽기
	- 2019년, 인용수 122
- Deep modular co-attention networks for visual question answering
	- LSTM 기반 질문 인코더를 사실상 Transformer-style attention으로 대체한 초기 대표 연구
	- 2019년, 인용수 1390
- Blip: Bootstrapping language-image pre-training for unified vision-language understanding and generation
	- image–text retrieval, captioning, VQA를 하나의 vision-language pretraining 틀로 통합
	- 2022년, 인용수 10265

## 5. 멀티모달 기초

#### 논문 리뷰
- [Learning Transferable Visual Models From Natural Language Supervision](./5.%20Fundamentals%20of%20Multimodal%20AI(VLM)/Learning%20Transferable%20Visual%20Models%20From%20Natural%20Language%20Supervision.md)
	- CLIP 읽기: 자연어 감독으로 전이 가능한 시각 모델 학습하기
	- 2021년, 인용수 71035
- [Learning to Prompt for Vision-Language Models](./5.%20Fundamentals%20of%20Multimodal%20AI(VLM)/Learning%20to%20Prompt%20for%20Vision-Language%20Models.md)
	- CLIP에게 프롬프트를 학습시키다 — CoOp 논문 읽기
	- 2022년, 인용수 5741

## 6. Continual Learning 관련 연구

#### 대표적인 Survey 논문
- [A Comprehensive Survey of Continual Learning - Theory, Method and Application](./6.%20Continual%20Learning/A%20Comprehensive%20Survey%20of%20Continual%20Learning%20-%20Theory,%20Method%20and%20Application.md)
	- AI는 새로운 지식을 배우면서 과거의 지식을 어떻게 잊지 않을 수 있을까?
	- 2024년, 인용수 2581
#### 논문 리뷰
- [Overcoming catastrophic forgetting in neural networks](./6.%20Continual%20Learning/Overcoming%20catastrophic%20forgetting%20in%20neural%20networks.md)
	- 파국적 망각을 넘어서: EWC 논문 읽기
	- 2017년, 인용수 14536
- [Continual Learning Through Synaptic Intelligence](./6.%20Continual%20Learning/Continual%20Learning%20Through%20Synaptic%20Intelligence.md)
	- 시냅스에 지능을 더하다 — Continual Learning Through Synaptic Intelligence 읽기
	- 2017년, 인용수 4635
- [DualPrompt Complementary Prompting for Rehearsal-free Continual Learning](./6.%20Continual%20Learning/DualPrompt%20Complementary%20Prompting%20for%20Rehearsal-free%20Continual%20Learning.md)
	- DualPrompt: 버퍼 없이 망각을 막는 연속학습 — 두 종류의 프롬프트 이야기
	- 2022년, 인용수 1261
- [Learning to Prompt for Continual Learning](./6.%20Continual%20Learning/Learning%20to%20Prompt%20for%20Continual%20Learning.md)
	- Learning to Prompt for Continual Learning (L2P) 읽기
	- 2022년, 인용수 1879
- [Introducing Language Guidance in Prompt-based Continual Learning](./6.%20Continual%20Learning/Introducing%20Language%20Guidance%20in%20Prompt-based%20Continual%20Learning.md)
	- 언어 표현은 태스크가 바뀌어도 같은 의미 공간 안에 놓여 있다
	- 2023년, 인용수 105

## 7. Time Series Forecasting 관련 연구

#### 논문 리뷰
- [TimeXer- Empowering Transformers for Time Series Forecasting with Exogenous Variables](./7.%20Time%20Series/TimeXer-%20Empowering%20Transformers%20for%20Time%20Series%20Forecasting%20with%20Exogenous%20Variables.md)
	- TimeXer: 외생 변수를 품은 시계열 예측 Transformer
	- 2024년, 인용수 844
- [Towards a General Time Series Anomaly Detector with Adaptive Bottlenecks and Dual Adversarial Decoders](./7.%20Time%20Series/Towards%20a%20General%20Time%20Series%20Anomaly%20Detector%20with%20Adaptive%20Bottlenecks%20and%20Dual%20Adversarial%20Decoders.md)
	- 하나의 모델로 모든 도메인의 이상을 탐지한다: DADA 논문 소개
	- 2025년, 인용수 71


## 8.  추가 예정

- Ontology Reasoning with Deep Neural Networks
	- 2021년, 인용수 173

- Extracting Tree-Structured Representations of Trained Networks
	- 논리규칙 습득으로 볼 수 있을까?
	- 1995년, 인용수 1381
- Extracting refined rules from knowledge based neural networks
	- 1993년, 인용수 1142
- Differentiable learning of logical rules for knowledge base reasoning
	- 2017년, 인용수 1098
- Improved logical reasoning of language models via differentiable symbolic programming
	- 2023년, 인용수 64
- Mastering the game of Go with deep neural networks and tree search
	- 2016년, 인용수 26384
- Toolformer: Language Models Can Teach Themselves to Use Tools
	- 2023년, 인용수 8270

- Solving olympiad geometry without human demonstrations
	- 
	- 2024년, 인용수 1325

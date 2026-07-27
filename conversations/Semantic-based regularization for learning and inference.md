# 대화 전체 원문

## 사용자


# Files mentioned by the user:

## P37.Semantic-based regularization for learning and inference.pdf: /Users/sangunpark/Dropbox/0. 공부/2026 연대세미나/P37.Semantic-based regularization for learning and inference.pdf

## My request for Codex:
이 논문의 개요를 설명해줘

---

## 어시스턴트

PDF 읽기·검증 지침을 적용해 논문의 구조, 핵심 수식, 실험 결과를 확인한 뒤 한국어로 개요를 정리하겠습니다. 우선 원문에서 초록과 주요 절을 추출해 전체 흐름을 잡을게요.

---

## 어시스턴트

원문은 23쪽이며 암호화되지 않았습니다. 현재 기본 환경에 PDF 텍스트 추출 도구가 없어, 문서 작업용 번들 런타임을 불러와 추출과 페이지 렌더링을 이어가겠습니다.

---

## 어시스턴트

핵심 구조가 확인됐습니다. 이 논문은 단순히 “논리 규칙을 후처리에 쓰는” 방식이 아니라, 퍼지화한 1차 논리의 위반량을 정규화 항으로 넣어 커널 분류기까지 역전파하는 접근입니다. 이제 세 실험(CORA, WebKB, ArnetMiner)의 비교표와 논문이 스스로 밝힌 한계를 확인하고 있습니다.

---

## 어시스턴트

이 논문의 핵심은 **데이터로부터의 학습과 논리적 추론을 하나의 최적화 문제로 통합**하는 것입니다. 저자들은 이를 **Semantic-Based Regularization(SBR)**이라고 부릅니다.

### 1. 문제의식

기존 방법은 대체로 두 부류로 나뉩니다.

- SVM·커널 머신은 텍스트의 TF-IDF 같은 고차원 실수 특징을 잘 처리하지만, 객체 사이의 관계나 논리 규칙을 직접 표현하기 어렵습니다.
- Markov Logic Network(MLN), Probabilistic Soft Logic(PSL) 같은 통계적 관계 학습은 논리와 관계를 다루지만, 복잡한 특징 벡터를 효율적으로 학습 모델과 결합하기 어렵습니다.

SBR은 커널 머신과 1차 논리(First-Order Logic)를 연결하여, **논리적 추론의 결과가 특징 기반 분류기 학습에도 영향을 주도록** 설계합니다.

### 2. 핵심 아이디어

학습해야 할 술어 \(p_k(x)\)를 커널 머신 \(f_k(x)\)로 근사합니다. 예를 들어

\[
Category_i(x)
\]

라는 술어는 문서 \(x\)가 범주 \(i\)에 속하는 정도를 출력하는 함수 \(f_i(x)\)가 됩니다.

한편 다음과 같은 사전지식이 있다고 합시다.

\[
\forall x,y\quad Cite(x,y)\Rightarrow
\big(Category_i(x)\leftrightarrow Category_i(y)\big)
\]

즉, “서로 인용하는 논문은 같은 분야일 가능성이 높다”는 규칙입니다.

SBR은 이 논리식을 퍼지 논리로 변환해 \(0\)과 \(1\) 사이의 만족도

\[
\Phi_h(f(X))
\]

를 계산합니다. 규칙이 완전히 만족되면 1, 위반될수록 0에 가까워집니다.

최종 학습 목적함수는 다음 형태입니다.

\[
C[f]
=
\sum_k \lVert f_k\rVert_{\mathcal H_k}^{2}
+
\sum_h \lambda_h\bigl(1-\Phi_h(f(X))\bigr)
\]

- 첫 번째 항: 함수가 지나치게 복잡해지는 것을 막는 일반적인 커널 정규화
- 두 번째 항: 논리 규칙의 위반에 대한 페널티
- \(\lambda_h\): 각 규칙의 중요도

따라서 모델은 **데이터를 잘 설명하면서도 주어진 의미론적 규칙을 최대한 만족하는 함수**를 찾습니다.

### 3. 논리를 미분 가능한 형태로 바꾸는 방법

논리 연산은 t-norm 기반 퍼지 논리로 연속화됩니다.

- 부정: \(\neg a \rightarrow 1-a\)
- 논리곱: \(\min(a,b)\) 또는 \(ab\)
- 논리합: \(\max(a,b)\) 등
- 존재 한정자 \(\exists\): 가능한 grounding 중 최댓값
- 전칭 한정자 \(\forall\): 모든 grounding의 평균

고전적인 퍼지 논리에서는 \(\forall\)를 최솟값으로 표현하지만, 이 논문은 학습 효율을 위해 평균을 사용합니다. 이렇게 얻은 식은 거의 모든 지점에서 미분 가능하므로, 논리 규칙의 오차를 커널 머신까지 역전파할 수 있습니다.

### 4. SBR의 구조

논문은 SBR을 다음과 같은 다층 구조로 해석합니다.

1. 커널 머신이 각 술어의 grounding 값을 계산
2. 퍼지 논리 연산이 명제 부분의 만족도를 계산
3. 한정자 층이 grounding들을 평균 또는 최댓값으로 집계
4. 모든 공식의 위반량을 합산
5. 그 오차를 역전파하여 커널 머신을 수정

핵심은 논리 추론이 별도의 후처리가 아니라 **기초 분류기 자체를 바꾸는 학습 신호**로 작동한다는 점입니다.

특징 벡터가 없는 순수한 기호 객체도 처리할 수 있습니다. 이 경우 Gram 행렬을 단위행렬로 놓고 각 객체의 술어 값을 직접 최적화합니다. 따라서 특징이 있는 논문과 특징이 없는 저자처럼 서로 다른 종류의 객체를 하나의 모델에서 다룰 수 있습니다.

### 5. 기존 방법과의 관계

저자들은 여러 기존 학습법이 SBR의 특수한 경우임을 보입니다.

- 지도학습 규칙만 사용하면 일반적인 SVM
- 연결된 객체의 예측을 비슷하게 만드는 규칙을 넣으면 manifold/Laplacian 정규화
- 예측을 0 또는 1에 가깝게 만드는 규칙을 넣으면 Transductive SVM
- 특징 없이 관계만 사용하면 graph regularization
- 클래스 간 포함관계를 규칙으로 넣으면 계층적 분류

또한 확률적으로는 SBR을 **커널 머신의 Gaussian prior와 퍼지 논리 potential을 결합한 MAP 추정**으로 해석할 수 있습니다. 저자들은 이를 MLN의 연속적인 퍼지 확장인 Markov Fuzzy Logic Network와 연결합니다.

### 6. Collective classification

학습이 끝난 뒤 테스트 객체들을 개별적으로 분류하는 대신, 테스트 집합 전체의 관계를 함께 고려해 예측을 조정할 수도 있습니다.

\[
C_{\text{cc}}
=
\frac12\sum_k
\lVert \bar f_k(X')-f_k(X')\rVert^2
+
\sum_h\bigl(1-\Phi_h(\bar f(X'))\bigr)
\]

즉,

- 원래 커널 머신의 예측에서 너무 멀어지지 않으면서
- 테스트 집합에서도 논리 규칙을 만족하도록

예측값을 공동으로 수정합니다.

### 7. 실험 결과

세 가지 관계형 데이터셋에서 평가합니다.

- **CORA:** 논문 특징, 인용 관계, 저자 관계를 결합합니다. 논문 분류 정확도에서 SBR은 라벨 비율 10%일 때 0.511, 50%일 때 0.656을 기록해 SVM, TSVM, MLN, PSL보다 높았습니다.
- **WebKB:** 웹페이지와 링크 앵커를 함께 분류합니다. SBR의 collective classification은 F1 기준 웹페이지 0.810, 앵커 0.700으로 비교 방법 중 가장 좋았습니다.
- **ArnetMiner:** 영화 제목과 감독·작가·제작자 관계를 사용합니다. SBR이 SVM, TSVM, Laplacian SVM, PSL보다 높았고, 테스트 관계까지 이용한 SBR-CC가 일반 SBR을 조금씩 개선했습니다.

특히 라벨이 적거나 특징 표현이 빈약한 객체에서 논리 및 관계 정보의 효과가 크게 나타납니다.

### 8. 한계

이 접근법에도 분명한 제약이 있습니다.

- 변수와 한정자가 많아지면 grounding 수가 지수적으로 증가합니다.
- 복잡한 논리식은 목적함수에 많은 지역 최솟값을 만듭니다.
- 논리 규칙의 가중치 \(\lambda_h\), 커널, t-norm 등을 선택해야 합니다.
- 학습 안정화를 위해 쉬운 규칙부터 단계적으로 추가하는 `dom` 휴리스틱을 사용하지만, 전역 최적해를 보장하지는 않습니다.
- 논리 규칙이 항상 참인 절대적 지식이라기보다 가중된 소프트 제약으로 취급됩니다.

### 한 줄 평가

이 논문은 **“신경망 또는 커널 모델의 예측 위에 논리를 얹는 것”을 넘어, 논리적 지식을 미분 가능한 손실로 만들어 기초 학습기까지 공동 훈련한다**는 점이 핵심입니다. 오늘날의 신경-기호 학습(neuro-symbolic learning), 논리 기반 정규화, constraint-based learning으로 이어지는 중요한 초기 통합 프레임워크로 볼 수 있습니다.

---

## 사용자

“서로 인용하는 논문은 같은 분야일 가능성이 높다”는 규칙을 논문이 제안하는 퍼지 논리식으로 변환해서 이 값을 올바르게 계산하는 것이 목적이야? 그럼 퍼지 논리식은 신경망의 한 레이어처럼 작동해?

---

## 어시스턴트

맞습니다. 다만 목적은 퍼지 논리식의 값을 계산하는 데서 끝나는 것이 아니라, **그 만족도가 높아지도록 분류기 자체를 학습하는 것**입니다.

예를 들어 \(a=C_i(x)\), \(b=C_i(y)\)를 두 논문이 분야 \(i\)에 속할 예측값이라고 하면, 규칙은

\[
\forall x,y\quad Cite(x,y)\Rightarrow
\bigl[(C_i(x)\land C_i(y))
\lor(\neg C_i(x)\land\neg C_i(y))\bigr]
\]

입니다.

논문에서 사용한 minimum t-norm을 적용하고 \(Cite(x,y)=1\)인 인용 관계만 보면, 해당 쌍의 규칙 만족도는

\[
s(x,y)
=
\max\left(
\min(a,b),
\min(1-a,1-b)
\right)
\]

가 됩니다.

예를 들면 다음과 같습니다.

| \(a\) | \(b\) | 만족도 \(s(x,y)\) | 해석 |
|---:|---:|---:|---|
| 0.9 | 0.8 | 0.8 | 둘 다 같은 분야라고 예측 |
| 0.1 | 0.2 | 0.8 | 둘 다 그 분야가 아니라고 예측 |
| 0.9 | 0.1 | 0.1 | 예측이 서로 충돌 |
| 0.5 | 0.5 | 0.5 | 일치하지만 불확실 |

모든 논문 쌍을 고려한 전체 규칙 만족도는 논문에서 전칭 한정자를 평균으로 바꾸었기 때문에 대략

\[
\Phi_i
=
\frac{1}{|P|^2}
\sum_{x,y\in P}
\Phi_{i,x,y}
\]

처럼 계산됩니다. \(Cite(x,y)=0\)인 쌍은 implication이 자동으로 참이므로 보통 만족도 1을 주며, 실제 구현에서는 이런 자명한 grounding을 제거할 수도 있습니다.

학습 목적함수에는 다음과 같이 들어갑니다.

\[
C[f]
=
\underbrace{\sum_i\lVert f_i\rVert^2}_{\text{모델 복잡도}}
+
\underbrace{\text{지도 데이터 손실}}_{\text{라벨 적합}}
+
\underbrace{\lambda_i(1-\Phi_i)}_{\text{인용 규칙 위반}}
\]

따라서 인용한 두 논문의 예측이 다르면 \(1-\Phi_i\)가 커지고, 이 손실의 기울기가 분류기 파라미터까지 전달되어 두 예측을 더 일관되게 만듭니다.

### 신경망 레이어처럼 작동하는가?

네. **계산 그래프라는 관점에서는 신경망의 고정된 레이어처럼 작동합니다.**

```text
논문의 특징 벡터
        ↓
커널 분류기 f_i(x)
        ↓
술어의 퍼지 진릿값
        ↓
AND·OR·NOT 등의 t-norm 연산
        ↓
∀·∃ 한정자 집계
        ↓
규칙 만족도 Φ
        ↓
규칙 손실 1 - Φ
        ↓
분류기 파라미터로 역전파
```

논문도 이를 명시적으로 다층 구조라고 설명합니다.

다만 일반적인 신경망 레이어와는 차이가 있습니다.

- 원 논문의 하위 학습기는 신경망이 아니라 **커널 머신**입니다.
- 논리 레이어의 AND, OR, NOT 연산은 데이터로 학습되는 것이 아니라 **논리식에 따라 고정**됩니다.
- 학습되는 것은 주로 술어를 근사하는 함수 \(f_i\)의 파라미터입니다.
- `min`과 `max`는 일부 지점에서 미분 불가능하지만, 거의 모든 지점에서는 gradient/subgradient를 계산할 수 있습니다.
- 규칙 가중치 \(\lambda_i\)는 실험에서 주로 검증 데이터로 정하지만, 논문은 확률적 해석을 통해 이를 학습할 가능성도 언급합니다.

현대적인 표현으로 바꾸면 SBR의 퍼지 논리 부분은 **신경망 위에 붙이는 differentiable logic layer 또는 logic-based loss layer**에 가깝습니다. 중요한 점은 이 레이어가 추론 결과만 내는 것이 아니라, 역전파를 통해 특징 기반 분류기의 학습 방향까지 바꾼다는 것입니다.

---

## 사용자

주어진 문제에 따라 퍼지논리식이 바뀔 것 같은데, 그럼 그에 맞춰서 레이어를 새로 설계해야 해?

---

## 어시스턴트

네, 문제마다 논리식과 계산 그래프의 모양은 달라집니다. 하지만 보통 **레이어 코드를 매번 처음부터 새로 설계할 필요는 없습니다.** 기본 논리 연산자를 재사용해 논리식을 계산 그래프로 자동 변환하면 됩니다.

예를 들어 공통 연산자를 미리 구현합니다.

\[
\begin{aligned}
\operatorname{NOT}(a)&=1-a\\
\operatorname{AND}(a,b)&=\min(a,b)\\
\operatorname{OR}(a,b)&=\max(a,b)\\
\operatorname{FORALL}(a_1,\dots,a_n)&=\frac1n\sum_j a_j\\
\operatorname{EXISTS}(a_1,\dots,a_n)&=\max_j a_j
\end{aligned}
\]

그다음 문제에 맞는 지식만 논리식으로 작성합니다.

```text
Cite(x,y) → SameCategory(x,y)
AuthorOf(a,p) → SameField(a,p)
SameDirector(x,y) → SameGenre(x,y)
```

시스템은 각 논리식을 구문 트리로 바꾸고, 기존 연산자를 조립해 계산 그래프를 생성합니다.

```text
Cite(x,y) → (Category(x) ↔ Category(y))
                          │
          ┌───────────────┴──────────────┐
     AND(Cx, Cy)               AND(NOT(Cx), NOT(Cy))
          └───────────────┬──────────────┘
                          OR
                           │
                      implication
                           │
                  FORALL over (x,y)
                           │
                    규칙 만족도 Φ
```

따라서 역할은 다음처럼 분리됩니다.

- 사용자가 설계하는 것: 술어, 객체 관계, 논리 규칙
- 시스템이 자동 생성하는 것: grounding과 논리 계산 그래프
- 미리 구현해 두는 것: AND, OR, NOT, implication, \(\forall\), \(\exists\) 같은 미분 가능한 연산
- 학습되는 것: 술어를 예측하는 모델 파라미터
- 조정하는 것: 규칙별 가중치 \(\lambda_h\)

논문에서도 논리식마다 한정자 수가 다르므로 논리 네트워크의 깊이는 고정되지 않는다고 설명합니다. 즉, SBR의 논리 레이어는 고정된 크기의 Dense layer라기보다 **논리식을 입력받아 동적으로 구성되는 계산 그래프**입니다.

새 연산을 구현해야 하는 경우는 제한적입니다.

- 기존 FOL에 없는 새로운 관계 연산이 필요한 경우
- “최소 3개가 참이어야 한다” 같은 특수 한정자를 추가하는 경우
- `min/max` 대신 다른 퍼지 의미론을 사용하려는 경우
- 시간적 논리나 확률적 규칙처럼 FOL 밖의 표현을 도입하는 경우

결론적으로, **문제마다 바뀌는 것은 레이어의 구성이지 기본 레이어 구현 자체가 아닙니다.** 현대적인 구현이라면 논리식 파서가 퍼지 연산 모듈을 조합하여 loss를 자동 생성하는 형태가 가장 자연스럽습니다. 이는 PyTorch에서 모델 구조를 매번 직접 코딩하기보다, 논리식을 일종의 선언형 모델 명세로 사용하는 것에 가깝습니다.

---

## 사용자

category를 예측하는 문제라면 출력이 category별로 계산된 값이 될 것 같은데, 위와 같은 제약은 어느 단계에 사용되는거지?

---

## 어시스턴트

Category 예측 문제라면 각 객체 \(x\)에 대해 다음과 같은 출력 벡터가 나옵니다.

\[
\mathbf p(x)=
[p_1(x),p_2(x),\ldots,p_K(x)]
\]

여기서 \(p_c(x)\)는 \(x\)가 category \(c\)에 속하는 정도입니다. 논리 제약은 보통 **이 출력값이 계산된 직후, 최종 loss를 만드는 단계**에서 사용됩니다.

```text
입력 특징 x
   ↓
분류기
   ↓
category별 출력 p₁(x), …, pK(x)
   ├── 지도학습 loss
   └── 퍼지 논리 제약 loss
              ↓
        전체 loss 합산
              ↓
           역전파
```

### 인용 관계 제약의 사용

논문 \(x\)가 \(y\)를 인용한다면 각 category \(c\)에 대해

\[
Cite(x,y)\Rightarrow
\bigl(C_c(x)\leftrightarrow C_c(y)\bigr)
\]

를 적용할 수 있습니다.

예를 들어 분류기가 다음과 같이 출력했다고 합시다.

\[
\mathbf p(x)=[0.8,0.1,0.1]
\]

\[
\mathbf p(y)=[0.2,0.7,0.1]
\]

두 논문은 서로 인용하지만 첫 번째와 두 번째 category에서 예측이 크게 다릅니다. 따라서 논리 규칙의 만족도가 낮아지고, 제약 손실

\[
L_{\text{relation}}
=
\lambda\left(1-\Phi_{\text{relation}}\right)
\]

이 커집니다.

최종 학습 손실은 개념적으로

\[
L
=
L_{\text{supervised}}
+
\lambda_{\text{cite}}L_{\text{cite}}
+
\lambda_{\text{exclusive}}L_{\text{exclusive}}
+
L_{\text{regularization}}
\]

처럼 구성됩니다.

역전파가 일어나면 인용 관계 제약은 \(x\)와 \(y\)의 category 예측이 더 비슷해지도록 분류기 파라미터를 수정합니다. 하지만 라벨 데이터가 두 논문이 실제로 다른 category임을 강하게 보여준다면, 지도학습 손실과 논리 제약 사이에서 절충하게 됩니다. 따라서 “인용하면 반드시 같은 category”라는 hard rule이라기보다 가중치가 있는 경향성입니다.

### Category 사이의 제약도 가능

관계 제약뿐 아니라 동일한 객체의 category 출력 사이에도 규칙을 적용할 수 있습니다. 단일 분류 문제라면

\[
\forall x,\quad
\text{정확히 하나의 }C_1(x),\ldots,C_K(x)\text{가 참}
\]

이라는 제약을 사용할 수 있습니다.

원 논문은 각 category를 별도의 퍼지 술어로 취급하므로, 출력이 반드시 합계 1인 softmax일 필요는 없습니다. 대신 “하나의 category만 선택해야 한다”는 조건도 논리 규칙으로 표현합니다. 멀티라벨 문제라면 이 배타적 제약을 넣지 않습니다.

### 학습 때만 사용하는가?

두 가지 방식이 있습니다.

1. **일반적인 SBR 학습**

   논리 제약은 학습 loss에 들어갑니다. 학습이 끝난 뒤에는 새 논문을 분류기에 넣어 바로 category를 예측합니다. 이때 논리 레이어를 다시 실행하지 않아도 됩니다. 제약의 효과가 이미 분류기 파라미터에 반영되어 있기 때문입니다.

2. **Collective classification**

   테스트 단계에도 인용 관계가 주어진다면 논리 제약을 다시 사용합니다. 먼저 분류기의 예측 \(p_c(x)\)를 만든 다음,

   \[
   \frac12\sum_{x,c}
   \left(\bar p_c(x)-p_c(x)\right)^2
   +
   \lambda\sum_h(1-\Phi_h(\bar p))
   \]

   를 최소화합니다. 원래 예측에서 너무 멀어지지 않으면서 테스트 집합 전체가 인용 규칙을 더 잘 만족하도록 \(\bar p\)를 공동으로 조정하는 것입니다.

결국 제약은 주로 **category 출력과 최종 loss 사이**에 위치합니다. 학습 때는 분류기 파라미터를 바꾸는 보조 손실로 사용되고, collective classification에서는 테스트 출력 자체를 공동으로 보정하는 추론 모듈로 한 번 더 사용됩니다.

---

## 사용자

인용 관계 제약은 주어진 두 논문에 대해서 계산되어야 하잖아. 반면에 카테고리 예측은 하나의 논문에 대해 실행하면 되고. 그렇다면 학습할 때에는 모든 논문 쌍에 대해서 제약을 만족하는지를 봐야 해? 그 말은 제약 만족에 대한 학습을 위해 전체 논문을 한꺼번에 입력으로 받아야 한다는 뜻이야?

---

## 어시스턴트

수학적으로는 전칭 한정자 때문에 모든 논문 쌍을 고려하지만, 실제 계산에서는 **인용 관계가 있는 쌍만 보면 됩니다.** 그리고 전체 논문을 하나의 거대한 입력으로 넣을 필요도 없습니다.

핵심은 분류기는 여전히 논문 하나씩 처리하지만, **학습 손실이 두 논문의 예측을 결합한다**는 것입니다.

### 1. 모든 \(N^2\)개의 논문 쌍을 계산해야 하나?

규칙이

\[
\forall x,y\quad Cite(x,y)\Rightarrow
\bigl(C_c(x)\leftrightarrow C_c(y)\bigr)
\]

라고 하면 형식적으로는 \(N^2\)개의 \((x,y)\) 조합이 존재합니다.

하지만 \(Cite(x,y)=0\)이면 implication은 자동으로 참입니다.

\[
0\Rightarrow \text{anything} = 1
\]

따라서 인용 관계가 없는 쌍은 항상 만족도 1을 주며, 학습 gradient도 0입니다. 계산할 필요가 없습니다. 실제 제약 손실은 인용 edge 집합 \(E\)에 대해서만

\[
L_{\text{cite}}
=
\frac{1}{|E|}
\sum_{(x,y)\in E}
\sum_{c=1}^{K}
\left[
1-\operatorname{Eq}_{\text{fuzzy}}
\bigl(p_c(x),p_c(y)\bigr)
\right]
\]

처럼 계산할 수 있습니다.

즉 계산량은 일반적으로 \(O(N^2)\)가 아니라 인용 관계 수에 비례하는 \(O(|E|)\)입니다. 논문에서도 자명하게 참이 되는 grounding을 제거하는 pruning을 사용합니다.

### 2. 전체 논문을 한꺼번에 입력해야 하나?

필요하지 않습니다. 분류기와 제약 손실을 분리해서 보면 명확합니다.

```text
논문 x ─→ 동일한 분류기 ─→ p(x) ─┐
                                  ├→ 인용 관계 loss
논문 y ─→ 동일한 분류기 ─→ p(y) ─┘
```

분류기는 각 논문을 독립적으로 처리합니다.

\[
p(x)=f_\theta(x),\qquad p(y)=f_\theta(y)
\]

다만 학습 loss를 계산할 때 두 출력 \(p(x)\), \(p(y)\)를 함께 사용합니다. Siamese network나 contrastive learning에서 두 샘플의 출력을 비교하는 것과 비슷한 구조입니다.

### 3. 미니배치 학습은 어떻게 하나?

현대적인 신경망 구현이라면 논문이 아니라 **인용 edge를 미니배치로 샘플링**할 수 있습니다.

예를 들어 한 배치가 다음 인용 관계를 포함한다고 합시다.

```text
(A, B), (A, C), (D, E)
```

이 배치에서 필요한 고유 논문은

```text
A, B, C, D, E
```

입니다. 각 논문을 분류기에 한 번씩 통과시킵니다.

\[
p(A),p(B),p(C),p(D),p(E)
\]

그다음 edge별로 제약 손실을 계산합니다.

\[
L_{\text{batch}}
=
\ell(p(A),p(B))
+\ell(p(A),p(C))
+\ell(p(D),p(E))
\]

여기에 라벨이 있는 논문의 지도학습 손실을 더합니다.

\[
L
=
L_{\text{supervised}}
+
\lambda L_{\text{batch}}
\]

이렇게 하면 전체 인용 그래프를 한 번에 메모리에 올리지 않고도 stochastic gradient descent를 사용할 수 있습니다. 여러 배치에서 같은 분류기 파라미터 \(\theta\)를 공유하기 때문에 전체 그래프의 제약이 점차 모델에 반영됩니다.

### 4. 원 논문의 방식과 현대적 구현의 차이

원 논문의 이론적 목적함수는 유한한 데이터 영역에 대한 모든 grounding을 한꺼번에 정의합니다. 커널 머신과 Gram 행렬을 사용하므로 실제 구현도 현대적인 신경망 미니배치 방식보다 전체 데이터에 더 가까운 형태입니다. 대신 다음과 같이 계산량을 줄입니다.

- 인용 관계가 없어 규칙이 자명하게 참인 grounding 제거
- evidence만으로 결과가 결정되는 grounding 제거
- FROG 등의 MLN grounding pruning 기법 활용

하지만 SBR의 논리적 원리가 전체 논문을 반드시 한 입력 텐서로 요구하는 것은 아닙니다. 신경망으로 구현한다면 edge sampling이나 subgraph sampling으로 근사할 수 있습니다.

### 5. 전체 데이터를 봐야 하는 제약도 있다

모든 제약이 edge 미니배치로 처리되는 것은 아닙니다. 예를 들어

\[
\text{전체 논문의 약 20\%는 category }c
\]

또는

\[
\exists_n x\; C_c(x)
\]

처럼 데이터셋 전체의 개수나 분포를 요구하는 규칙은 더 넓은 범위를 봐야 합니다. 이런 경우에는 다음 중 하나가 필요합니다.

- 전체 데이터에 대해 주기적으로 계산
- 큰 배치나 그래프 컴포넌트 단위 계산
- 미니배치 통계로 전체 통계를 추정
- 이동평균으로 전역 통계를 근사

정리하면, **점별 예측기는 논문 하나를 입력받고, 관계 제약은 인용 edge로 연결된 두 예측을 학습 loss에서 결합합니다.** 이론적으로는 전체 grounding 위의 규칙이지만, 인용 제약의 경우에는 실제 edge만 미니배치로 샘플링하면 되므로 전체 논문을 매번 한꺼번에 입력할 필요는 없습니다.

---

## 사용자

이 논문에 representer theorem이 사용되고 있어?

---

## 어시스턴트

네. **Representer Theorem이 SBR의 핵심 수학적 연결고리로 사용됩니다.**

논문은 먼저 함수 공간에서 다음 최적화 문제를 정의합니다.

\[
C[f]
=
\sum_{k=1}^{T}\|f_k\|_{\mathcal H_k}^2
+
\sum_{h=1}^{H}
\lambda_h\left(1-\Phi_h(f(X))\right)
\tag{2}
\]

여기서

- \(f_k\): 술어 \(k\)를 근사하는 함수
- \(\mathcal H_k\): \(f_k\)가 속하는 RKHS
- \(\|f_k\|_{\mathcal H_k}^2\): 함수 복잡도 정규화
- \(\Phi_h(f(X))\): 유한한 데이터 grounding에서 계산한 논리식 \(h\)의 퍼지 만족도

입니다.

Representer Theorem을 적용하면 최적해가 다음과 같은 유한한 커널 전개 형태를 갖습니다.

\[
f_k^*(x)
=
\beta_k+
\sum_{i=1}^{|X_k^\circ|}
w_{ki}^*K_k(x_{ki},x)
\tag{3}
\]

즉, 무한차원 함수 공간 전체에서 \(f_k\)를 찾을 필요 없이 학습 데이터에 놓인 커널 함수들의 선형결합만 찾으면 됩니다.

### 왜 Representer Theorem을 적용할 수 있나?

목적함수가 다음 두 정보에만 의존하기 때문입니다.

1. 함수의 RKHS norm

\[
\|f_k\|_{\mathcal H_k}^2
\]

2. 유한한 grounding에서의 함수값

\[
f_k(x_{k1}),\ldots,f_k(x_{kn})
\]

퍼지 논리 제약이 복잡하고 비선형이더라도, 결국 유한한 데이터 지점에서 계산된 \(f_k(x)\)들에만 의존합니다. 따라서 최적 함수가 데이터에 대한 커널 전개로 표현된다는 representer theorem의 논리를 그대로 사용할 수 있습니다.

### 적용 이후의 목적함수

커널 전개를 대입하면 함수 최적화가 커널 계수 \(w_k\)에 대한 유한차원 최적화로 바뀝니다.

\[
C(w)
=
\sum_{k=1}^{T}w_k^\top G_kw_k
+
\sum_{h=1}^{H}
\lambda_h
\left(
1-\Phi_h(f(X))
\right)
\tag{4}
\]

여기서 \(G_k\)는

\[
(G_k)_{ij}=K_k(x_{ki},x_{kj})
\]

로 정의되는 Gram 행렬입니다. 그리고 데이터에서의 예측값은

\[
f_k(X_k^\circ)=G_kw_k
\]

가 됩니다.

전체 흐름은 다음과 같습니다.

```text
FOL 규칙
   ↓ 퍼지 논리 변환
유한 데이터에서의 제약 Φ(f(X))
   ↓ Representer Theorem
f(x) = Σᵢ wᵢK(xᵢ,x)
   ↓
커널 계수 w에 대한 유한차원 최적화
   ↓
퍼지 논리 손실을 w까지 역전파
```

### 일반적인 커널 방법과 다른 점

일반적인 SVM에서는 representer theorem을 적용한 뒤 지도학습 손실을 최적화합니다.

\[
\text{RKHS norm}+\text{supervised loss}
\]

SBR에서는 그 손실 부분이 더 일반적인 퍼지 논리 제약으로 확장됩니다.

\[
\text{RKHS norm}
+
\text{supervised constraint}
+
\text{relational constraint}
+
\text{semantic constraint}
\]

따라서 representer theorem은 **논리 규칙을 직접 처리하는 이론**이라기보다, 퍼지 논리로 정의된 함수 최적화 문제를 실제로 계산 가능한 커널 계수 최적화 문제로 변환하는 역할을 합니다.

논문은 이 결과를 기존 representer theorem의 “straightforward extension”이라고 설명하며, 상세한 증명은 저자들의 이전 연구인 참고문헌 [10]에 맡깁니다. 특징 벡터가 없는 기호 객체에 대해서는 \(G_k=I\)로 놓아, 커널 일반화 대신 각 객체의 술어값 자체를 직접 최적화합니다.

---

## 사용자

커널 머신의 역할은 주어진 입력에 대한 grounding을 구하는 거야? 입력의 형태를 예를 들어서 보여주고 커널 머신에 의해 어떻게 변환되는지 설명해줘

---

## 어시스턴트

정확히 말하면, **커널 머신이 grounding을 구하는 것은 아닙니다.**

- Grounding: 논리 변수에 실제 객체를 대입하는 과정
- 특징 변환: 그 객체를 벡터로 표현하는 과정
- 커널 머신: grounded된 객체의 특징을 받아 미지의 술어가 참일 정도를 계산하는 과정

즉, 커널 머신의 역할은 **grounded atom의 퍼지 진릿값을 예측하는 것**입니다.

### 예: 논문의 분야 예측

다음과 같은 논리 술어가 있다고 하겠습니다.

\[
AI(x)
\]

이는 “논문 \(x\)가 AI 분야에 속한다”는 의미입니다.

논리 변수 \(x\)의 domain이

\[
\mathcal P=\{p_1,p_2,p_3\}
\]

이면 가능한 grounding은

\[
AI(p_1),\quad AI(p_2),\quad AI(p_3)
\]

입니다. 이 목록을 만드는 것이 grounding입니다. 아직 커널 머신은 사용되지 않았습니다.

각 논문에는 TF-IDF 특징 벡터가 연결되어 있다고 하겠습니다.

\[
p_1\mapsto \mathbf x_1=[1,0]
\]

\[
p_2\mapsto \mathbf x_2=[0,1]
\]

\[
p_3\mapsto \mathbf x_3=[0.8,0.2]
\]

커널 머신은 논문 식별자 \(p_3\) 자체가 아니라 그 표현 \(\mathbf x_3\)를 입력받습니다.

```text
논리 객체 p₃
    ↓ 특징 조회
TF-IDF 벡터 x₃ = [0.8, 0.2]
    ↓ 커널 머신
AI 분야 점수 fAI(x₃)
    ↓ sigmoid
퍼지 진릿값 AI(p₃) ≈ 0.668
```

### 커널 머신의 계산

Representer theorem에 따라 AI 술어를 근사하는 함수는 다음 형태입니다.

\[
f_{AI}(\mathbf x)
=
\beta+
\sum_{i=1}^{n}w_iK(\mathbf x_i,\mathbf x)
\]

여기서 \(\mathbf x_i\)들은 학습에 사용된 논문 표현입니다.

간단히 선형 커널을 사용해 보겠습니다.

\[
K(\mathbf x_i,\mathbf x)=\mathbf x_i^\top\mathbf x
\]

학습된 계수와 bias가

\[
w_1=1.2,\qquad w_2=-0.8,\qquad \beta=-0.1
\]

이라고 합시다. 새로운 논문 \(p_3\)의 특징은

\[
\mathbf x_3=[0.8,0.2]
\]

입니다. 먼저 커널 유사도를 계산합니다.

\[
K(\mathbf x_1,\mathbf x_3)
=
[1,0]^\top[0.8,0.2]
=
0.8
\]

\[
K(\mathbf x_2,\mathbf x_3)
=
[0,1]^\top[0.8,0.2]
=
0.2
\]

따라서

\[
\begin{aligned}
f_{AI}(\mathbf x_3)
&=-0.1+1.2(0.8)-0.8(0.2)\\
&=0.7
\end{aligned}
\]

SBR에서는 이를 sigmoid로 \([0,1]\) 범위에 넣습니다.

\[
\sigma(0.7)\approx 0.668
\]

그러면 grounded atom

\[
AI(p_3)
\]

의 퍼지 진릿값을

\[
AI(p_3)\approx 0.668
\]

로 사용할 수 있습니다.

### 여러 category가 있다면

Category가 AI, Theory, Systems 세 개라면 각각 별도의 미지 술어를 둘 수 있습니다.

\[
AI(x),\quad Theory(x),\quad Systems(x)
\]

각 술어는 별도의 커널 함수를 가집니다.

\[
\begin{aligned}
f_{AI}(\mathbf x_3)&\rightarrow 0.668\\
f_{Theory}(\mathbf x_3)&\rightarrow 0.121\\
f_{Systems}(\mathbf x_3)&\rightarrow 0.304
\end{aligned}
\]

따라서 논문 \(p_3\)에 대한 출력은

\[
\mathbf p(p_3)=[0.668,0.121,0.304]
\]

가 됩니다.

원 논문의 구성에서는 각 category가 독립적인 퍼지 술어이기 때문에 이 값들의 합이 반드시 1일 필요는 없습니다. 단일 category만 허용하려면 “정확히 하나가 참이어야 한다”는 논리 제약을 추가합니다.

### 이 값이 논리 레이어로 전달된다

이제 다음 인용 규칙을 생각해 보겠습니다.

\[
Cite(x,y)\Rightarrow
\bigl(AI(x)\leftrightarrow AI(y)\bigr)
\]

\(x=p_3,\;y=p_1\)로 grounding하면

\[
Cite(p_3,p_1)\Rightarrow
\bigl(AI(p_3)\leftrightarrow AI(p_1)\bigr)
\]

이 됩니다.

각 atom의 값은 서로 다른 방식으로 얻습니다.

- \(Cite(p_3,p_1)\): 데이터에 주어진 evidence, 예를 들어 1
- \(AI(p_3)\): 커널 머신 출력, 예를 들어 0.668
- \(AI(p_1)\): 커널 머신 출력, 예를 들어 0.9

따라서 논리 레이어의 입력은 다음과 같습니다.

\[
Cite(p_3,p_1)=1,\qquad
AI(p_3)=0.668,\qquad
AI(p_1)=0.9
\]

minimum t-norm을 쓰면 같은 category라는 부분의 만족도는

\[
\max
\left[
\min(0.668,0.9),
\min(1-0.668,1-0.9)
\right]
\]

이므로

\[
\max(0.668,0.1)=0.668
\]

입니다. 이 만족도가 낮으면 그에 대한 손실이 커널 머신의 계수 \(w_i\)까지 역전파됩니다.

### 전체 과정

```text
1. Grounding
   x = p₃, y = p₁

2. 특징 조회
   p₃ → x₃
   p₁ → x₁

3. 커널 머신 평가
   x₃ → AI(p₃) ≈ 0.668
   x₁ → AI(p₁) ≈ 0.900

4. Evidence 조회
   Cite(p₃,p₁) = 1

5. 퍼지 논리 계산
   Cite(p₃,p₁) → (AI(p₃) ↔ AI(p₁))

6. 규칙 위반 손실 계산

7. 커널 계수로 역전파
```

### 이항 술어를 학습할 때

커널 머신이 반드시 단일 객체만 받는 것은 아닙니다. 학습할 술어가

\[
SameTopic(x,y)
\]

처럼 이항 술어라면 입력 공간은 논문 쌍의 공간입니다.

\[
X^\circ=\mathcal P\times\mathcal P
\]

하나의 입력은

\[
(\mathbf x,\mathbf y)
\]

가 되며, 두 벡터의 결합을 사용할 수 있습니다.

\[
\mathbf z_{xy}
=
[\mathbf x;\mathbf y]
\]

또는 쌍에 직접 정의된 커널을 사용할 수 있습니다.

\[
K_{\text{pair}}
\bigl((x,y),(x',y')\bigr)
=
K(x,x')K(y,y')
\]

그러면 커널 머신이 \(SameTopic(p_i,p_j)\)의 퍼지 진릿값을 예측합니다. 다만 논문의 CORA 실험에서 `Cite` 같은 관계는 보통 학습 대상이 아니라 데이터로 주어진 evidence입니다.

요약하면, **grounding은 “어떤 객체에 대해 술어를 평가할 것인가”를 정하고, 커널 머신은 그 객체의 특징 표현을 이용해 해당 grounded 술어의 진릿값을 계산합니다.** 커널 머신은 grounding 생성기가 아니라, 논리식에 들어갈 미지 술어값을 제공하는 함수 근사기입니다.

---

## 사용자

그럼 학습을 통해 예측하고자 하는 것은 하나의 술어야? 입력은 그 술어의 변수에 대한 특성값이고, 출력은 술어에 대한 퍼지 진릿값이고 라벨인 실제 진릿값과의 차이와 규칙에 대한 만족도를 이용해서 커널 머신을 학습해?

---

## 어시스턴트

네, 거의 정확하게 이해했습니다. 더 정확히 표현하면 **하나의 커널 머신이 하나의 미지 술어를 근사하며, 여러 미지 술어를 동시에 공동 학습할 수도 있습니다.**

### 하나의 술어를 학습하는 경우

예측하려는 술어가

\[
AI(x)
\]

라면 이를 하나의 함수로 근사합니다.

\[
AI(x)\approx \sigma(f_{AI}(\mathbf x))
\]

- \(x\): 논문이라는 논리적 객체
- \(\mathbf x\): 논문의 TF-IDF 같은 특징 벡터
- \(f_{AI}\): 학습할 커널 머신
- \(\sigma(f_{AI}(\mathbf x))\): \(AI(x)\)의 퍼지 진릿값

따라서 다음처럼 이해할 수 있습니다.

```text
입력: 논문 x의 특징 벡터
           ↓
    커널 머신 fAI
           ↓
출력: AI(x)의 퍼지 진릿값 [0,1]
```

예를 들어

\[
\sigma(f_{AI}(\mathbf x))=0.85
\]

라면 “논문 \(x\)가 AI 분야라는 술어가 0.85의 정도로 참”이라는 뜻입니다.

### 라벨은 어떻게 사용되는가?

논문 \(p_1\)이 실제로 AI 분야라면

\[
AI(p_1)=1
\]

이라는 supervision이 주어집니다. 반대로 AI 분야가 아니라면

\[
AI(p_1)=0
\]

입니다.

현대적인 구현에서는 단순히 다음과 같은 지도학습 손실을 생각할 수 있습니다.

\[
L_{\text{sup}}
=
\sum_{x\in X_L}
\ell\left(y_x,\sigma(f_{AI}(\mathbf x))\right)
\]

다만 원 논문에서는 라벨 손실도 논리 규칙으로 표현합니다. 예를 들어

\[
\forall x\quad PositiveAI(x)\Rightarrow AI(x)
\]

\[
\forall x\quad NegativeAI(x)\Rightarrow \neg AI(x)
\]

여기서 `PositiveAI`와 `NegativeAI`는 정답으로 주어진 evidence 술어입니다. 이를 minimum t-norm으로 변환하면 일반적인 SVM의 hinge loss와 같은 형태가 나옵니다.

즉, “라벨과 예측값의 차이”도 SBR에서는 **supervision을 나타내는 논리식의 위반량**으로 통일해 표현할 수 있습니다.

### 의미적 규칙도 함께 사용한다

라벨 이외에 인용 관계 규칙이 있다고 하겠습니다.

\[
Cite(x,y)\Rightarrow
\bigl(AI(x)\leftrightarrow AI(y)\bigr)
\]

이를 퍼지 논리로 변환하면 인용 관계가 있는 두 논문의 출력

\[
\sigma(f_{AI}(\mathbf x)),\qquad
\sigma(f_{AI}(\mathbf y))
\]

이 서로 비슷하도록 만드는 추가 손실이 됩니다.

최종 목적함수는 개념적으로 다음과 같습니다.

\[
L
=
\underbrace{\|f_{AI}\|_{\mathcal H}^2}_{\text{함수 복잡도}}
+
\underbrace{\lambda_{\text{sup}}
\left(1-\Phi_{\text{sup}}\right)}_{\text{라벨 만족도}}
+
\underbrace{\lambda_{\text{cite}}
\left(1-\Phi_{\text{cite}}\right)}_{\text{인용 규칙 만족도}}
\]

따라서 학습은 동시에 세 가지를 추구합니다.

- 라벨이 있는 논문에서는 실제 진릿값을 맞춘다.
- 인용 관계가 있는 논문들의 예측을 일관되게 만든다.
- 지나치게 복잡한 함수를 피한다.

### 여러 술어를 공동 학습하는 경우

실제 문제에서는 하나의 술어만 학습하지 않을 수 있습니다. Category가 \(K\)개라면 보통

\[
C_1(x),C_2(x),\ldots,C_K(x)
\]

라는 \(K\)개의 미지 술어를 정의합니다. 논문의 수식에서는 이를

\[
f=\{f_1,\ldots,f_T\}
\]

로 나타냅니다. 각 \(f_k\)가 하나의 query predicate를 근사합니다.

예를 들어

\[
\begin{aligned}
AI(x)&\approx \sigma(f_{AI}(\mathbf x))\\
Theory(x)&\approx \sigma(f_{Theory}(\mathbf x))\\
Systems(x)&\approx \sigma(f_{Systems}(\mathbf x))
\end{aligned}
\]

입니다.

그리고 category 간 규칙도 넣을 수 있습니다.

\[
AI(x)\Rightarrow \neg Systems(x)
\]

또는 단일 라벨 분류라면

\[
\text{ExactlyOne}\{AI(x),Theory(x),Systems(x)\}
\]

를 추가할 수 있습니다. 이런 규칙은 여러 커널 머신의 출력을 함께 묶으므로, 각각의 술어가 독립적으로가 아니라 공동으로 학습됩니다.

### 술어의 변수 개수와 입력

입력 형태는 술어의 arity에 따라 달라집니다.

- 단항 술어:

  \[
  AI(x)
  \quad\Rightarrow\quad
  f_{AI}(\mathbf x)
  \]

- 이항 술어:

  \[
  SameTopic(x,y)
  \quad\Rightarrow\quad
  f_{\text{SameTopic}}(\mathbf x,\mathbf y)
  \]

- 삼항 술어:

  \[
  Relation(x,y,z)
  \quad\Rightarrow\quad
  f_{\text{Relation}}(\mathbf x,\mathbf y,\mathbf z)
  \]

즉, 술어의 변수에 grounding된 객체들의 특징 표현이 커널 머신 입력이 됩니다.

정리하면 다음과 같습니다.

> 각 미지 술어를 커널 함수로 근사하고, grounded 객체의 특징을 입력받아 술어의 퍼지 진릿값을 출력한다. 이 함수는 라벨을 나타내는 supervision 규칙, 객체 사이의 의미적 규칙, 그리고 RKHS 정규화를 동시에 만족하도록 학습된다.

따라서 질문에서 말한 설명이 맞으며, 한 가지 보완점은 **반드시 술어 하나만 학습하는 것은 아니고, 여러 술어의 커널 머신을 논리 규칙으로 연결해 공동 학습할 수 있다**는 것입니다.

---

## 사용자

만약 단항 술어를 학습하는 경우라면, 입력이 변수 하나의 특성값이기 때문에 cite(x,y)와 같은 binary 술어가 포함된 규칙에 대한 만족도는 계산 못하는 것 아니야? 

---

## 어시스턴트

계산할 수 있습니다. 핵심은 **규칙 전체를 단항 커널 머신 하나에 입력하는 것이 아니라, 규칙을 구성하는 각 atom을 각각 평가한 뒤 논리 레이어에서 결합한다**는 점입니다.

예를 들어 학습 대상이 단항 술어

\[
C_c(x)
\]

이고 다음 규칙이 있다고 합시다.

\[
\forall x,y\quad Cite(x,y)\Rightarrow
\bigl(C_c(x)\leftrightarrow C_c(y)\bigr)
\]

여기에는 두 종류의 술어가 있습니다.

- \(C_c(x)\): 학습해야 하는 단항 query predicate
- \(Cite(x,y)\): 인용 그래프에서 이미 주어진 이항 evidence predicate

### 하나의 grounding 계산

변수들을

\[
x=p_i,\qquad y=p_j
\]

로 grounding하면 규칙은

\[
Cite(p_i,p_j)\Rightarrow
\bigl(C_c(p_i)\leftrightarrow C_c(p_j)\bigr)
\]

가 됩니다.

각 atom은 별도로 계산됩니다.

\[
C_c(p_i)=\sigma(f_c(\mathbf x_i))
\]

\[
C_c(p_j)=\sigma(f_c(\mathbf x_j))
\]

그리고 인용 관계는 데이터에서 조회합니다.

\[
Cite(p_i,p_j)=A_{ij}
\]

여기서 \(A\)는 인용 그래프의 adjacency matrix입니다.

따라서 논리 레이어가 받는 값은

\[
A_{ij},\quad
\sigma(f_c(\mathbf x_i)),\quad
\sigma(f_c(\mathbf x_j))
\]

세 개입니다.

```text
논문 pi의 특징 xi ─→ 단항 커널 머신 fc ─→ Cc(pi)
                                              │
인용 그래프 Aij ───────────────────────── Cite(pi,pj) ─→ 퍼지 규칙 만족도
                                              │
논문 pj의 특징 xj ─→ 단항 커널 머신 fc ─→ Cc(pj)
```

단항 커널 머신은 두 번 실행되지만, 같은 파라미터를 공유합니다.

### 수치 예시

분류기의 출력이

\[
C_c(p_i)=0.8,\qquad C_c(p_j)=0.7
\]

이고 실제 인용 관계가

\[
Cite(p_i,p_j)=1
\]

이라고 합시다.

Minimum t-norm을 사용하면 두 category 예측의 일치도는

\[
\max\left(
\min(0.8,0.7),
\min(1-0.8,1-0.7)
\right)
\]

이므로

\[
\max(0.7,0.2)=0.7
\]

입니다. 따라서 이 grounding에 대한 규칙 만족도는 0.7이고, 위반량은

\[
1-0.7=0.3
\]

입니다. 이 위반량의 gradient가 \(f_c(\mathbf x_i)\)와 \(f_c(\mathbf x_j)\) 양쪽으로 전달됩니다.

반대로 예측이

\[
C_c(p_i)=0.9,\qquad C_c(p_j)=0.1
\]

이라면 만족도는

\[
\max(\min(0.9,0.1),\min(0.1,0.9))=0.1
\]

로 낮아집니다. 이 경우 규칙 손실이 두 예측을 더 일관되게 만들도록 커널 머신을 업데이트합니다.

### 중요한 구분

여기서 구분해야 하는 것은 두 가지입니다.

- **커널 머신의 입력 arity**: 근사하는 술어 하나의 변수 개수
- **논리식의 변수 개수**: 규칙 전체에 등장하는 변수 개수

\(C_c\)는 단항 술어이므로 커널 머신은 한 논문의 특징만 받습니다.

\[
f_c(\mathbf x)
\]

하지만 규칙에는 \(C_c(x)\)와 \(C_c(y)\)가 모두 등장할 수 있으므로, 같은 단항 커널 머신을 서로 다른 두 객체에 적용합니다.

\[
f_c(\mathbf x),\qquad f_c(\mathbf y)
\]

그 결과들을 이항 evidence인 \(Cite(x,y)\)와 논리 레이어에서 결합합니다.

### Cite도 미지의 술어라면?

지금까지는 \(Cite(x,y)\)가 데이터에 주어진 evidence라고 가정했습니다. 만약 인용 관계 자체도 예측해야 한다면 별도의 이항 query predicate가 필요합니다.

\[
Cite(x,y)
\approx
\sigma\left(f_{\text{cite}}(\mathbf x,\mathbf y)\right)
\]

그러면 두 함수가 공동 학습됩니다.

\[
\begin{aligned}
C_c(x)&\approx f_c(\mathbf x)
&&\text{단항 함수}\\
Cite(x,y)&\approx f_{\text{cite}}(\mathbf x,\mathbf y)
&&\text{이항 함수}
\end{aligned}
\]

이항 함수의 입력에는 두 논문의 특징을 결합하거나 pair kernel을 사용할 수 있습니다.

\[
K_{\mathrm{pair}}
\bigl((x,y),(x',y')\bigr)
=
K(x,x')K(y,y')
\]

결론적으로, **단항 술어를 학습하더라도 이항 evidence가 포함된 규칙의 만족도를 계산할 수 있습니다.** 단항 커널 머신을 \(x\)와 \(y\)에 각각 적용하고, `Cite(x,y)`는 관계 데이터에서 조회한 뒤, 세 값을 퍼지 논리 레이어에서 조합하면 됩니다.

---

## 사용자

그럼 단항 술어를 학습하더라도 규칙 만족도를 학습하려면 규칙에서 필요로 하는 arity 만큼의 입력을 넣어줘야 한다는 얘기야? 아니면 입력은 하나가 들어가지만 다른 하나는 내부적으로 생성해서 규칙 만족도를 보는거야? 

---

## 어시스턴트

정확히는 **규칙 만족도 손실을 계산하는 학습 단계에서는 규칙에 등장하는 모든 변수의 grounding이 필요합니다.** 하지만 단항 커널 머신 자체는 여전히 객체 하나의 특징만 입력받습니다.

즉, `Cite(x,y)` 규칙이라면 학습 데이터가 \((x,y)\) 쌍을 제공하고, 같은 단항 모델을 \(x\)와 \(y\)에 각각 적용합니다.

```text
규칙 학습 배치: (x, y)

x의 특징 ─→ fθ ─→ C(x) ─┐
                         ├─→ 규칙 만족도
y의 특징 ─→ fθ ─→ C(y) ─┘
         동일한 모델
```

수식으로는

\[
q_x=f_\theta(\mathbf x),\qquad
q_y=f_\theta(\mathbf y)
\]

를 각각 계산한 뒤,

\[
Cite(x,y)\Rightarrow
\bigl(C(x)\leftrightarrow C(y)\bigr)
\]

의 만족도를 계산합니다.

### 다른 입력 \(y\)는 어디서 오는가?

단항 모델이 \(x\)만 보고 \(y\)를 스스로 생성하는 것은 아닙니다. \(y\)는 인용 관계 데이터에서 가져옵니다.

예를 들어 데이터가 다음처럼 구성됩니다.

```text
논문 특징:
p1 → [0.8, 0.1, ...]
p2 → [0.2, 0.9, ...]
p3 → [0.7, 0.3, ...]

인용 edge:
(p1, p2)
(p1, p3)
```

학습할 때 grounding 모듈 또는 데이터 로더가 인용 edge를 읽어 다음 입력 쌍을 만듭니다.

\[
(x,y)=(p_1,p_2),\quad(p_1,p_3)
\]

그다음 각 객체의 특징을 조회합니다.

\[
(\mathbf x_1,\mathbf x_2),\quad
(\mathbf x_1,\mathbf x_3)
\]

따라서 “내부적으로 생성한다”기보다는 **grounding 과정이 관계 데이터에서 필요한 상대 객체를 조회하거나 샘플링한다**고 보는 것이 정확합니다.

### 구현 예시

한 배치에 인용 edge가 다음처럼 들어온다고 하겠습니다.

```python
edges = [
    (paper_1, paper_2),
    (paper_1, paper_3),
]
```

개념적으로 다음처럼 계산합니다.

```python
q1 = model(features[paper_1])
q2 = model(features[paper_2])
q3 = model(features[paper_3])

rule_loss = (
    fuzzy_equivalence(q1, q2)
    + fuzzy_equivalence(q1, q3)
)
```

실제 구현에서는 효율을 위해 고유한 논문들을 한 번에 batch 처리합니다.

```python
papers = [paper_1, paper_2, paper_3]
q = model(features[papers])

loss_12 = rule_violation(q[paper_1], q[paper_2])
loss_13 = rule_violation(q[paper_1], q[paper_3])
```

여기서 모델 입력은 여전히 `[batch_size, feature_dim]` 형태입니다. 모델이 논문 쌍을 하나의 입력으로 처리하는 것은 아닙니다. 모델 출력 두 개를 규칙 loss가 연결할 뿐입니다.

### 일반화하면

규칙이 세 변수를 사용한다면

\[
R(x,y,z)\Rightarrow
\bigl(P(x)\land Q(y)\Rightarrow S(z)\bigr)
\]

규칙 만족도 계산에는 \((x,y,z)\) grounding이 필요합니다. 하지만 각각의 단항 술어 모델은 여전히 하나씩 처리합니다.

\[
P(x)=f_P(\mathbf x),\qquad
Q(y)=f_Q(\mathbf y),\qquad
S(z)=f_S(\mathbf z)
\]

그 결과 세 개를 논리 연산으로 결합합니다.

따라서 두 단계를 구분해야 합니다.

- **단항 술어 예측:** 객체 하나를 모델에 입력
- **관계 규칙 손실 계산:** 규칙의 모든 변수에 해당하는 객체들을 준비하여 각 객체를 모델에 따로 입력

일반적인 단일 논문 추론에서는 \(x\) 하나만 있으면 됩니다. 하지만 `Cite(x,y)` 제약을 학습하거나 테스트 시 collective classification에 적용하려면 관계 데이터가 제공하는 \(x,y\) 쌍이 필요합니다.

---

## 사용자

아 이제 좀 이해가 되는 것 같아. 정리를 좀 해보면, 하나의 모델로 여러 술어를 학습할 수 있다. 각각의 술어에 대해 커널 머신을 생성한다. 술어에 해당하는 변수값을 입력으로 사용한다. 커널 머신이 모든 술어 값을 계산하면, 이 값들이 규칙들을 만족하는지 계산하기 위해, 인용 관계 데이터로부터 필요한 입력 쌍을 생성한다. 이 데이터와 커널 머신, 퍼지논리를 이용해 규칙 만족도를 계산한다. loss를 구해서 학습한다. 이런 단계라고 이해하면 될까?

---

## 어시스턴트

네, 거의 정확합니다. 몇 가지만 다듬으면 다음과 같이 정리할 수 있습니다.

### SBR의 전체 학습 과정

1. **학습할 술어를 정의한다**

   예를 들어 다음 술어들을 예측한다고 합시다.

   \[
   AI(x),\quad Systems(x),\quad Theory(x)
   \]

   이들은 값이 알려지지 않은 query predicate입니다.

   반면

   \[
   Cite(x,y),\quad AuthorOf(a,x)
   \]

   같은 관계가 데이터로 주어져 있다면 evidence predicate이므로 별도의 커널 머신으로 학습하지 않습니다.

2. **각 query predicate에 커널 머신을 둔다**

   원 논문의 구성에서는 일반적으로 각 술어마다 별도의 함수가 있습니다.

   \[
   \begin{aligned}
   AI(x)&\approx \sigma(f_{AI}(\mathbf x))\\
   Systems(x)&\approx \sigma(f_{Systems}(\mathbf x))\\
   Theory(x)&\approx \sigma(f_{Theory}(\mathbf x))
   \end{aligned}
   \]

   따라서 “하나의 모델”은 하나의 커널 머신이라기보다, **여러 커널 머신과 논리 규칙을 포함한 하나의 SBR 시스템**을 뜻합니다.

3. **논리 객체를 실제 객체로 grounding한다**

   규칙이

   \[
   Cite(x,y)\Rightarrow
   \bigl(AI(x)\leftrightarrow AI(y)\bigr)
   \]

   라면 인용 관계 데이터로부터

   \[
   (x,y)=(p_1,p_2),\;(p_1,p_3),\ldots
   \]

   같은 grounding을 얻습니다.

4. **grounded 객체의 특징을 커널 머신에 입력한다**

   객체 식별자 자체가 아니라 그 객체의 특징 표현을 사용합니다.

   \[
   p_1\mapsto \mathbf x_1,\qquad
   p_2\mapsto \mathbf x_2
   \]

   그리고 동일한 단항 커널 머신을 각각 적용합니다.

   \[
   AI(p_1)\approx\sigma(f_{AI}(\mathbf x_1))
   \]

   \[
   AI(p_2)\approx\sigma(f_{AI}(\mathbf x_2))
   \]

   다만 특징이 없는 기호 객체라면 객체별 술어값을 직접 최적화할 수도 있습니다.

5. **evidence와 예측값을 모아 퍼지 논리를 계산한다**

   예를 들어

   \[
   Cite(p_1,p_2)=1
   \]

   이고 커널 머신의 출력이

   \[
   AI(p_1)=0.8,\qquad AI(p_2)=0.7
   \]

   이라면 다음 grounded rule의 만족도를 계산합니다.

   \[
   Cite(p_1,p_2)\Rightarrow
   \bigl(AI(p_1)\leftrightarrow AI(p_2)\bigr)
   \]

   여기서 `Cite` 값은 관계 데이터에서, `AI` 값들은 커널 머신에서 옵니다.

6. **모든 관련 grounding을 집계한다**

   인용 edge 집합을 \(E\)라 하면 대략

   \[
   \Phi_{\text{cite}}
   =
   \frac{1}{|E|}
   \sum_{(x,y)\in E}
   \operatorname{Eq}_{\text{fuzzy}}
   \bigl(AI(x),AI(y)\bigr)
   \]

   처럼 전체 규칙 만족도를 계산합니다.

7. **규칙 위반량과 supervision을 loss로 만든다**

   최종 목적함수는 다음과 같은 구조입니다.

   \[
   L
   =
   \underbrace{\sum_k\|f_k\|_{\mathcal H_k}^2}_{\text{커널 머신 정규화}}
   +
   \underbrace{\sum_h\lambda_h(1-\Phi_h)}_{\text{논리 규칙 위반}}
   \]

   여기에는 다음이 모두 포함될 수 있습니다.

   - 라벨을 맞추기 위한 supervision 규칙
   - 인용 관계의 일관성 규칙
   - category 간 배타성 규칙
   - category 계층 규칙
   - 클래스 분포에 관한 규칙

8. **loss를 역전파해 커널 머신들을 공동 학습한다**

   하나의 규칙이 여러 술어를 포함하면 해당 규칙의 gradient가 관련된 모든 커널 머신으로 전달됩니다.

   예를 들어

   \[
   AI(x)\Rightarrow ResearchArea(x)
   \]

   라는 규칙의 손실은 \(f_{AI}\)와 \(f_{ResearchArea}\)를 동시에 업데이트할 수 있습니다.

### 한 가지 순서상의 보완

“모든 술어값을 먼저 계산한 다음 입력 쌍을 생성한다”기보다는, 일반적으로는 다음 순서가 더 효율적입니다.

```text
논리 규칙 확인
    ↓
관계 데이터에서 필요한 grounding 생성
    ↓
그 grounding에 필요한 객체와 술어 확인
    ↓
필요한 커널 머신 출력 계산
    ↓
퍼지 규칙 만족도 계산
    ↓
전체 loss 계산 및 공동 학습
```

물론 전체 데이터가 작다면 모든 객체의 모든 술어값을 먼저 계산한 다음 필요한 값을 꺼내 사용해도 수학적으로는 같습니다.

따라서 정리한 내용은 맞습니다. 가장 핵심적인 표현은 다음과 같습니다.

> SBR은 각 미지 술어를 커널 머신으로 근사하고, 관계 데이터로부터 생성된 grounding 위에서 이 예측값과 evidence를 퍼지 논리로 결합한다. 지도 라벨과 의미적 규칙의 위반량을 하나의 loss로 만들고, 이를 통해 여러 술어의 커널 머신을 공동 학습한다.

---

## 사용자

이 논문은 술어에 대한 함수로 커널 머신을 사용하고 representer theorem을 사용했는데, 대신 신경망으로 술어 함수를 근사할수도 있지 않아? 커널 머신을 학습하거나 예측하려면 항상 전체 값이 필요한데, 신경망은 더 효율적으로 예측과 학습을 할 수 있지 않을까? 대신 최적 함수를 반드시 찾는다는 보장은 없어지겠지만.

---

## 어시스턴트

맞습니다. 술어 함수를 커널 머신 대신 신경망으로 근사할 수 있습니다. 퍼지 논리 부분은 술어값이 \([0,1]\) 범위에서 미분 가능하게 나오기만 하면 작동하므로, 반드시 커널 머신이어야 하는 것은 아닙니다.

\[
p_k(x)\approx \sigma(f_k(\mathbf x))
\]

에서 \(f_k\)를 커널 머신 대신 신경망 \(g_{\theta_k}\)로 바꾸면 됩니다.

\[
p_k(x)\approx \sigma(g_{\theta_k}(\mathbf x))
\]

### 신경망으로 바꾼 구조

여러 category 술어가 있다면 각각 별도 신경망을 둘 수도 있지만, 보통은 공통 표현을 공유하는 편이 효율적입니다.

\[
\mathbf h(x)=Encoder_\theta(\mathbf x)
\]

\[
p_k(x)=\sigma(Head_{\phi_k}(\mathbf h(x)))
\]

```text
논문 특징 x
    ↓
공유 신경망 encoder
    ↓
표현 h(x)
    ├─→ AI head      → AI(x)
    ├─→ Theory head  → Theory(x)
    └─→ Systems head → Systems(x)
                          ↓
                 퍼지 논리 규칙
                          ↓
                    semantic loss
```

최종 손실은 다음처럼 구성할 수 있습니다.

\[
L(\theta,\phi)
=
L_{\text{sup}}
+
\sum_h\lambda_h\left(1-\Phi_h(p_\theta)\right)
+
\lambda_{\text{reg}}\|\theta\|^2
\]

예를 들어 edge 미니배치를 샘플링해

\[
E_B=\{(x_1,y_1),\ldots,(x_B,y_B)\}
\]

각 객체의 술어값을 신경망으로 계산한 뒤,

\[
L_{\text{cite}}
=
\frac1B\sum_{(x,y)\in E_B}
\left[
1-\operatorname{Eq}_{\text{fuzzy}}
\bigl(p_k(x),p_k(y)\bigr)
\right]
\]

를 지도학습 손실에 더하면 됩니다.

### 신경망이 더 효율적일 수 있는 이유

커널 머신은 representer theorem에 의해

\[
f(x)=\sum_{i=1}^{N}w_iK(x_i,x)
\]

형태를 갖습니다. 따라서 새로운 객체 하나를 예측할 때도 일반적으로 여러 학습 객체와의 커널값을 계산해야 합니다. 계수가 희소하지 않다면 예측 비용이 데이터 수 \(N\)에 따라 증가합니다.

또한 학습에서는 \(N\times N\) Gram 행렬이 필요할 수 있어 다음 문제가 생깁니다.

- Gram 행렬 메모리: 대략 \(O(N^2)\)
- 커널값 계산 비용 증가
- 데이터가 추가되면 커널 전개도 커짐
- 이미지나 텍스트의 표현 자체를 학습하기 어려움

반면 신경망은 학습이 끝나면 고정된 수의 파라미터만 사용합니다.

\[
p(x)=g_\theta(x)
\]

예측 비용은 주로 네트워크 크기에 의존하고 학습 데이터 수에는 직접 의존하지 않습니다. 또한 다음이 가능합니다.

- 객체와 관계의 미니배치 학습
- 이미지·텍스트의 표현까지 end-to-end 학습
- 여러 술어가 encoder를 공유
- 새로운 객체에 대한 빠른 inductive prediction
- 매우 큰 관계 그래프에서 edge/subgraph sampling

다만 커널 머신도 항상 문자 그대로 “전체 값을 한 번에 입력”해야 하는 것은 아닙니다. 희소 커널, Nyström 근사, random feature, online kernel 방법 등을 사용할 수 있습니다. 그래도 기본적인 SBR 표현은 전체 grounding에 대한 Gram 행렬에 상당히 의존합니다.

### Representer theorem에 관한 중요한 보정

신경망을 사용하면 “최적 함수를 반드시 찾는 보장”이 사라진다는 말은 절반만 맞습니다.

Representer theorem이 보장하는 것은 다음입니다.

> 목적함수의 최적해가 존재한다면, 그 최적해를 학습 데이터에 놓인 커널 함수들의 유한한 선형결합으로 표현할 수 있다.

즉,

\[
f^*(x)=\sum_iw_i^*K(x_i,x)
\]

형태만 탐색해도 충분하다는 **표현 형식의 보장**입니다.

하지만 representer theorem 자체는 다음을 보장하지 않습니다.

- 목적함수가 convex라는 것
- 최적해가 유일하다는 것
- gradient descent가 전역 최적해를 찾는다는 것
- 실제 최적화 알고리즘이 수렴한다는 것

특히 이 논문의 SBR 목적함수는 퍼지 논리 규칙 때문에 일반적으로 비볼록입니다. 논문도 복잡한 논리식의 참인 assignment 수에 대응하여 많은 지역 최솟값이 생길 수 있다고 설명합니다. 그래서 쉬운 규칙부터 추가하는 stage-based learning 휴리스틱을 사용합니다.

따라서 커널 머신을 썼더라도 이미 “전역 최적해를 반드시 찾는다”는 보장은 없습니다. 신경망으로 바꾸면 추가로 잃는 것은 주로 다음입니다.

- 최적 함수의 유한 커널 전개 보장
- RKHS norm에 기반한 명시적 함수 복잡도 해석
- 커널이 정한 함수 공간에 관한 이론적 분석 가능성

반면 얻는 것은 표현 학습과 확장성입니다.

### 신경망으로 바꿔도 남는 문제

신경망은 술어 평가를 효율화하지만, 논리 grounding 문제까지 자동으로 해결하지는 않습니다.

예를 들어

\[
\forall x,y,z\quad R(x,y,z)\Rightarrow F(x,y,z)
\]

에는 잠재적으로 \(O(N^3)\)개의 grounding이 존재합니다. 신경망이 각 술어값을 빠르게 계산하더라도 모든 조합을 열거하면 여전히 비쌉니다. 따라서 다음이 별도로 필요합니다.

- evidence에 의해 관련된 grounding만 생성
- 자명하게 참인 grounding 제거
- edge 또는 subgraph sampling
- hard-negative grounding 선택
- quantifier의 미니배치 근사

특히 \(\forall\)를 평균으로 처리하면 미니배치 추정이 비교적 쉽지만, \(\exists\)를 `max`로 처리하거나 “최소 \(n\)개가 참” 같은 전역 제약을 사용하면 작은 미니배치가 전체 논리 의미를 제대로 반영하지 못할 수 있습니다.

### 결론

커널 머신을 신경망으로 바꾸는 것은 자연스럽고 실용적인 확장입니다.

| 구성요소 | 원 논문 | 신경망 버전 |
|---|---|---|
| 술어 함수 | 커널 머신 \(f_k\) | 신경망 \(g_{\theta_k}\) |
| 술어 출력 | sigmoid를 거친 진릿값 | 동일 |
| 논리 규칙 | t-norm 퍼지 논리 | 동일 |
| 학습 신호 | 규칙 위반량 | 동일 |
| 함수 복잡도 | RKHS norm | weight decay 등 |
| 학습 방식 | Gram 행렬 중심 | 미니배치 SGD |
| 예측 비용 | 학습점 수에 의존 가능 | 네트워크 크기에 의존 |
| 표현 학습 | 제한적 | end-to-end 가능 |

즉, SBR의 본질은 커널 머신 자체가 아니라 **학습 가능한 술어 함수의 출력에 퍼지 논리 제약을 걸고, 그 위반량을 역전파하는 것**입니다. 커널 머신은 이 논문이 선택한 함수 근사 방식이며, 이를 신경망으로 대체하면 현대적인 미분 가능 신경-기호 학습 구조가 됩니다.

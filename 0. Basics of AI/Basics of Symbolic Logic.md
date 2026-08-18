# 명제논리에서 ILP까지 - 논리와 학습은 어떻게 연결되는가

컴퓨터가 지식을 표현하고, 추론하고, 새로운 규칙까지 학습하게 하려면 무엇이 필요할까? 이 질문을 이해하기 위해서는 명제논리, 일차논리(FOL), grounding, SAT, 그리고 Inductive Logic Programming(ILP)의 관계를 차례대로 살펴볼 필요가 있다.

이 개념들은 서로 독립적으로 보이지만 실제로는 다음과 같은 흐름으로 연결된다.

$\text{명제논리} \rightarrow \text{FOL} \rightarrow \text{Grounding} \rightarrow \text{SAT} \rightarrow \text{ILP에서의 규칙 탐색}$

---

## 1. 명제논리: 문장 전체를 하나의 단위로 다루기

명제논리는 참 또는 거짓을 명확히 판단할 수 있는 문장을 다룬다.

예를 들어 다음 두 문장을 생각해보자.

- $P$: 비가 온다.
- $Q$: 길이 젖는다.

이들을 논리 연산자로 연결하면 다음과 같은 규칙을 표현할 수 있다.

$P\rightarrow Q$

이는 “비가 오면 길이 젖는다”라는 뜻이다.

명제논리에서 자주 사용하는 연산자는 다음과 같다.

| 기호                     | 의미                    |
| ---------------------- | --------------------- |
| $\neg P$             | $P$가 아니다            |
| $P\land Q$           | $P$이고 $Q$이다       |
| $P\lor Q$            | $P$이거나 $Q$이다      |
| $P\rightarrow Q$     | $P$이면 $Q$이다       |
| $P\leftrightarrow Q$ | $P$와 $Q$의 진릿값이 같다 |

명제논리는 단순하고 계산하기 쉽지만 문장의 내부 구조를 표현하지 못한다.

예를 들어

$P=\text{소크라테스는 인간이다}$

라고 표현하면, 명제논리는 `소크라테스`라는 개체와 `인간이다`라는 속성을 구분하지 않는다. 문장 전체를 하나의 기호 $P$로 취급할 뿐이다.

---

## 2. FOL: 개체와 관계까지 표현하기

FOL(First-Order Logic, 일차논리)은 명제 내부를 개체, 속성, 관계로 나누어 표현한다.

“소크라테스는 인간이다”는 다음과 같이 쓸 수 있다.

$Human(socrates)$

여기서 `socrates`는 개체이고, $Human(x)$는 $x$가 인간이라는 속성을 나타내는 술어다.

FOL은 변수와 양화사도 사용할 수 있다.

“모든 인간은 죽는다”는 다음과 같이 표현된다.

$\forall x\,(Human(x)\rightarrow Mortal(x))$

“어떤 인간은 철학자다”는 다음과 같다.

$\exists x\,(Human(x)\land Philosopher(x))$

또한 여러 개체 사이의 관계도 표현할 수 있다.

$Loves(alice,bob)$

즉, 명제논리가 문장과 문장 사이의 관계를 다룬다면 FOL은 문장 내부의 개체와 관계까지 다룬다.

|구분|명제논리|FOL|
|---|---|---|
|기본 단위|문장 전체|개체, 속성, 관계|
|변수|없음|사용 가능|
|양화사|없음|$\forall,\exists$|
|표현력|제한적|높음|
|자동 추론|비교적 단순|일반적으로 복잡함|

---

## 3. Grounding: 변수를 구체적인 개체로 바꾸기

Grounding은 FOL이나 논리 프로그램의 변수를 구체적인 개체를 나타내는 항으로 치환하는 과정이다.

예를 들어 다음 규칙이 있다고 하자.

$Human(x)\rightarrow Mortal(x)$

변수 $x$를 `socrates`로 치환하면 다음과 같은 ground formula가 된다.

$Human(socrates)\rightarrow Mortal(socrates)$

도메인이 `alice`와 `bob` 두 사람으로만 구성되어 있다면

$\forall x\,(Human(x)\rightarrow Mortal(x))$

는 다음과 같이 전개할 수 있다.

$(Human(alice)\rightarrow Mortal(alice)) \land (Human(bob)\rightarrow Mortal(bob))$

각각의 ground atom을 하나의 명제변수로 취급하면 이 식을 명제논리 문제로 다룰 수 있다.

양화사는 유한한 도메인에서 다음과 같이 전개된다.

$\forall x\,P(x) \quad\longrightarrow\quad P(a)\land P(b)\land\cdots$$\exists x\,P(x) \quad\longrightarrow\quad P(a)\lor P(b)\lor\cdots$

다만 도메인이 무한하거나 함수 기호를 통해 무한히 많은 항이 생성된다면 grounding 결과 역시 무한해질 수 있다. 따라서 모든 FOL 문제를 항상 유한한 명제논리 문제로 변환할 수 있는 것은 아니다.

---

## 4. SAT: 논리식을 참으로 만드는 할당 찾기

SAT(Boolean Satisfiability Problem)는 주어진 명제논리식을 참으로 만드는 변수 할당이 존재하는지를 묻는 문제다.

다음 식을 살펴보자.

$(P\lor Q)\land\neg P$

여기서

$P=False,\qquad Q=True$

로 지정하면 전체 식이 참이 된다. 따라서 이 식은 만족 가능하며 `SAT`이라고 한다.

반면 다음 식은 어떤 값을 대입해도 참이 될 수 없다.

$P\land\neg P$

따라서 이 식은 `UNSAT`, 즉 만족 불가능하다.

SAT 문제는 일반적으로 CNF(논리곱 표준형)로 표현된다.

$(P\lor\neg Q) \land (Q\lor R) \land (\neg P\lor\neg R)$

CNF는 리터럴들을 OR로 연결한 절을 여러 개의 AND로 연결한 형태다. 전체 식이 참이 되려면 모든 절에서 적어도 하나의 리터럴이 참이어야 한다.

SAT는 논리적 추론에도 활용할 수 있다. 지식 $K$로부터 결론 $Q$가 반드시 따라오는지 알고 싶다면 다음 식을 검사한다.

$K\land\neg Q$

- `SAT`이면 $Q$가 거짓인 반례가 존재한다.
- `UNSAT`이면 반례가 없으므로 $K\models Q$이다.

즉, 논리적 증명을 “반례를 찾을 수 있는가?”라는 만족 가능성 문제로 바꿀 수 있다.

---

## 5. ILP: 데이터에서 논리 규칙 학습하기

Inductive Logic Programming, 즉 ILP는 관찰된 예제와 배경지식으로부터 논리 규칙을 학습하는 방법이다.

ILP 문제는 보통 다음 세 요소로 구성된다.

- $B$: 배경지식
- $E$: 학습 예제
- $H$: 학습할 가설

목표는 대체로 다음 조건을 만족하는 가설 $H$를 찾는 것이다.

$B\cup H\models E^+$$B\cup H\not\models E^-$

여기서 $E^+$는 설명해야 하는 양성 예제이고, $E^-$는 잘못 유도해서는 안 되는 음성 예제다.

예를 들어 다음과 같은 배경지식이 있다고 하자.

```
parent(alice, bob).
parent(bob, charlie).
```

그리고 다음 양성 예제가 주어진다.

```
grandparent(alice, charlie).
```

ILP는 이를 설명하는 규칙을 학습할 수 있다.

```
grandparent(X, Z) :-
    parent(X, Y),
    parent(Y, Z).
```

논리식으로 쓰면 다음과 같다.

$parent(X,Y)\land parent(Y,Z) \rightarrow grandparent(X,Z)$

일반적인 머신러닝이 수치 파라미터를 학습한다면, ILP는 사람이 읽고 이해할 수 있는 논리 규칙을 학습한다는 특징이 있다.

---

## 6. Horn Clause: FOL과 논리 프로그램을 잇는 규칙 형식

Horn clause는 ILP와 논리 프로그래밍에서 가장 널리 사용되는 논리식의 형태다. 명제논리의 CNF 관점에서는 **긍정 리터럴을 최대 하나만 포함하는 절**을 뜻한다.

예를 들어 다음은 Horn clause다.

$\neg parent(X,Y)\lor \neg parent(Y,Z)\lor grandparent(X,Z)$

긍정 리터럴은 \(grandparent(X,Z)\) 하나뿐이다. 이 식은 조건문을 이용해 다음과 같이 다시 쓸 수 있다.

$parent(X,Y)\land parent(Y,Z) \rightarrow grandparent(X,Z)$

논리 프로그래밍에서는 일반적으로 다음과 같이 표현한다.

```
grandparent(X, Z) :-
    parent(X, Y),
    parent(Y, Z).
```

여기서 각 부분의 역할은 다음과 같다.

- `grandparent(X, Z)`: 규칙의 머리(head), 즉 결론
- `parent(X, Y), parent(Y, Z)`: 규칙의 몸체(body), 즉 전제
- `:-`: 오른쪽 조건이 성립하면 왼쪽 결론이 성립한다는 의미

### 6.1 Horn clause의 주요 형태

Horn clause는 긍정 리터럴의 수에 따라 몇 가지 형태로 구분할 수 있다.

#### 사실

조건 없이 하나의 긍정 리터럴만 있는 경우다.

$parent(alice,bob)$

```
parent(alice, bob).
```

이는 주어진 사실을 직접 나타낸다.

#### 규칙

긍정 리터럴 하나와 여러 개의 부정 리터럴을 포함한다.

$\neg parent(X,Y)\lor \neg parent(Y,Z)\lor grandparent(X,Z)$

이를 논리적 함의로 바꾸면 다음과 같다.

$parent(X,Y)\land parent(Y,Z) \rightarrow grandparent(X,Z)$

#### 목표 또는 제약

긍정 리터럴이 하나도 없는 Horn clause다.

$\neg parent(X,Y)\lor\neg parent(Y,Z)$

이는 다음 조건이 동시에 성립해서는 안 된다는 제약으로 해석할 수 있다.

$\neg\bigl(parent(X,Y)\land parent(Y,Z)\bigr)$

엄밀하게는 긍정 리터럴이 정확히 하나인 절을 **definite clause**라고 한다. 따라서 모든 definite clause는 Horn clause지만, 긍정 리터럴이 없는 제약도 Horn clause에 포함된다.

### 6.2 FOL과의 관계

변수가 포함된 논리 프로그램의 규칙은 일반적으로 변수들이 전칭 양화되어 있다고 해석한다.

```
grandparent(X, Z) :-
    parent(X, Y),
    parent(Y, Z).
```

이를 FOL로 표현하면 다음과 같다.

$\forall X\,\forall Y\,\forall Z\, \Bigl( parent(X,Y)\land parent(Y,Z) \rightarrow grandparent(X,Z) \Bigr)$

따라서 Horn clause는 FOL과 별개의 논리라기보다, FOL에서 계산하기 쉬운 형태를 선택한 **제한된 부분 언어**라고 볼 수 있다.

$\text{FOL} \supset \text{Horn clause} \supset \text{definite clause}$

일반적인 FOL보다 표현력은 제한되지만 추론 절차를 효율적으로 설계할 수 있다는 장점이 있다. Prolog와 Datalog 같은 논리 프로그래밍 언어가 Horn clause를 중심으로 구성되는 이유도 여기에 있다.

### 6.3 ILP가 Horn clause를 사용하는 이유

ILP가 일반적인 FOL의 모든 문장을 후보 가설로 허용한다면 탐색 공간이 지나치게 커진다. Horn clause로 가설의 형태를 제한하면 학습할 규칙을 다음처럼 일정한 구조로 표현할 수 있다.

$Head\leftarrow Body_1\land\cdots\land Body_n$

예를 들어 ILP는 다음 후보들을 일반화하거나 특수화하면서 적절한 규칙을 탐색할 수 있다.

```
grandparent(X, Z) :-
    parent(X, Y).
```

```
grandparent(X, Z) :-
    parent(X, Y),
    parent(Y, Z).
```

첫 번째 규칙이 지나치게 일반적이어서 음성 예제까지 설명한다면, 두 번째처럼 몸체에 조건을 추가해 규칙을 특수화할 수 있다.

Horn clause는 ILP에 세 가지 중요한 이점을 제공한다.

- 학습된 규칙을 사람이 쉽게 이해할 수 있다.
- 일반화와 특수화 연산을 체계적으로 정의할 수 있다.
- Prolog식 추론이나 grounding, SAT 계열 기법과 연결하기 쉽다.

### 6.4 Horn clause와 SAT의 관계

Horn clause의 변수들을 구체적인 상수로 grounding하면 명제논리의 Horn clause를 얻을 수 있다.

예를 들어

$parent(X,Y)\land parent(Y,Z)\rightarrow grandparent(X,Z)$

에 \(X=alice\), \(Y=bob\), \(Z=charlie\)를 대입하면 다음과 같다.

$parent(alice,bob)\land parent(bob,charlie) \rightarrow grandparent(alice,charlie)$

조건문을 제거하면 CNF의 한 절이 된다.

$\neg parent(alice,bob) \lor \neg parent(bob,charlie) \lor grandparent(alice,charlie)$

각 ground atom을 명제변수로 보면 SAT 기법으로 처리할 수 있다. 특히 모든 절이 Horn clause인 만족 가능성 문제를 **Horn-SAT**이라고 한다. 일반 SAT가 NP-완전인 것과 달리 Horn-SAT는 단위 전파나 전방 추론을 이용해 다항시간에 해결할 수 있다.

이 때문에 Horn clause는 다음 개념들을 연결하는 핵심적인 중간 지점이 된다.

$\boxed{ \text{FOL 표현} \rightarrow \text{Horn 규칙} \rightarrow \text{Grounding} \rightarrow \text{Horn-SAT 또는 SAT 추론} }$

ILP는 이 구조 위에서 예제를 설명할 Horn 규칙을 탐색하고, 논리 프로그래밍이나 SAT 기반 방법은 선택된 규칙의 결과를 계산하고 검증한다.

---

## 7. ILP와 FOL의 관계

FOL은 지식을 표현하고 추론하기 위한 논리 체계다. 반면 ILP는 그 논리 언어로 표현된 규칙을 데이터에서 찾아내는 학습 방법이다.

$\boxed{ \text{FOL은 표현과 추론의 언어이고, ILP는 그 언어로 된 규칙을 학습한다.} }$

다만 대부분의 ILP 시스템은 일반적인 FOL 전체를 사용하지 않는다. 탐색과 추론이 지나치게 어려워질 수 있기 때문이다.

주로 다음과 같은 Horn clause를 사용한다.

$A\leftarrow B_1\land B_2\land\cdots\land B_n$

이는 FOL에서 다음 형태에 해당한다.

$\forall\vec{x}\, (B_1\land B_2\land\cdots\land B_n\rightarrow A)$

Prolog, Datalog 등의 논리 프로그래밍 언어는 이러한 제한된 논리 형태를 중심으로 구성된다.

---

## 8. ILP와 SAT는 어떻게 연결되는가?

ILP와 SAT는 같은 문제가 아니다.

- ILP는 어떤 논리 규칙을 학습할 것인지 묻는다.
- SAT는 어떤 참·거짓 선택이 주어진 제약을 만족하는지 묻는다.

하지만 ILP의 규칙 탐색을 SAT 문제로 부호화할 수 있다.

후보 규칙 $r_1,r_2,\ldots$가 있다면 각 규칙의 선택 여부를 명제변수로 만든다.

$S_1=\text{규칙 }r_1\text{을 선택한다}$$S_2=\text{규칙 }r_2\text{를 선택한다}$

그런 다음 다음 조건을 명제논리 제약으로 표현한다.

- 모든 양성 예제를 설명해야 한다.
- 음성 예제를 유도해서는 안 된다.
- 규칙은 허용된 문법을 따라야 한다.
- 가능하면 적은 수의 단순한 규칙을 선택해야 한다.

SAT solver가 이 제약을 만족하는 $S_1,S_2,\ldots$의 값을 찾으면, 선택된 규칙들을 모아 가설 $H$를 구성할 수 있다.

$\text{ILP 규칙 탐색} \rightarrow \text{Grounding과 명제화} \rightarrow \text{SAT 제약 생성} \rightarrow \text{가설 선택}$

잡음이 있는 데이터에서는 모든 조건을 완벽하게 만족하기 어려울 수 있다. 이때는 일부 조건 위반을 허용하면서 가장 좋은 해를 찾는 MaxSAT를 사용할 수 있다.

예를 들어 다음 두 요소를 함께 최소화할 수 있다.

$\text{예제 오류 수} + \lambda\cdot\text{규칙 복잡도}$

이는 데이터는 잘 설명하면서도 지나치게 복잡하지 않은 규칙을 선택하려는 것이다.

---

## 9. 전체 관계 정리

지금까지 살펴본 개념은 다음과 같이 연결된다.

|개념|역할|
|---|---|
|명제논리|참·거짓 명제와 논리적 연결을 표현한다|
|FOL|개체, 변수, 속성, 관계, 양화사를 표현한다|
|Grounding|변수 있는 논리식을 구체적인 개체별 식으로 만든다|
|SAT|명제논리 제약을 만족하는 할당을 찾는다|
|ILP|배경지식과 예제로부터 논리 규칙을 학습한다|

FOL은 풍부한 표현력을 제공하지만 추론이 어렵다. Grounding은 제한된 FOL 문제를 명제논리 문제로 바꾸는 연결 고리다. SAT solver는 그렇게 만들어진 명제논리 제약을 효율적으로 해결한다. ILP는 이 기술들을 활용해 어떤 규칙이 데이터를 가장 잘 설명하는지 탐색할 수 있다.

결국 이들의 관계는 다음 한 문장으로 정리할 수 있다.

> FOL로 지식을 표현하고, grounding을 통해 명제화하며, SAT로 제약을 해결하고, ILP로 데이터에서 새로운 논리 규칙을 학습한다.
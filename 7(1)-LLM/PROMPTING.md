# 1. Results

| Prompting Type         | 0-shot | 3-shot  | 5-shot |
| ---------------------- | ------ | ------- | ------ |
| Direct Prompting       | 22%    | 18%     | 18%    |
| Chain-of-Thought (CoT) | 74%    | 72%     | 74%    |
| My Prompting           | 80%    | **84%** | 78%    |

# 2. CoT Prompting이 Direct Prompting보다 좋은 이유

Direct Prompting은 문제에 대한 정답만을 생성하도록 유도하기 때문에, 모델이 문제를 해결하는 과정 자체를 고려하지 않습니다.  
이로 인해 수학처럼 단계적인 추론이 필요한 문제에서는 계산 실수나 논리 오류가 발생하기 쉽습니다.

반면 Chain-of-Thought (CoT) Prompting은 모델이 문제 해결 과정을 단계적으로 서술하도록 유도함으로써, 모델의 reasoning 능력을 적극적으로 활용할 수 있게 합니다.

# 3. CoT Prompting이 Direct Prompting보다 좋은 이유

My Prompting은 CoT의 장점인 단계별 추론을 유지하면서, 다음과 같은 전략을 추가해 성능을 더 향상시켰습니다:

1. **역할 지시(Persona 설정)**:  
   "You are a brilliant and careful math tutor."  
   -> LLM에게 신중하고 체계적인 사고를 유도하며, CoT보다 더 명확한 행동 프레임을 제공합니다.

2. **Self-Consistency 적용**:  
   동일한 질문에 대해 모델이 5 번 응답한 결과 중 가장 빈도 높은 응답을 최종 정답으로 선택함으로써,우연한 오류나 편차를 줄이고 일관된 정답을 선택할 수 있도록 했습니다.

3. **Temperature 조정 (0.7)**:  
   약간의 다양성을 허용함으로써 reasoning 방식의 다양성을 확보하고, Self-Consistency와 함께 활용하여 성능을 보완했습니다.

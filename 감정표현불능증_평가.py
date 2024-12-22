import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoTokenizer, ElectraModel
from kiwipiepy import Kiwi

# 사전에 정의된 44개의 감정 라벨 및 상황별 오답 감정
LABELS = ['불평/불만', '환영/호의', '감동/감탄', '지긋지긋', '고마움', '슬픔', '화남/분노', '존경', '기대감', '우쭐댐/무시함', '안타까움/실망', '비장함', '의심/불신', '뿌듯함', '편안/쾌적', '신기함/관심', '아껴주는', '부끄러움', '공포/무서움', '절망', '한심함', '역겨움/징그러움', '짜증', '어이없음', '없음', '패배/자기혐오', '귀찮음', '힘듦/지침', '즐거움/신남', '깨달음', '죄책감', '증오/혐오', '흐뭇함(귀여움/예쁨)', '당황/난처', '경악', '부담/안_내킴', '서러움', '재미없음', '불쌍함/연민', '놀람', '행복', '불안/걱정', '기쁨', '안심/신뢰']
forbidden_emotions = {
    'situ_1-1': ['환영/호의', '감동/감탄', '고마움', '우쭐댐/무시함', '뿌듯함', '편안/쾌적', '아껴주는', '즐거움/신남', '흐뭇함(귀여움/예쁨)', '기쁨', '안심/신뢰', '행복', '존경'],
    'situ_1-2': ['불평/불만', '슬픔', '화남/분노', '안타까움/실망', '의심/불신', '공포/무서움', '절망', '역겨움/징그러움', '짜증', '패배/자기혐오', '증오/혐오', '당황/난처', '경악', '부담/안_내킴', '서러움', '재미없음'],
    'situ_1-3': ['불평/불만', '지긋지긋', '슬픔', '안타까움/실망', '절망', '한심함', '역겨움/징그러움', '짜증', '어이없음', '패배/자기혐오', '죄책감', '증오/혐오', '당황/난처', '서러움', '불안/걱정', '불쌍함/연민'],
    'situ_2-1': ['비장함', '안타까움/실망', '부끄러움', '환영/호의', '감동/감탄', '고마움', '뿌듯함', '편안/쾌적', '아껴주는', '즐거움/신남', '흐뭇함(귀여움/예쁨)', '기쁨', '안심/신뢰', '행복', '존경', '기대감'],
    'situ_2-2': ['환영/호의', '감동/감탄', '고마움', '우쭐댐/무시함', '뿌듯함', '편안/쾌적', '아껴주는', '흐뭇함(귀여움/예쁨)', '기쁨', '안심/신뢰', '행복', '존경', '즐거움/신남'],
    'situ_2-3': ['환영/호의', '고마움', '편안/쾌적', '아껴주는', '흐뭇함(귀여움/예쁨)', '기쁨', '행복', '공포/무서움', '즐거움/신남', '존경', '감동/감탄', '안심/신뢰', '기대감', '뿌듯함'],
    'situ_3-1': ['불평/불만', '지긋지긋', '절망', '짜증', '어이없음', '힘듦/지침', '당황/난처', '불안/걱정', '의심/불신', '슬픔', '경악', '서러움', '안타까움/실망', '불쌍함/연민', '화남/분노', '부담/안_내킴', '역겨움/징그러움', '패배/자기혐오', '증오/혐오', '귀찮음', '공포/무서움', '재미없음', '죄책감', '재미없음', '한심함'],
    'situ_3-2': ['불평/불만', '지긋지긋', '절망', '짜증', '어이없음', '힘듦/지침', '당황/난처', '불안/걱정', '의심/불신', '슬픔', '경악', '서러움', '안타까움/실망', '불쌍함/연민', '화남/분노', '부담/안_내킴', '역겨움/징그러움', '패배/자기혐오', '증오/혐오', '귀찮음', '우쭐댐/무시함', '공포/무서움', '죄책감', '재미없음', '한심함'],
    'situ_3-3': ['환영/호의', '고마움', '우쭐댐/무시함', '뿌듯함', '편안/쾌적', '아껴주는', '흐뭇함(귀여움/예쁨)', '기쁨', '행복', '공포/무서움', '역겨움/징그러움', '즐거움/신남', '지긋지긋', '짜증', '어이없음', '화남/분노', '경악', '귀찮음', '증오/혐오']
}

# GPU/CPU 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class KOTEtagger(pl.LightningModule):
    """
    사전 학습된 Electra 모델("beomi/KcELECTRA-base")을 사용하여 44개 감정 라벨에 대한 다중 라벨 분류를 진행하는 모델 클래스
    """
    def __init__(self):
        super().__init__()
        # 1) ElectraModel 로드
        self.electra = ElectraModel.from_pretrained("beomi/KcELECTRA-base", revision='v2021').to(device)
        # 2) Electra 모델의 토크나이저
        self.tokenizer = AutoTokenizer.from_pretrained("beomi/KcELECTRA-base", revision='v2021')
        # 3) 출력(hidden size) -> 44개 감정 라벨에 대한 classifier 레이어
        self.classifier = nn.Linear(self.electra.config.hidden_size, 44).to(device)

    def forward(self, text: str):
        """
        입력 텍스트를 토크나이징하고 Electra 모델을 통과시킨 뒤, Linear Layer로 감정 분류 점수를 출력 (시그모이드 적용)
        """
        # 텍스트 토큰화
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=512,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors='pt',
        ).to(device)

        # Electra 모델 forward
        output = self.electra(
            encoding["input_ids"], 
            attention_mask=encoding["attention_mask"]
        )
        # [batch_size, seq_len, hidden_size] 중 CLS 토큰 위치([:,0,:])의 임베딩만 추출
        output = output.last_hidden_state[:, 0, :]

        # 분류 레이어 -> 시그모이드로 각 감정별 점수 산출
        output = self.classifier(output)
        output = torch.sigmoid(output)

        # 사용 후 GPU 메모리 정리
        torch.cuda.empty_cache()

        return output

class ScriptEvaluator:
    """
    감정 분류 모델(KOTEtagger)를 이용해 사용자가 입력한 스크립트에 대해 3가지 척도를 계산하는 클래스
    """
    def __init__(self, model_path="kote_pytorch_lightning.bin", lexicon_path="lexicon_with_token.csv"): # model_path: 사전 학습된 감정 분류 모델 경로, lexicon_path: 사전 토큰화된 감정 사전 경로
        # 1) KOTEtagger 로드
        self.trained_model = KOTEtagger()
        self.trained_model.load_state_dict(torch.load(model_path), strict=False)
        self.trained_model.eval()

        # 2) 감정 사전(lexicon) 로드
        self.lexicon = pd.read_csv(lexicon_path)[['단어', '감정', 'token']]
        # 토큰화 시 감정 전달이 안 되는 단어 제거
        self.lexicon = self.lexicon[~self.lexicon['단어'].isin([
            '기분 좋다', '보기 좋다', '기분 나쁘다', '치 떨리다', '기운 없다', '가슴 아프다', '코웃음 치다'
        ])].reset_index(drop=True)

        # token 컬럼만 리스트로 추출
        self.lexicon_token = self.lexicon['token'].tolist()

        # Kiwi 형태소 분석기
        self.kiwi = Kiwi()
    
    # 상황 포맷팅 함수
    def format_situation(self, situation: str):
        """
        'situ_1-1' 형태가 아니라면, situ_ 를 자동으로 붙여주는 등 상황 키(situation)를 맞춰주는 함수
        예: "1-1" -> "situ_1-1", "2-1-1" -> "situ_2-1"
        """
        if not situation.startswith("situ_"):
            if "-" in situation:
                parts = situation.split("-")
                situation = f"situ_{parts[0]}-{parts[1]}"
            else:
                situation = f"situ_{situation}"
        return situation

    def evaluate_script(self, script: str, situation: str):
        """
        스크립트(문장)와 상황(situation)에 대해, 44개 감정 라벨 점수(0~1)를 예측하여 딕셔너리 형태로 반환
        """
        situation = self.format_situation(situation)
        result = {'situation': situation}
        
        # 모델 예측
        preds = self.trained_model(script)[0].detach().cpu().numpy()
        
        # 라벨, 점수 매핑
        for l, p in zip(LABELS, preds):
            result[l] = p
        return result

    # 척도 1(감정 분류 점수) 계산
    def measure1(self, script: str, situation: str):
        """
        forbidden_emotions에 정의된 라벨(emotion)이 threshold 이상인 경우 감정표현불능증 점수에 반영하는 함수
        평균 점수로 계산하기 위해 forbidden_emotions의 감정 라벨 개수로 나눔
        """
        situation = self.format_situation(situation)
        
        # 스크립트 전체 감정 분류 결과
        result = self.evaluate_script(script, situation)
        
        # 해당 상황에서 오답 감정 리스트
        forbidden = forbidden_emotions.get(situation, [])
        
        score = 0.0
        # threshold 이상인 감정을 합산
        for emotion in forbidden:
            prob = result.get(emotion, 0)
            if prob > 0.0: # threshold = 0.0
                score += prob

        # 오답 감정 개수로 나눈 평균값
        if len(forbidden) == 0:
            return 0.0
        return score / len(forbidden)

    def extract_tokens(self, script: str):
        """
        입력된 text를 문장 단위로 분할(split_into_sents)한 뒤, Kiwi를 이용하여 형태소 분석을 진행하고, 각 토큰을 추출해 리스트로 반환
        """
        tokens = []
        # 문장 단위로 split
        for sentence in self.kiwi.split_into_sents(script):
            # Kiwi로 문장 분석
            for token, _, _, _ in self.kiwi.analyze(sentence.text)[0][0]:
                tokens.append(token)
        return tokens

    # 척도 2(어휘 표현 다양성 점수) 계산
    def measure2(self, script: str):
        """
        감정 토큰의 고유 개수 / 전체 감정 토큰 수
        """
        # 전체 토큰
        script_tokens = self.extract_tokens(script)
        # 감정 사전에 포함된 토큰만 추출
        script_emotions = [token for token in script_tokens if token in self.lexicon_token]

        # 고유 토큰 수 / 전체 감정 토큰 수
        unique = set(script_emotions)
        return len(unique) / len(script_emotions) if script_emotions else 0.0

    # 척도 3(감정 표현 밀도) 계산
    def measure3(self, script: str):
        """
        감정 토큰 수 / 전체 토큰 수
        """
        # 전체 토큰
        script_tokens = self.extract_tokens(script)
        # 감정 토큰
        script_emotions = [token for token in script_tokens if token in self.lexicon_token]
        return len(script_emotions) / len(script_tokens) if script_tokens else 0.0


if __name__ == "__main__":
    # 평가 클래스 초기화
    evaluator = ScriptEvaluator()

    # 사용자 입력 받기
    user_script = input("Enter your script: ")
    situation = input("Enter the situation: ")

    # 세 가지 척도 결과 출력
    print("감정표현불능증 평가 결과")
    print(f"척도 1: {evaluator.measure1(user_script, situation)}")
    print(f"척도 2: {evaluator.measure2(user_script)}")
    print(f"척도 3: {evaluator.measure3(user_script)}")
# 1) 베이스 이미지 선택
FROM python:3.9-slim

# 2) 컨테이너 내에서 작업할 디렉터리 생성/설정
WORKDIR /app

# 3) requirements.txt 복사 및 의존성 설치
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# 4) 소스 코드 및 기타 파일(.py, .ipynb, .bin, .csv, .xlsx 등) 복사
COPY 감정표현불능증_평가척도_개발및검증.ipynb /app/
COPY 감정표현불능증_평가.py /app/
COPY kote_pytorch_lightning.bin /app/
COPY lexicon_with_token.csv /app/
COPY 감성대화말뭉치(최종데이터)_Training.xlsx /app/

# 5) 컨테이너 실행 시 어떤 명령을 수행할지 지정
#    - Python 파일 실행(예: 감정표현불능증_평가.py) 또는
#      Jupyter Notebook 등 원하는 명령을 작성
#    - 여기서는 일단 Python 스크립트 실행을 디폴트로 설정
CMD ["python", "감정표현불능증_평가.py"]
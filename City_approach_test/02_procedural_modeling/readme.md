1. 개별 빌딩 meta data, svg 저장

2. footprint library
step1. 특징 추출 (Feature Extraction)
이 폴리곤의 기하학적 특성을 계산하여 5가지 핵심 특징(Feature)을 추출합니다.
면적 (area): 건물이 얼마나 넓은가.
가로세로비 (ar - aspect ratio): 건물이 얼마나 길쭉한가. (1에 가까우면 정사각형, 값이 크면 길쭉한 직사각형)
볼록성 (convexity): 건물이 얼마나 볼록한 형태인가. (1에 가까우면 사각형처럼 볼록하고, 작을수록 'ㄷ'자나 'ㅁ'자처럼 오목한 부분이 많음)
복잡도 (complexity): 면적 대비 둘레가 얼마나 긴가. (값이 클수록 외벽이 복잡하고 요철이 많음)
구멍 개수 (holes): 건물 내부에 'ㅁ'자 형태의 중정이 몇 개나 있는가.
결과: 수천 개의 건물이 각각 5개의 숫자(특징 벡터)로 표현된 거대한 데이터 테이블(features.csv의 원본)이 만들어집니다.

Step2. 클러스터링 (Clustering)
목표: '비슷한 성격'의 건물들끼리 자동으로 묶어 그룹을 만듭니다.
알고리즘: K-평균 클러스터링 (K-Means Clustering)
scikit-learn 라이브러리의 KMeans를 사용합니다.
스크립트 실행 시 --k라는 인자로 그룹의 개수(K)를 지정합니다. (기본값 10)
1단계에서 추출한 5차원 특징 공간에서 서로 가까이 모여 있는 건물 데이터들을 찾아 K개의 군집(Cluster)으로 묶습니다.
중요: K-Means는 데이터의 단위(scale)에 민감하므로, StandardScaler를 사용해 모든 특징 값의 범위를 비슷하게 맞춰주는 표준화(Standardization) 전처리 과정을 거칩니다. (예: 면적은 수천 ㎡인데 가로세로비는 1~10 사이이므로, 이 영향력을 비슷하게 만들어 줌)
결과: 모든 건물은 자신이 속한 그룹 번호(예: 0번 그룹, 1번 그룹, ...)를 할당받게 됩니다.

출력 파일 형식 변경:
library_raw.json (신규 생성): LLM에게 입력을 주기 위한 핵심 파일입니다. 각 cluster_i에 어떤 건물 ID들이 속하는지와 전체 비율 정보를 담고 있습니다. 기존의 library.json을 대체합니다.
library_report.json (내용 변경): LLM이 각 클러스터의 특징을 잘 이해할 수 있도록, 상세 통계 리포트의 그룹 이름을 cluster_i 기준으로 변경했습니다.
features.csv (내용 변경): 개별 건물 데이터 파일에서 마지막 'group' 열에 'tower' 대신 cluster_i가 들어가도록 수정했습니다.

3. LLM 으로 그룹이름 붙이기
LLM prompt 1 + library_report.json 주고 naming.json 만들기
namping.py로 마지막 library.josn 만들기


4. Procedural generator 짜기


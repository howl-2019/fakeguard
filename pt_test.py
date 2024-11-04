import numpy as np

# 텍스트 파일 경로
file1 = 'target.txt'  # 첫 번째 파일 경로
file2 = 'source.txt'  # 두 번째 파일 경로

# 파일에서 데이터 로드
data1 = np.loadtxt(file1)
data2 = np.loadtxt(file2)

# 부호 비교
signs_comparison = np.sign(data1) == np.sign(data2)

# 결과 출력 (부호가 같으면 0, 다르면 1)
results = np.where(signs_comparison, 0, 1)

# 결과 출력
count_of_ones = np.sum(results)

# 결과 출력
print("부호가 다른 필드의 개수:", count_of_ones)
print(results)

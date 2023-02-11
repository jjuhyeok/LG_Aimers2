## brief conclusion

1. iterativeimputer 사용 
- train NaN : median 대체
- test NaN : train 에 iterativeimputer fit, test에 transform.
- 분포도가 그닥 좋지 않음. (0:11개, 1:287개, 2:12개)

2. iterativeimputer 사용
- train NaN : iterativeimputer 대체
- test NaN : train 에 iterativeimputer fit, test에 transform.
- 0을 예측 못함

3. 중복 feature 제거 + 변수선택법으로 feature dimension 좀 줄이면 성능이 향상될 수 도
4. CTGAN으로 data augument 생각
- https://github.com/sdv-dev/CTGAN

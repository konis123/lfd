# lfd
adasone 현장실습 할 때, 과제임


'''
알고있는 정보 (단위: cm)

카메라 높이 : 160
카메라에서 보닛까지의 거리 : 100
보닛에서 체스보드까지와의 거리 : 100
체스보드한칸의 길이 : 20
체스보드 밑변과 바닥까지의 거리 : 100
체스보드 세로 코너 수 : 15
체스보드 가로 코너 수 : 5
'''

#사진에서의 체스보드 모양과 위의 정보를 토대로,
#카메라에서 입력받은 숫자만큼 떨어져있는 선을 표시해줌.

#마우스로 더블 클릭하면 카메라기준으로 얼마나 멀리 떨어져있는지 계산함.



#현재 체스보드인식할때 체스보드 주변이 흰색바탕으로 되어있어야 체스보드를 제대로 인식할 수 있음
#그래서 지금 정확히 실행되는 샘플이 iphone3_white0.jpg 밖에 없음.(다른거도 흰색으로 주변을 표시하면 됨...)
#뭔가 사진에서 체스보드만 추출한다음에 findchessboardcorners 함수 쓰면 될거같기도한데....

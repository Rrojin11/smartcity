import pickle
import sys
import os
import pandas as pd
import numpy as np
path_dir = 'data_df8'
num_base = 4670
#데이터 합치기
output = pd.concat([pd.read_pickle(os.path.join(path_dir,x) ) for x in os.listdir('data_df8')])
with open('df_8'+ '.picle', 'wb') as f:
    pickle.dump(output, f)


#hop cnt정보 불러오기
'''
with open('data/hopadj.picle', 'rb') as f:
    hopadj = pickle.load(f)
'''
# 그래프정보 불러오기
with open('data/Graph_info.pkl', 'rb') as f:
    graph = pickle.load(f)
print(graph)
print(graph.shape)
def most_find(df, n): #해당 base Idx의 df -> top n개 Target Idx를 list로 반환
    rate = df.loc[:,['TargetIdx', 'Rate']]
    lst = sorted(df, key=lambda x: x[1], reverse=True) #내림 차순 정렬
    return lst[:n]
'''
for node in range(num_base):
    print('Node : ',node)
    for cnt in range(1,4):
        print(cnt)
        hop_idx = np.where(np.array(hopadj[node]) == cnt)[0]
        print(hop_idx)

print('<<LIST>> ', os.listdir('data_df8'))
print(output.shape)
print(output.columns)

for base in range(2): #테스트용
#for base in range(num_base): #실제 전체
    locals()['df8_{}'.format(base)] = output[output['BaseIdx']==0]
    locals()['df8_sort_{}'.format(base)] = locals()['df8_{}'.format(base)].sort_values(by=['Rate'], axis =0, ascending = False)
    print(locals()['df8_sort_{}'.format(base)])

#rate 별 내림차순으로 sort해서 저장하기
df_sort_8 = []
for base in range(num_base):
    if base%100==0:
        print(base)
    locals()['df8_{}'.format(base)] = output[output['BaseIdx'] == 0]
    locals()['df8_sort_{}'.format(base)] = locals()['df8_{}'.format(base)].sort_values(by=['Rate'], axis=0, ascending=False)
    df_sort_8.append(locals()['df8_sort_{}'.format(base)])

df_sort_8 = pd.concat(df_sort_8)
with open('df_sort_8.pickle', 'wb') as f:
    pickle.dump(df_sort_8, f)
'''
with open('df_sort_8.pickle', 'rb') as f:
    df = pickle.load(f)

print(df.columns)
'''
시간대별 변화량을 추정하는 dnn 모델을 만든 이후, 모든 노드에 대해서 hop count 3이내의 node들을 변화시켰을 때의 변화량을 계산하였습니다. 
현재는 오전 8시를 target으로 잡은 계산 결과만 있습니다. (--> 이 계산 과정이 상당히 오래 걸립니다ㅠㅠ)
계산 결과는 dataframe형식으로 정리하였습니다. df의 columns들은 ['Time', 'BaseIdx', 'TargetIdx', 'Hop', 'Origin', 'Changed', 'Rate'] 입니다.
Time 은 해당 데이터들의 시간대입니다.(현재 = 8)
BaseIdx가 기준이 되는 노드이고, TargetIdx가 변화시켜본 node 입니다. TargetIdx의 속도를 0.5배로 변화시켰을 때 BaseIdx가 얼만큼 변화는 지 확인합니다.
Hop는 baseIdx에 대해서 TargetIdx가 몇 hop count 떨어져 있는지 표현한 것이고, Origin은 원래 속도(0,1 scaler적용값), Changed는 변화한 속도 입니다.
Rate는 np.abs((changed - original) / original).item() *100 과 같이 변화 퍼센트를 계산했습니다.
최종적으로, baseIdx를 기준으로 Rate를 내림차순으로 정리하였습니다. 
df_sort_8.pickle 파일 공유드리겠습니다. 
'''
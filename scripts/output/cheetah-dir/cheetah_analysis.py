import pandas as pd
import matplotlib.pyplot as plt


#df_1=pd.read_csv(r'2020_03_21_20_51_06/progress.csv') Short one
df_2=pd.read_csv(r'2020_03_21_22_27_19/progress.csv')
df_3=pd.read_csv(r'2020_03_22_08_33_29/progress.csv')
#df_4=pd.read_csv(r'2020_03_22_08_33_33/progress.csv')

df_list=[df_2,df_3]

for df in df_list:
    plt.plot(df['Number of env steps total'], df['AverageReturn_all_test_tasks'])
plt.show()
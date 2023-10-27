import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import missingno as msno
import numpy as np

class Visualize():
    def predictive_mean_matching(self, data, target_column, n_neighbors=1):
        # 完全なデータと欠測データを分離
        data = data.copy()
        complete_data = data.dropna()
        missing_data = data[data[target_column].isnull()]
        missing_data = missing_data.fillna(complete_data.median())
        
        # 予測モデルの学習
        X_complete = complete_data.drop(columns=[target_column])
        y_complete = complete_data[target_column]
        model = LinearRegression().fit(X_complete, y_complete)
        
        # 欠測データに対する予測を行う
        X_missing = missing_data.drop(columns=[target_column])
        predicted = model.predict(X_missing)
        
        # 予測値に基づいて、最も近い観測値を見つける
        imputed_values = []
        for p in predicted:
            distances = np.abs(y_complete - p)
            closest_indices = distances.nsmallest(n_neighbors).index
            imputed_value = y_complete.loc[np.random.choice(closest_indices)]
            imputed_values.append(imputed_value)
        
        # 欠測値を補完
        data.loc[data[target_column].isnull(), target_column] = imputed_values
        return data
    
    def splitdata(self, df, i):
        sorted_rows = df.iloc[:,i].sort_values(ascending=False).index
        # 並べ替えた行順でデータフレームを再構成
        missing_data = df.reindex(sorted_rows)
        missing_data1 = missing_data[missing_data.iloc[:,i]==1]
        missing_data2 = missing_data[missing_data.iloc[:,i]==0]
        if missing_data1[missing_data1.iloc[:,i+1]==1].shape[0] != 0:
            missing_data1 = self.splitdata(missing_data1, i+1)
        if missing_data2[missing_data2.iloc[:,i+1]==1].shape[0] != 0:
            missing_data2 = self.splitdata(missing_data2, i+1)
        return pd.concat([missing_data1, missing_data2], axis=0)

    def visualize(self, df):
        # 欠損値の有無を示す真偽値のデータフレームを作成
        missing_data = df.isnull()

        # 各列の欠損値の数を取得し、降順でソート
        sorted_columns = missing_data.sum().sort_values(ascending=False).index

        # 並べ替えた列順でデータフレームを再構成
        missing_data = missing_data[sorted_columns]

        # 各行の欠損値の数を取得し、降順でソート
        missing_data = self.splitdata(missing_data, 0)


        # ヒートマップを使用して欠損値を可視化
        plt.figure(figsize=(8, 6))
        sns.heatmap(missing_data, cbar=False, cmap='Reds', yticklabels=False)

        # グラフのタイトルとラベルを設定
        plt.title("Missing Data Visualization")
        plt.xlabel("Columns")
        plt.ylabel("Rows")
        plt.savefig("vis.png")
        plt.close()

        # 各列の欠損値の数を取得
        missing_counts = missing_data.sum()

        # ヒストグラムを表示
        missing_counts.plot(kind='bar', color='skyblue')
        plt.title('Number of Missing Values Per Column')
        plt.ylabel('Number of Missing Values')
        plt.xlabel('Columns')
        plt.savefig("vis_column.png")
        plt.close()

        # 各列の欠損値の割合を取得
        missing_counts = missing_data.sum() / missing_data.shape[0]

        # ヒストグラムを表示
        missing_counts.plot(kind='bar', color='skyblue')
        plt.title('Percentage of Missing Values Per Column')
        plt.ylabel('Percentage of Missing Values')
        plt.xlabel('Columns')
        plt.savefig("vis_column_percent.png")
        plt.close()

        # 1. 全体の重複データの数を確認
        total_duplicates = missing_data.duplicated().sum()
        duplicates = missing_data[missing_data.duplicated(keep=False)]
        sorted_duplicates = duplicates.groupby(by=list(missing_data.columns)).size().reset_index(name='Count').sort_values(by='Count', ascending=False)
        sorted_duplicates.set_index('Count', inplace=True)

        # ヒートマップで表示
        plt.figure(figsize=(10, 8))
        sns.heatmap(sorted_duplicates*1, annot=False, cmap='Reds', cbar=False)
        plt.title('pattern of missing data')
        plt.savefig("vis_column_missing_patern.png")


    def vis_distribusion(self, df1, hue):
        # A列に欠損がある行のB, Cの分布
        df = df1.copy()
        missing_data = df.isnull() * 1
        other_col = list(df.columns)
        other_col.remove(hue)
        df.loc[:,hue] = missing_data.loc[:,hue]
        # 一つの図に表示するための設定
        fig, axes = plt.subplots(ncols=len(df.columns) - 1, figsize=(4 * (len(df.columns) - 1), 4))

        # 各説明変数の分布をviolinplotで表示
        for ax, col in zip(axes, df.drop(hue, axis=1).columns):
            print(col)
            sns.boxplot(data=df, x=hue, y=col, hue=hue, whis=[0, 100], width=.6, ax=ax)
            sns.stripplot(df, x=hue, y=col, hue=hue, size=4, palette='dark:.3', alpha=.3, dodge=True, ax=ax)
            ax.set_title(f'Distribution of {col} by Flag')
            ax.set_xlabel(hue)
            ax.set_ylabel(col)

        plt.tight_layout()
        plt.savefig("vis_pair.png")
        plt.close()
        '''
        sns.catplot(data=df, kind="violin", x="day", y="total_bill", hue="smoker", split=True)
        df[df.iloc[:,col].isnull()][other_col].hist(color='skyblue', alpha=0.7, label='Missing in A')
        plt.suptitle('Distribution when A is missing')
        plt.legend()
        plt.show()

        # A列に欠損がない行のB, Cの分布
        df[df.iloc[:,col].notnull()][other_col].hist(color='salmon', alpha=0.7, label='Not Missing in A')
        plt.suptitle('Distribution when A is not missing')
        plt.legend()
        plt.show()
        '''

    def vis_y(self, df, missing_df, col):
        missing_data = missing_df[col].isnull() * 1
        df1 = df[missing_data==1]
        df2 = df[missing_data==0]
        df1 = df1[col]
        df2 = df2[col]

        #dff = pd.concat([df1, df2], axis=1)
        plt.figure(figsize=(8, 5))
        #sns.histplot(data=df1, color="blue", label="observation")
        #sns.histplot(data=df2, color="red", label="missing")
        sns.kdeplot(data=df1, fill=False, color="blue", bw_adjust=0.3, label="observation")
        sns.kdeplot(data=df2, fill=False, color="red", bw_adjust=0.3, label="missing")
        sns.kdeplot(data=df[col], fill=False, color="green", bw_adjust=0.3, label="total")
        plt.legend()
        plt.savefig("vis_impute.png")
        plt.close()
        #dff.plot.hist(alpha=0.5)




if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    print(df.isnull().sum())
    Visualize().visualize(df)
    ax1 = msno.heatmap(df)
    plt.savefig("vis_mno.png")

    # カテゴリカルデータのカラムだけを取得してエンコーディング
    label_encoders = {}  # 各カラムのエンコーダを保存するための辞書
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = pd.Series(le.fit_transform(df[col][df[col].notnull()]), index=df[col][df[col].notnull()].index)
        label_encoders[col] = le  # 必要に応じてエンコーダを保存

    Visualize().vis_distribusion(df, "Cabin")

    # 欠測値補完
    imputed_data = Visualize().predictive_mean_matching(df, 'Cabin', n_neighbors=1)
    Visualize().vis_y(imputed_data, df, 'Cabin')
    a = 0


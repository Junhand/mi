import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class Visualize():
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

    def vis_distribusion(self, df, hue):
        # A列に欠損がある行のB, Cの分布
        missing_data = df.isnull() * 1
        other_col = list(df.columns)
        other_col.remove(hue)
        df.loc[:,hue] = missing_data.loc[:,hue]
        # 一つの図に表示するための設定
        fig, axes = plt.subplots(ncols=len(df.columns) - 1, figsize=(4 * (len(df.columns) - 1), 4))

        # 各説明変数の分布をviolinplotで表示
        for ax, col in zip(axes, df.drop(hue, axis=1).columns):
            print(col)
            sns.violinplot(data=df, y=col, x=hue, hue=hue, ax=ax)
            ax.set_title(f'Distribution of {col} by Flag')
            ax.set_xlabel(hue)
            ax.set_ylabel(col)

        plt.tight_layout()
        plt.savefig("vis_pair.png")
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


if __name__ == "__main__":
    df = pd.read_csv('train.csv')
    print(df.isnull().sum())
    Visualize().visualize(df)
    # カテゴリカルデータのカラムだけを取得してエンコーディング
    label_encoders = {}  # 各カラムのエンコーダを保存するための辞書
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = pd.Series(le.fit_transform(df[col][df[col].notnull()]), index=df[col][df[col].notnull()].index)
        label_encoders[col] = le  # 必要に応じてエンコーダを保存

    Visualize().vis_distribusion(df, "Cabin")

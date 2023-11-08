import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind
import numpy as np

def plot_df_t(df, target_col):

    fig = go.Figure()

    missing_data = df.isnull() * 1
    other_col = list(df.columns)
    other_col.remove(target_col)
    df.loc[:,target_col] = missing_data.loc[:,target_col]
    observed_data = df[missing_data[target_col] == 0]
    missing_data = df[missing_data[target_col] == 1]
    significant_cols = []
    p_values = []
    for col in df.drop(target_col, axis=1).columns:
        # 欠損していないデータと欠損データの両方を取得
        observed = observed_data[col].dropna()
        missing = missing_data[col].dropna()
        
        # t検定を実施
        t_stat, p_value = ttest_ind(observed, missing, equal_var=False)
        
        # 有意な結果（例えばp値が0.05未満）の場合はリストにカラム名を追加
        if p_value < 0.05:
            significant_cols.append(col)
            p_values.append(p_value)

    # 一つの図に表示するための設定
    num_cols = len(significant_cols)
    fig = make_subplots(rows=1, cols=len(significant_cols), horizontal_spacing=0.1) # 例として0.05を設定)
    # バイオリンプロットとストリッププロットを追加
    annotations = []
    for i, col in enumerate(significant_cols, start=1):
        # 観測データのボックスプロットを追加
        fig.add_trace(go.Violin(y=observed_data[col], name=f'Observed', 
                             points='all', jitter=1, pointpos=0, box_visible=True,
                             width=0.35, marker_color='blue', line_color='blue'), row=1, col=i)

        # 欠測データのボックスプロットを追加
        fig.add_trace(go.Violin(y=missing_data[col], name=f'Missing', 
                             points='all', jitter=1, pointpos=0, box_visible=True,
                             width=0.35, marker_color='red', line_color='red'), row=1, col=i)
        
        annotations.append({
            'x': 0.5,  # サブプロットのインデックスをx座標として使用
            'y': 1.05,  # y座標をプロットエリアの上部に設定
            'xref': f'x{i}', 
            'yref': 'paper',  # ペーパー座標を使用
            'text': f'p={p_values[i-1]:.3f}',
            'showarrow': False,
            'font': {'color': 'red', 'size': 8},
        })      
        fig.update_xaxes(title_text=target_col, row=1, col=i)  
        fig.update_yaxes(title_text=col, automargin=True, row=1, col=i)  

    # 更新オプション
    fig.update_layout(
        title=f"列{target_col}の観測と欠測における分布の違い（p：ウェルチのt検定）",
        title_x = 0.5,
        boxgap=0.5,      # ボックス間の間隔を広げる
        annotations=annotations,
        violinmode='group',
        showlegend=False,
        width=150*num_cols,
        height=600,
    )
    fig.update_xaxes(tickangle=-90, automargin=True)

    # 表示
    fig.show()

    # 保存
    fig.write_image("vis_pair_plotly.png")

def plot_df_scatter(df, target_col):
    # サブプロットの行と列の数を決定
    missing_data = df.isnull() * 1
    observed_df_other = df[missing_data[target_col] == 0]
    missing_df_other = df[missing_data[target_col] == 1]

    significant_cols = []
    for col in df.drop(target_col, axis=1).columns:
        # 欠損していないデータと欠損データの両方を取得
        observed = observed_df_other[col].dropna()
        missing = missing_df_other[col].dropna()
        
        # t検定を実施
        t_stat, p_value = ttest_ind(observed, missing, equal_var=False)
        
        # 有意な結果（例えばp値が0.05未満）の場合はリストにカラム名を追加
        if p_value < 0.05:
            significant_cols.append(col)

    num_cols = len(significant_cols)

    specs = [[0 for _ in range(num_cols*4)] for _ in range(4)]
    column_widths = []
    for i in range(num_cols*4):
        if i % 4==0:
            specs[0][i] = {"type": "box"}
            specs[1][i] = {"type": "box"}
            specs[2][i] = {"type": "box"}
            specs[3][i] = {"type": "box"}
            column_widths.append(0.2)
        elif i % 4==1:
            specs[0][i] = {"type": "box"}
            specs[1][i] = {"type": "scatter"}
            specs[2][i] = {"type": "box"}
            specs[3][i] = {"type": "box"}
            column_widths.append(0.1)
        elif i % 4==2:
            specs[0][i] = {"type": "box"}
            specs[1][i] = {"type": "scatter"}
            specs[2][i] = {"type": "scatter"}
            specs[3][i] = {"type": "box"}
            column_widths.append(0.6)
        elif i % 4==3:
            specs[0][i] = {"type": "box"}
            specs[1][i] = {"type": "box"}
            specs[2][i] = {"type": "box"}
            specs[3][i] = {"type": "box"}
            column_widths.append(0.1)
   
    fig = make_subplots(
        rows=4, cols=num_cols*4,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.005,
        horizontal_spacing=0.001,
        column_widths = column_widths,
        row_heights = [0.1, 0.6, 0.1, 0.2],
        specs=specs
    )

    # 各列に対してプロットを追加
    for i, col in enumerate(significant_cols, start=1):
        observed_df_target = df[missing_data[col] == 0]
        missing_df_target = df[missing_data[col] == 1]

        # 主要な散布図を描画
        fig.add_trace(
            go.Scatter(
                x=df[col], 
                y=df[target_col], 
                mode='markers', 
                name=f'Scatterplot {col}'
            ),
            row=2, col=i*4-1
        )
        a = missing_df_other[col].values
        b = np.zeros(len(missing_df_other[col]))
        fig.add_trace(
            go.Scatter(x=missing_df_other[col],
                       y=np.zeros(len(missing_df_other[col])),
                      mode ='markers',
                      marker_color='red',
                      ),
            row=3, col=i*4-1
        )
        fig.add_trace(
            go.Scatter(x=np.zeros(len(missing_df_target[target_col])),
                        y=missing_df_target[target_col].values, 
                      marker_color='red',  # カラースケール変更
                      mode ='markers',
                      ),
            row=2, col=i*4-2
        )

        # 箱ひげ図を描画
        fig.add_trace(
            go.Box( 
                name=f'Observed',
                x=observed_df_other[col], 
                jitter=0.5, pointpos=0, 
                offsetgroup=1,
                boxpoints='all',
                marker_color='blue',
                orientation='h'
            ),
            row=4, col=i*4-1
        )
        fig.add_trace(
            go.Box( 
                name=f'Missing',
                x=missing_df_other[col], 
                jitter=0.5, pointpos=0, 
                offsetgroup=1,
                boxpoints='all',
                marker_color='red',
                orientation='h'
            ),
            row=4, col=i*4-1
        )

        fig.add_trace(
            go.Box( 
                name=f'Observed',
                y=observed_df_target[target_col], 
                jitter=0.5, pointpos=0, 
                offsetgroup=1,
                boxpoints='all',
                marker_color='blue',
            ),
            row=2, col=i*4-3
        )
 
        # 箱ひげ図を描画
        fig.add_trace(
            go.Box(
                name=f'Missing',
                y=missing_df_target[target_col], 
                jitter=0.5, pointpos=0, 
                offsetgroup=1,
                boxpoints='all',
                marker_color='red',
            ),
            row=2, col=i*4-3
        )
        # 軸の設定
        #
        fig.update_xaxes(title_text=col, row=4, col=i*4-1)
        fig.update_xaxes(showgrid=False, row=2, col=i*4-2)
        fig.update_xaxes(showticklabels=True, tickangle=-90, automargin=True, row=2, col=i*4-3)

        fig.update_yaxes(showticklabels=True, row=4, col=i*4-1)
        fig.update_yaxes(showgrid=False, row=3, col=i*4-1)
        fig.update_yaxes(showticklabels=False, row=2, col=i*4-1)
        fig.update_yaxes(showticklabels=False, row=2, col=i*4-2)
        fig.update_yaxes(title_text=target_col, showticklabels=True, row=2, col=i*4-3)

    # グリッド内のプロット間のスペースを調整
    fig.update_layout(
        showlegend=False,
        boxmode='group',
        width=600*num_cols,
        height=600,
        margin=dict(t=20, b=20, l=20, r=20),
    )

    # プロットの表示
    fig.show()



def plot_df_ts(df):
    # サブプロットの行と列の数を決定
    missing_data = df.isnull() * 1

    t_lists = [[]]
    for i, col in enumerate(range(df.columns)):
        observed_df_other = df[missing_data[col] == 0]
        missing_df_other = df[missing_data[col] == 1]

        for j in range(df[col].drop().columns):
            if df[j].isnull().sum() == 0: continue
            # t検定を実施
            t_stat, p_value = ttest_ind(observed_df_other[j], missing_df_other[j], equal_var=False)
            

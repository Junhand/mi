import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind

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
    fig = make_subplots(rows=1, cols=len(significant_cols), horizontal_spacing=0.05 ) # 例として0.05を設定)
    # バイオリンプロットとストリッププロットを追加
    annotations = []
    for i, col in enumerate(significant_cols, start=1):
        # 観測データのボックスプロットを追加
        fig.add_trace(go.Violin(y=observed_data[col], name=f'Observed {col}', 
                             points='all', jitter=1, pointpos=0, 
                             width=0.35, marker_color='blue', line_color='blue'), row=1, col=i)

        # 欠測データのボックスプロットを追加
        fig.add_trace(go.Violin(y=missing_data[col], name=f'Missing {col}', 
                             points='all', jitter=1, pointpos=0, 
                             width=0.35, marker_color='red', line_color='red'), row=1, col=i)
        

        annotations.append({
            'x': 0.5,  # サブプロットのインデックスをx座標として使用
            'y': 1.1,  # y座標をプロットエリアの上部に設定
            'xref': f'x{i}', 
            'yref': 'paper',  # ペーパー座標を使用
            'text': f'p={p_values[i-1]:.3f}',
            'showarrow': False,
            'font': {'color': 'red', 'size': 8},
        })        

    # 更新オプション
    fig.update_layout(
        title=f"列{target_col}の観測と欠測における分布の違い（p：スチューデントのt検定）",
        title_x = 0.5,
        boxgap=0.5,      # ボックス間の間隔を広げる
        annotations=annotations,
        violinmode='group'
    )
    fig.update_xaxes(tickangle=-90, automargin=True)

    # 表示
    fig.show()

    # 保存
    fig.write_image("vis_pair_plotly.png")

def plot_df_scatter(df, target_col):
    # サブプロットの行と列の数を決定
    num_cols = len(df.columns)
    missing_data = df.isnull() * 1
    fig = make_subplots(
        rows=2, cols=num_cols*2,
        #shared_xaxes=True,
        #shared_yaxes=True,
        vertical_spacing=1/8,
        horizontal_spacing=1/(num_cols*2-1)/8,
        column_widths = [0.2/(num_cols*2) if i % 2 == 0 else 0.8/(num_cols*2) for i in range(num_cols * 2)],
        row_heights = [0.8, 0.2],
        specs=[[{"type": "box"} if i%2==0 else {"type": "scatter"} for i in range(num_cols*2)] for _ in range(1)] + 
             [[{"type": "box"} if i%2==0 else {"type": "box"} for i in range(num_cols*2)] for _ in range(1)]
    )

    observed_df_other = df[missing_data[target_col] == 0]
    missing_df_other = df[missing_data[target_col] == 1]
    # 各列に対してプロットを追加
    for i, col in enumerate(df.columns, start=1):
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
            row=1, col=i*2
        )
        # 軸の設定
        fig.update_xaxes(title_text=col, row=1, col=i*2)
        fig.update_yaxes(title_text=target_col, row=1, col=i*2)
        # 全体の幅と高さを設定する
        #fig.update_layout(width=800, height=800)

        
        # 箱ひげ図を描画
        fig.add_trace(
            go.Box( 
                name=f'Observed{target_col}',
                x=observed_df_other[col], 
                orientation='h'
            ),
            row=2, col=i*2
        )
        fig.add_trace(
            go.Box( 
                name=f'Missing{target_col}',
                x=missing_df_other[col], 
                orientation='h'
            ),
            row=2, col=i*2
        )

        fig.add_trace(
            go.Box( 
                name=f'Observed{col}',
                y=observed_df_target[target_col], 
            ),
            row=1, col=i*2-1
        )

        # 箱ひげ図を描画
        fig.add_trace(
            go.Box(
                name=f'Missing{col}',
                y=missing_df_target[target_col], 
            ),
            row=1, col=i*2-1
        )
        
        # # 軸の設定
        #fig.update_xaxes(title_text=col, row=1, col=i)
        #fig.update_yaxes(title_text=target_col, row=1, col=i)

        #fig.update_xaxes(showticklabels=False, row=2, col=i+1)
        #if i % 2 == 0:
        #    fig.update_yaxes(showticklabels=False, row=1, col=i+1)

    # グリッド内のプロット間のスペースを調整
    fig.update_layout(
        showlegend=False,
        boxmode='group',
        width=400*(num_cols-1),
        height=400,
        grid={'rows': 2, 'columns': num_cols*2}
    )

    # プロットの表示
    fig.show()

    
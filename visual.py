import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.stats import ttest_ind
import numpy as np
import pandas as pd

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
        vertical_spacing=0.00,
        horizontal_spacing=0.00,
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

        fig.add_trace(
            go.Scatter(x=missing_df_other[col],
                       y=np.zeros(len(missing_df_other[col])),
                      mode ='markers',
                      marker_color='red',
                      ),
            row=3, col=i*4-1
        )
        fig.add_trace(
            go.Scatter(x=np.zeros(len(missing_df_target[target_col]))if len(missing_df_target[target_col]) != 0 else [0],
                       y=missing_df_target[target_col] if len(missing_df_target[target_col]) != 0 else [0], 
                       opacity= 0 if len(missing_df_target[target_col]) == 0 else 1,
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
                jitter=1.0, 
                pointpos=0, 
                offsetgroup=1,
                boxpoints='all',
                marker=dict(size=3),
                marker_color='blue',
                orientation='h'
            ),
            row=4, col=i*4-1
        )
        fig.add_trace(
            go.Box( 
                name=f'Missing',
                x=missing_df_other[col], 
                jitter=1, 
                pointpos=0, 
                offsetgroup=1,
                boxpoints='all',
                marker=dict(size=3),
                marker_color='red',
                orientation='h'
            ),
            row=4, col=i*4-1
        )

        fig.add_trace(
            go.Box( 
                name=f'Observed',
                y=[0] if observed_df_target[target_col] is None else observed_df_target[target_col], 
                jitter=1, 
                pointpos=0, 
                offsetgroup=1,
                opacity= 0 if len(observed_df_target[target_col]) == 0 else 1,
                boxpoints='all',
                marker=dict(size=3),
                marker_color='blue',
            ),
            row=2, col=i*4-3
        )
 
        # 箱ひげ図を描画
        fig.add_trace(
            go.Box(
                name=f'Missing',
                y=[0] if len(missing_df_target[target_col]) == 0 else missing_df_target[target_col], 
                jitter=1, 
                pointpos=0, 
                offsetgroup=1,
                opacity= 0 if len(missing_df_target[target_col]) == 0 else 1,
                boxpoints='all',
                marker=dict(size=3),
                marker_color='red',
            ),
            row=2, col=i*4-3
        )
        # 軸の設定
        #
        fig.update_xaxes(row=2, col=i*4-1,linecolor='gray', linewidth=1.5 ,mirror=True)
        fig.update_xaxes(showgrid=False, row=2, col=i*4-2,linecolor='gray', linewidth=1.5 ,mirror=True)
        fig.update_xaxes(showticklabels=True, tickangle=-90, automargin=True, row=2, col=i*4-3,linecolor='gray', linewidth=1.5 ,mirror=True)
        fig.update_xaxes(showgrid=False, row=3, col=i*4-1)
        fig.update_xaxes(title_text=col, row=4, col=i*4-1,linecolor='gray', linewidth=1.5,mirror=True)
        
        
        fig.update_yaxes(showticklabels=False, row=2, col=i*4-1,linecolor='gray', linewidth=1.5 ,mirror=True)
        fig.update_yaxes(showticklabels=False, row=2, col=i*4-2)
        fig.update_yaxes(title_text=target_col, showticklabels=True, row=2, col=i*4-3,linecolor='gray', linewidth=1.5,mirror=True)
        fig.update_yaxes(showgrid=False, row=3, col=i*4-1,linecolor='gray', linewidth=1.5 ,mirror=True)
        fig.update_yaxes(showticklabels=True, row=4, col=i*4-1,linecolor='gray', linewidth=1.5 ,mirror=True)
        


    # グリッド内のプロット間のスペースを調整
    fig.update_layout(
        showlegend=False,
        boxmode='group',
        width=600*num_cols,
        height=600,
        margin=dict(t=20, b=20, l=20, r=20),
    )

    # プロットの表示
    return fig



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
            

def plot_listwize_df(df: pd.DataFrame, subset = None, title: str = "リストワイズ前後の分布の違い", fig_width:int=None, fig_height:int=None, font_size: int =16, bins:int=None) -> None:
    listwise_df = df.dropna()
    original_stats = df.describe()
    original_stats.loc['skew'] = df.skew() # 歪度の追加
    original_stats.loc['kurtosis'] = df.kurt() # 尖度の追加
    original_stats.loc['variance'] = df.var() # 分散の追加

    listwise_stats = listwise_df.describe()
    listwise_stats.loc['skew'] = listwise_df.skew()  # 歪度の追加
    listwise_stats.loc['kurtosis'] = listwise_df.kurt() # 尖度の追加
    listwise_stats.loc['variance'] = listwise_df.var() # 分散の追加

    specs=[[{"type": "xy"} if i %2 == 0 else {"type": "image"} for i in range(2*len(df.columns)) ]]
    fig = make_subplots(rows=1, cols=2*len(df.columns), horizontal_spacing=0.0, print_grid=True, vertical_spacing=0.1, specs=specs ) # 例として0.05を設定)
    # バイオリンプロットとストリッププロットを追加
    annotations = []
    for i, col in enumerate(df.columns):
        # 観測データのヒストグラムを追加
        fig.add_trace(go.Histogram(x=df[col], name='Original', legendwidth=fig_width, #, legend=f'legend{i+1}',#-58.5
                                marker_color='blue', opacity=0.5, legendgroup=f'{i}',
                                  nbinsx=bins), row=1, col=2*i+1)
        
        # リストワイズ後のヒストグラムを追加
        fig.add_trace(go.Histogram(x=listwise_df[col], name='Listwise',# legend=f'legend{i}',#, showlegend=True,
                                marker_color='red', opacity=0.5, legendgroup=f'{i}',
                                  nbinsx=bins), row=1, col=2*i+1)
        #fig.add_trace(go.Image(z=[-1,0]), row=1, col=2*i+2)
        
        # describe() を使用して統計情報を取得
        original_stats_info = df[col].describe()
        listwise_stats_info = listwise_df[col].describe()
        original_stats_info['skew'] = df[col].skew() # 歪度の追加
        original_stats_info['kurtosis'] = df[col].kurt() # 尖度の追加

        listwise_stats = listwise_df.describe()
        listwise_stats_info['skew'] = listwise_df[col].skew()  # 歪度の追加
        listwise_stats_info['kurtosis'] = listwise_df[col].kurt() # 尖度の追加

        # 統計情報をテキストとして整形
        original_stats = "<br>".join([f"{stat}: {original_stats_info[stat]:.2f}" for stat in original_stats_info.index])
        listwise_stats = "<br>".join([f"{stat}: {listwise_stats_info[stat]:.2f}" for stat in listwise_stats_info.index])
        max_length = max(len(f"{value:.2f}")+6 for value in original_stats_info)

        # 各サブプロットに注釈を追加
        annotations.append({
            'x': 1/len(df.columns)/2 + 1/len(df.columns)*i, #1, 
            'y': 1, 
            'xref': 'paper', 
            'yref': 'paper', 
            'text': original_stats,
            'showarrow': False,
            'xanchor':'left',  # アノテーションのテキストを左端に固定
            'yanchor':'top',
            'align':'left',  
            'font': {'color': 'gray', 'size': font_size},
            'xshift': 5,
            'yshift': -font_size*1.5,
        })
        annotations.append({
            'x': 1/len(df.columns)/2 + 1/len(df.columns)*i, 
            'y': 1, 
            'xref': 'paper', 
            'yref': 'paper', 
            'text': "Original",
            'showarrow': False,
            'xanchor':'left',  # アノテーションのテキストを左端に固定
            'yanchor':'top',
            'align':'left',  
            'font': {'color': '#0000FF', 'size': font_size+2},
            'xshift': 5,#+font_size*max_length/1.333,
            'yshift': 0,
        })
        annotations.append({
            'x': 1/len(df.columns)/2 + 1/len(df.columns)*i, 
            'y': 1, 
            'xref': 'paper', 
            'yref': 'paper', 
            'text': listwise_stats,
            'showarrow': False,
            'xanchor':'left',  # アノテーションのテキストを左端に固定
            'yanchor':'top',
            'align':'left',  
            'font': {'color': 'gray', 'size': font_size},
            #'bgcolor':'white',
            'xshift': 5+font_size*max_length/1.333,
            'yshift': -font_size*1.5,
        })
        annotations.append({
            'x': 1/len(df.columns)/2 + 1/len(df.columns)*i, #5, 
            'y': 1,
            'xref': 'paper',#f'x{2*i+2}', 
            'yref': 'paper', 
            'text': "Listwise",
            'showarrow': False,
            'xanchor':'left',  # アノテーションのテキストを左端に固定
            'yanchor':'top',
            'align':'left',  
            'font': {'color': 'red', 'size': font_size+2},
            'xshift': 5+font_size*max_length/1.333,
            'yshift': 0,
        })



        #fig.add_annotation(x=1/len(df.columns)/2+1/len(df.columns)*i, y=1, xref= "paper", yref= "paper", text=original_stats, showarrow=False)
        #fig.add_annotation(x=0, y=1, text=listwise_stats, showarrow=False, row=1, col=2*i+1)


        fig.update_xaxes(title_text=col, row=1, col=2*i+1)
        #fig.update_xaxes(#visible=False,range=[0, 10], showgrid=False, row=1, col=2*i)
       #fig.update_yaxes(visible=False, showgrid=False, row=1, col=2*i)
        #fig.update_traces(paper_bgcolor='rgba(255,255,255,1)', row=1, col=2*i-1)


    # タイトルと図の幅・高さを設定
    fig.update_layout(
        title='データの分布ヒストグラム',
        title_x = 0.5,
        #boxgap=0.5,      # ボックス間の間隔を広げる
        barmode='overlay',  # ヒストグラムを重ねる
        annotations=annotations,
        showlegend=True,
        #plot_bgcolor='rgba(255,255,255,1)',
        #paper_bgcolor='rgba(255,255,255,1)'

        #legend=dict(x=1/len(df.columns)/2),
        # legend=dict(
        #     traceorder= 'grouped',
        #     orientation='h',  # 水平方向に並べる
        #     x=1,  # 凡例のX軸位置を右端に設定
        #     y=1.15,  # 凡例のY軸位置を上端に設定
        #     xanchor='right',  # 凡例のX軸アンカーを右に
        #     yanchor='top',    # 凡例のY軸アンカーを上に
        #     #ho=200,
        #     #bgcolor='rgba(255, 255, 255, 0)',
        #     #itemwidth=30, 
        # ),
        #margin=dict(t=0, b=0, l=50, r=50),
    )
    cols = len(df.columns) * 2


    if fig_height is not None:
        fig.update_layout(
            height=fig_height
        )

    if fig_width is not None:
        fig.update_layout(
            width=fig_width*(2*len(df.columns))
        )
    # 表示
    return fig



# import dash
# from dash import html, dcc, Input, Output
# import plotly.express as px
# import pandas as pd

# # サンプルデータセット
# df = pd.DataFrame({
#     'A': range(1, 11),
#     'B': [2, 3, 4, 5, 4, 3, 2, 1, 2, 3],
#     'C': [5, 3, 3, 3, 5, 6, 7, 8, 9, 10]
# })

# # Dashアプリの初期化
# app = dash.Dash(__name__)
# app.layout = html.Div([
#     html.H4('列を選択', style={'color': 'white'}),
#     dcc.Dropdown(
#         id='column-dropdown',
#         options=[{'label': col, 'value': col} for col in df.columns],
#         value=['A'],
#         multi=True
#     ),
#     html.H4('行を選択', style={'color': 'white'}),
#     dcc.Dropdown(
#         id='row-dropdown',
#         options=[{'label': i, 'value': i} for i in df.index],
#         value=list(df.index),
#         multi=True
#     ),
#     dcc.Graph(id='graph-output')
# ])
# @app.callback(
#     Output('graph-output', 'figure'),
#     [Input('column-dropdown', 'value'),
#      Input('row-dropdown', 'value')]
# )
# def update_graph(selected_columns, selected_rows):
#     if not selected_columns or not selected_rows:
#         return px.line()  # 列または行が選択されていない場合は空のグラフを返す

#     filtered_df = df.loc[selected_rows, selected_columns]
#     return px.line(filtered_df, x=filtered_df.index, y=filtered_df.columns)
# if __name__ == '__main__':
#     app.run_server(debug=True, host='127.0.0.1', port=8050)

import numpy as np
import plotly.graph_objects as go


def plot_history(matrix, name=None, lane=None, kda=None, start=None, save_to=None):
    """
    Function to plot the history of a player in a heatmap.

    Parameters:
        matrix (np.array): The history matrix to plot.
        name (np.array): The name of the player.
        lane (np.array): The lane of the player.
        kda (np.array): The KDA of the player.
        start (np.array): The starting item of the player.
        save_to (str): The path to save the plot.
    """

    name = name if name is not None else np.empty(matrix.shape, dtype=str)
    lane = lane if lane is not None else np.empty(matrix.shape, dtype=str)
    kda = kda if kda is not None else np.empty(matrix.shape, dtype=str)
    start = start if start is not None else np.empty(matrix.shape, dtype=str)

    heatmap = go.Heatmap(
        z=matrix.astype(float),
        customdata=np.dstack((name, matrix.astype(bool), lane, kda, start)),
        hovertemplate='<b>Summoner: %{customdata[0]}</b><br>'
                      'Win: %{customdata[1]}<br>'
                      'Lane: %{customdata[2]}<br>'
                      'KDA: %{customdata[3]:.2f} <br>'
                      'Start: %{customdata[4]}',
        colorscale=[[0, 'rgba(199, 21, 133, 0.8)'], [1, 'rgba(60, 179, 113, 0.8)']],
        name="History",
        showscale=False
    )

    # Create a Figure and update layout for a cleaner look
    fig = go.Figure(data=[heatmap])
    fig.update_layout(  # Hiding x-axis ticks
        yaxis=go.layout.YAxis(
            title='Individual players',
            showticklabels=False
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        width=600, height=800 / 3,
        xaxis_title=f'History of {matrix.shape[1]} games',
    )

    # Show plot
    fig.show()

    if save_to is not None:
        with open(save_to, "w") as f:
            f.write(fig.to_json())

    return fig


def plot_compare_plotly(comp_df, save_to=None):
    """
    Function to plot the comparison of models using Plotly. Equivalent to `arviz.plot_compare`.

    Parameters:
        comp_df (pd.DataFrame): The comparison DataFrame.
        save_to (str): The path to save the plot.
    """

    comp_df = comp_df.sort_index()
    
    plot_kwargs = {
        "color_ic": "black",
        "marker_ic": "circle",
        "marker_fc": "white",
        "color_dse": "grey",
        "marker_dse": "triangle-up",
        "ls_min_ic": "dash",
        "color_ls_min_ic": "grey"
    }
    
    color_criterion = 'rgba(0,176,246,1.)'
    color_difference = 'rgba(231,107,243,1.)'
    
    linewidth = 2
    information_criterion = 'elpd_loo'
    
    n_models = len(comp_df)
    yticks_pos = np.arange(n_models)[::-1] * -1.5  # Increased spacing between ticks
    labels = comp_df.index.tolist()
    
    # Create the figure
    fig = go.Figure()
    
    # Add the ELPD difference error bars
    diff_df = comp_df[comp_df['rank']>0]
    
    fig.add_trace(go.Scatter(
        x=diff_df[information_criterion],
        y=yticks_pos[[int(x[0]) for x in diff_df.index]] + 0.6,
        error_x=dict(
            type='data', 
            array=comp_df['dse'][1:], 
            thickness=linewidth
        ),
        mode='markers+text',
        marker=dict(
            color=color_difference, 
            symbol=plot_kwargs["marker_dse"], 
            size=10,
            line=dict(
                width=linewidth
            )
        ),
        name="ELPD difference"
    ))
    
    # Add the ELPD error bars
    fig.add_trace(go.Scatter(
        x=comp_df[information_criterion],
        y=yticks_pos,
        error_x=dict(
            type='data', 
            array=comp_df['se'], 
            thickness=linewidth
        ),
        mode='markers+text',
        marker=dict(
            color=color_criterion, 
            symbol=plot_kwargs["marker_ic"], 
            size=10, 
            line=dict(
                #color=plot_kwargs["marker_fc"], 
                width=linewidth
            )
        ),
        name="ELPD"
    ))
    
    # Add a vertical line
    fig.add_shape(
        type="line", 
        x0=comp_df[comp_df['rank']==0][information_criterion].iloc[0], 
        y0=min(yticks_pos) - 1, 
        x1=comp_df[comp_df['rank']==0][information_criterion].iloc[0], 
        y1=max(yticks_pos) + 1,
        line=dict(color=plot_kwargs["color_ls_min_ic"], width=linewidth, dash=plot_kwargs["ls_min_ic"]),
    )
    
    fig.update_xaxes(showgrid=True, minor=dict(showgrid=True))
    fig.update_yaxes(showgrid=True, minor=dict(showgrid=True))
    # Update axes properties
    fig.update_layout(
        xaxis_title='log(ELPD LOO) [higher is better]',
        #yaxis_title='Model',
        yaxis=dict(
            tickmode='array',
            tickvals=yticks_pos,
            ticktext=labels
        ),
        margin=dict(l=20, r=20, t=20, b=20),
        width=400, height=300,
        yaxis_autorange='reversed',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.show()

    if save_to is not None:
        with open(save_to, "w") as f:
            f.write(fig.to_json())

    return fig

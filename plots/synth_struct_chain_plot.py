# import os, os.path, sys, argparse
import pandas as pd
# import plotly
import plotly.express as px
import math

import plotly.io as pio
pio.kaleido.scope.mathjax = None

TIMEOUT = 300

# Function for checking if a string "s" is a number.
# taken from https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float#354073
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    df = pd.read_csv('../results/exp-eval.csv')
    df = df.loc[df['family'] == 'synth_struct_chain']
    df = df.replace('na', TIMEOUT)
    df['var_num'] = df['var_num'].astype(int)
    df['vi_time'] = df['vi_time'].astype(float)
    max_vars = df['var_num'].max()
    df = df.sort_values(by=['var_num'])
    fig = px.line(df, x='var_num', y='vi_time', color='solver', markers=True,
                  labels={
                      'var_num': 'Number of variables',
                      'vi_time': 'Time (s)',
                      'solver': ''
                  })

    fig.add_hline(y=TIMEOUT,
                  line_color="grey",
                  line_dash="dot",
                  )
    fig.add_annotation(y=math.log10(TIMEOUT), x=max_vars-0.1,
                       text="Timeout (" + str(TIMEOUT) + "s)",
                       showarrow=False,
                       yshift=10,
                       font_color="grey"
                       )

    fig.update_yaxes(
        type="log",
        dtick=1
    )

    fig.update_layout(legend=dict(
        yanchor="top",
        y=0.99,
        xanchor="left",
        x=0.01,
        title=''
    ), xaxis_tickformat=',d', margin=dict(l=0, r=0, b=0, t=0))

    fig.write_image('synth_struct_chain_plot.pdf')

# import os, os.path, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
import math

TIMEOUT = 600

# Function for checking if a string "s" is a number.
# taken from https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-float#354073
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def _get_synth_chain_data():
    df = pd.read_csv('../results/exp-eval.csv')
    df = df.loc[df['family'] == 'synth_struct_chain']

    df = df[df['var_num'] > 1]  # skip first because times are zero (not well pictured in log scale)
    df = df[df['var_num'] <= 8]  # skip >= 8 because all timeouts
    # max_vars = df['var_num'].max()
    max_vars = 8

    df = df.replace('na', TIMEOUT)
    df = df.sort_values(by=['var_num'])
    df['var_num'] = df['var_num'].astype(int)
    df['vi_time'] = df['vi_time'].astype(float)
    df['tot_time'] = df['tot_time'].astype(float)

    df_compilation = df.loc[df['solver'] == 'mapl-cirup']
    df_compilation = df_compilation[['run', 'solver', 'var_num', 'tot_time']]
    df_compilation = df_compilation.replace('mapl-cirup', 'maple-cirup (kc+vi)')
    df_compilation = df_compilation.groupby(['solver', 'var_num'], as_index=False).agg({'tot_time': ['mean', 'std']})
    df_compilation.columns = ['solver', 'var_num', 'time', 'std']

    df = df[['run', 'solver', 'var_num', 'vi_time']]
    df = df.replace('mapl-cirup', 'maple-cirup (vi)')
    df = df.groupby(['solver', 'var_num'], as_index=False).agg({'vi_time': ['mean', 'std']})
    df.columns = ['solver', 'var_num', 'time', 'std']

    df = pd.concat([df, df_compilation])
    return df


def _get_synth_cross_stitch_data():
    df = pd.read_csv('../results/exp-eval.csv')
    df = df.loc[df['family'] == 'synth_struct_cross_stitch']
    df = df[['run', 'solver', 'var_num', 'vi_time']]
    df = df.replace('na', TIMEOUT)
    df['var_num'] = df['var_num'].astype(int)
    df['vi_time'] = df['vi_time'].astype(float)
    # max_vars = df['var_num'].max()
    df = df.sort_values(by=['var_num'])

    df = df.groupby(['solver', 'var_num'], as_index=False).agg({'vi_time': ['mean', 'std']})
    df.columns = ['solver', 'var_num', 'time', 'std']
    return df


def create_plot(out_filename=None):
    color_spudd = "tab:red"
    color_mapl = "tab:blue"
    label_spudd = "SPUDD"
    label_mapl = "\\texttt{mapl-cirup}"
    #
    # get data
    #
    data_chain = _get_synth_chain_data()
    data_cross_stitch = _get_synth_cross_stitch_data()

    #
    # start double column plot
    #
    plt.style.use("./tex.mplstyle")
    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))
    #fig.suptitle(f"instance {index + 1}", fontsize=18)

    #
    # chain
    #
    data_spudd = data_cross_stitch[data_cross_stitch["solver"] == "spudd"]
    x = data_spudd["var_num"]
    y = data_spudd["time"]
    assert len(x) > 0
    axs[0].plot(x, y, color=color_spudd, marker=".", label=label_spudd)

    data_mapl = data_cross_stitch[data_cross_stitch["solver"] == "mapl-cirup"]
    x = data_mapl["var_num"]
    y = data_mapl["time"]
    assert len(x) > 0
    axs[0].plot(x, y, color=color_mapl, marker=".", label=label_mapl)

    axs[0].axhline(y=TIMEOUT, color="gray", linestyle="dashed")
    #axs[0].text(2, TIMEOUT+100, "timeout (600s)", rotation=0)

    axs[0].set_yscale('log')
    axs[0].set_title("cross-stitch")
    axs[0].set_ylabel("time (s)")
    axs[0].set_xlabel("number of variables")
    axs[0].grid(axis="y")

    #
    # cross-stitch
    #
    data_spudd = data_chain[data_chain["solver"] == "spudd"]
    x = data_spudd["var_num"]
    y = data_spudd["time"]
    assert len(x) > 0
    axs[1].plot(x, y, color=color_spudd, marker=".", label=label_spudd)

    data_mapl = data_chain[data_chain["solver"] == "maple-cirup (vi)"]
    x = data_mapl["var_num"]
    y = data_mapl["time"]
    assert len(x) > 0
    axs[1].plot(x, y, color=color_mapl, marker=".", label=label_mapl)

    axs[1].axhline(y=TIMEOUT, color="gray", linestyle="dashed")
    #axs[0].text(2, TIMEOUT+100, "timeout (600s)", rotation=0)

    axs[1].set_yscale('log')
    axs[1].set_title("chain")
    # axs[1].set_ylabel("time (s)")
    axs[1].set_xlabel("number of variables")
    axs[1].grid(axis="y")
    # axs[1].tick_params(axis='x', colors='white')
    # axs[1].legend()

    #
    # finish
    #
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fancybox=False, shadow=False)
    fig.tight_layout()
    if out_filename is None:
        plt.show()
    else:
        plt.savefig(out_filename, bbox_inches="tight")


if __name__ == "__main__":
    create_plot("./synth_struct_combined_plot.pdf")



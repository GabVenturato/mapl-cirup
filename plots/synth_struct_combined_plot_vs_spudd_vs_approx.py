# import os, os.path, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import math

TIMEOUT = 600
CHAIN_FAMILY = 'synth_struct_chain'
CROSS_STITCH_FAMILY = 'synth_struct_cross_stitch'


def _get_synth_chain_data():
    df = pd.read_csv('../results/exp-eval.csv')
    df = df.loc[df['family'] == 'synth_struct_chain']

    df = df[df['var_num'] > 1]  # skip first because times are zero (not well pictured in log scale)
    df = df[df['var_num'] <= 12]  # skip >= 8 because all timeouts
    # max_vars = df['var_num'].max()
    # max_vars = 8

    df.loc[df['vi_time'] == 'na', 'tot_time'] = 'na'
    df = df.replace('na', TIMEOUT)
    df = df.sort_values(by=['var_num'])
    df['var_num'] = df['var_num'].astype(int)
    df['vi_time'] = df['vi_time'].astype(float)
    df['tot_time'] = df['tot_time'].astype(float)
    df.loc[df['vi_time'] < TIMEOUT, 'tot_time'] =  df['tot_time'] - df['jit_time'].astype(float)  # Remove jit time

    df_compilation = df.loc[df['solver'] == 'mapl-cirup']
    df_compilation = df_compilation[['run', 'solver', 'var_num', 'tot_time']]
    df_compilation = df_compilation.replace('mapl-cirup', 'mapl-cirup (kc+vi)')
    df_compilation = df_compilation.groupby(['solver', 'var_num'], as_index=False).agg({'tot_time': ['mean', 'std']})
    df_compilation.columns = ['solver', 'var_num', 'time', 'std']

    # retrieve approx mapl-cirup
    df_compilation_approx = df.loc[df['solver'] == 'mapl-cirup-approx']
    df_compilation_approx = df_compilation_approx[['run', 'solver', 'var_num', 'tot_time']]
    df_compilation_approx = df_compilation_approx.replace('mapl-cirup-approx', 'mapl-cirup-approx (kc+vi)')
    df_compilation_approx = df_compilation_approx.groupby(['solver', 'var_num'], as_index=False).agg(
        {'tot_time': ['mean', 'std']})
    df_compilation_approx.columns = ['solver', 'var_num', 'time', 'std']

    df = df[['run', 'solver', 'var_num', 'vi_time']]
    df = df.replace('mapl-cirup', 'mapl-cirup (vi)')
    df = df.groupby(['solver', 'var_num'], as_index=False).agg({'vi_time': ['mean', 'std']})
    df.columns = ['solver', 'var_num', 'time', 'std']

    df = pd.concat([df, df_compilation, df_compilation_approx])
    return df


def _get_synth_cross_stitch_data():
    df = pd.read_csv('../results/exp-eval.csv')
    df = df.loc[df['family'] == 'synth_struct_cross_stitch']

    df.loc[df['vi_time'] == 'na', 'tot_time'] = 'na'
    df = df.replace('na', TIMEOUT)
    df = df.sort_values(by=['var_num'])
    df['var_num'] = df['var_num'].astype(int)
    df['vi_time'] = df['vi_time'].astype(float)
    df['tot_time'] = df['tot_time'].astype(float)
    df.loc[df['vi_time'] < TIMEOUT, 'tot_time'] = df['tot_time'] - df['jit_time'].astype(float)  # Remove jit time

    df_compilation = df.loc[df['solver'] == 'mapl-cirup']
    df_compilation = df_compilation[['run', 'solver', 'var_num', 'tot_time']]
    df_compilation = df_compilation.replace('mapl-cirup', 'mapl-cirup (kc+vi)')
    df_compilation = df_compilation.groupby(['solver', 'var_num'], as_index=False).agg({'tot_time': ['mean', 'std']})
    df_compilation.columns = ['solver', 'var_num', 'time', 'std']

    # retrieve approx mapl-cirup
    df_compilation_approx = df.loc[df['solver'] == 'mapl-cirup-approx']
    df_compilation_approx = df_compilation_approx[['run', 'solver', 'var_num', 'tot_time']]
    df_compilation_approx = df_compilation_approx.replace('mapl-cirup-approx', 'mapl-cirup-approx (kc+vi)')
    df_compilation_approx = df_compilation_approx.groupby(['solver', 'var_num'], as_index=False).agg(
        {'tot_time': ['mean', 'std']})
    df_compilation_approx.columns = ['solver', 'var_num', 'time', 'std']

    df = df[['run', 'solver', 'var_num', 'vi_time']]
    df = df.replace('mapl-cirup', 'mapl-cirup (vi)')
    df = df.groupby(['solver', 'var_num'], as_index=False).agg({'vi_time': ['mean', 'std']})
    df.columns = ['solver', 'var_num', 'time', 'std']

    df = pd.concat([df, df_compilation, df_compilation_approx])
    return df


def create_plot(out_filename=None):
    color_spudd = "tab:red"
    color_mapl_vi = "tab:blue"
    color_mapl_kcvi = "tab:green"
    color_mapl_approx_vi = "tab:purple"
    color_mapl_approx_kcvi = "tab:pink"
    label_spudd = "SPUDD"
    label_mapl_vi = "\\texttt{mapl-cirup} (VI)"
    label_mapl_kcvi = "\\texttt{mapl-cirup} (KC+VI)"
    label_mapl_approx_vi = "\\texttt{mapl-cirup}$_\\approx$ (VI)"
    label_mapl_approx_kcvi = "\\texttt{mapl-cirup}$_\\approx$ (KC+VI)"

    #
    # get data
    #
    data_chain = _get_synth_chain_data()
    data_cross_stitch = _get_synth_cross_stitch_data()

    #
    # start double column plot
    #
    plt.style.use("tex.mplstyle")
    fig, axs = plt.subplots(nrows=1, ncols=2, sharey='row')
    fig.set_figwidth(5)
    fig.set_figheight(3)
    #fig.suptitle(f"instance {index + 1}", fontsize=18)

    #
    # cross-stitch
    #
    data_spudd = data_cross_stitch[data_cross_stitch["solver"] == "spudd"]
    data_spudd = data_spudd.replace(0, 0.001)  # Can't be zero the time. SPUDD precision is too low.
    x = data_spudd["var_num"]
    y = data_spudd["time"]
    assert len(x) > 0
    axs[0].plot(x, y, color=color_spudd, marker=".", label=label_spudd)

    data_mapl_vi = data_cross_stitch[data_cross_stitch["solver"] == "mapl-cirup (vi)"]
    x = data_mapl_vi.loc[data_mapl_vi['var_num'] <= 8, 'var_num']
    y = data_mapl_vi.loc[data_mapl_vi['var_num'] <= 8, 'time']
    assert len(x) > 0
    axs[0].plot(x, y, color=color_mapl_vi, marker="^", markersize=4, label=label_mapl_vi)

    data_mapl_kcvi = data_cross_stitch[data_cross_stitch["solver"] == "mapl-cirup (kc+vi)"]
    x = data_mapl_kcvi["var_num"]
    y = data_mapl_kcvi["time"]
    assert len(x) > 0
    axs[0].plot(x, y, color=color_mapl_kcvi, marker="s", markersize=4, label=label_mapl_kcvi)

    data_mapl_vi_approx = data_cross_stitch[data_cross_stitch["solver"] == "mapl-cirup-approx"]
    x = data_mapl_vi_approx.loc[data_mapl_vi_approx['var_num'] <= 16, 'var_num']
    y = data_mapl_vi_approx.loc[data_mapl_vi_approx['var_num'] <= 16, 'time']
    assert len(x) > 0
    axs[0].plot(x, y, color=color_mapl_approx_vi, marker="*", markersize=6, label=label_mapl_approx_vi)

    data_mapl_approx_kcvi = data_cross_stitch[data_cross_stitch["solver"] == "mapl-cirup-approx (kc+vi)"]
    x = data_mapl_approx_kcvi["var_num"]
    y = data_mapl_approx_kcvi["time"]
    assert len(x) > 0
    axs[0].plot(x, y, color=color_mapl_approx_kcvi, marker="D", markersize=4, label=label_mapl_approx_kcvi)

    axs[0].axhline(y=TIMEOUT, color="gray", linestyle="dashed")
    #axs[0].text(2, TIMEOUT+100, "timeout (600s)", rotation=0)

    axs[0].set_yscale('log')
    axs[0].set_title("cross-stitch")
    axs[0].set_ylabel("time (s)")
    axs[0].set_xlabel("number of variables")
    axs[0].grid(axis="y")
    axs[0].xaxis.set_major_locator(MaxNLocator(integer=True))

    #
    # chain
    #
    data_mapl_vi = data_chain[data_chain["solver"] == "mapl-cirup (vi)"]
    x = data_mapl_vi.loc[data_mapl_vi['var_num'] <= 6, 'var_num']
    y = data_mapl_vi.loc[data_mapl_vi['var_num'] <= 6, 'time']
    assert len(x) > 0
    axs[1].plot(x, y, color=color_mapl_vi, marker="^", markersize=4, label=label_mapl_vi)

    data_mapl_vi_approx = data_chain[data_chain["solver"] == "mapl-cirup-approx"]
    x = data_mapl_vi_approx.loc[data_mapl_vi_approx['var_num'] <= 11, 'var_num']
    y = data_mapl_vi_approx.loc[data_mapl_vi_approx['var_num'] <= 11, 'time']
    assert len(x) > 0
    axs[1].plot(x, y, color=color_mapl_approx_vi, marker="*", markersize=6, label=label_mapl_approx_vi)

    data_mapl_kcvi = data_chain[data_chain["solver"] == "mapl-cirup (kc+vi)"]
    x = data_mapl_kcvi["var_num"]
    y = data_mapl_kcvi["time"]
    assert len(x) > 0
    axs[1].plot(x, y, color=color_mapl_kcvi, marker="s", markersize=4, label=label_mapl_kcvi)

    data_mapl_approx_kcvi = data_chain[data_chain["solver"] == "mapl-cirup-approx (kc+vi)"]
    x = data_mapl_approx_kcvi["var_num"]
    y = data_mapl_approx_kcvi["time"]
    assert len(x) > 0
    axs[1].plot(x, y, color=color_mapl_approx_kcvi, marker="D", markersize=4, label=label_mapl_approx_kcvi)

    data_spudd = data_chain[data_chain["solver"] == "spudd"]
    x = data_spudd["var_num"]
    y = data_spudd["time"]
    assert len(x) > 0
    axs[1].plot(x, y, color=color_spudd, marker=".", label=label_spudd)

    axs[1].axhline(y=TIMEOUT, color="gray", linestyle="dashed")
    #axs[0].text(2, TIMEOUT+100, "timeout (600s)", rotation=0)

    axs[1].set_yscale('log')
    axs[1].set_title("chain")
    # axs[1].set_ylabel("time (s)")
    axs[1].set_xlabel("number of variables")
    axs[1].grid(axis="y")
    axs[1].tick_params(left=False)
    axs[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    # axs[1].legend()
    axs[1].minorticks_off()

    #
    # finish
    #
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=3, frameon=False, fancybox=False, shadow=False)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.03)
    if out_filename is None:
        plt.show()
    else:
        plt.savefig(out_filename, bbox_inches="tight")


if __name__ == "__main__":
    create_plot("./synth_struct_combined_plot_vs_spudd_vs_approx.pdf")



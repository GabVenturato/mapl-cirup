# import os, os.path, sys, argparse
import pandas as pd
import matplotlib.pyplot as plt
import math

TIMEOUT = 800
CHAIN_FAMILY = 'synth_struct_chain'
CROSS_STITCH_FAMILY = 'synth_struct_cross_stitch'


def _get_synth_gpu_data(family):
    df_gpu = pd.read_csv('../results/exp-eval-gpu.csv')
    df_gpu = df_gpu.loc[df_gpu['family'] == family]

    df_gpu = df_gpu[df_gpu['var_num'] > 1]  # skip first because times are zero (not well pictured in log scale)
    if family == CHAIN_FAMILY:
        df_gpu = df_gpu[df_gpu['var_num'] <= 8]  # skip >= 8 because all timeouts

    df_gpu.loc[df_gpu['vi_time'] == 'na', 'tot_time'] = 'na'

    df_gpu = df_gpu.replace('na', TIMEOUT)
    df_gpu = df_gpu.sort_values(by=['var_num'])
    df_gpu['var_num'] = df_gpu['var_num'].astype(int)
    df_gpu['vi_time'] = df_gpu['vi_time'].astype(float)
    df_gpu['tot_time'] = df_gpu['tot_time'].astype(float)
    df_gpu = df_gpu[['var_num', 'vi_time', 'tot_time']]

    return df_gpu


def _get_synth_chain_data():
    df = pd.read_csv('../results/exp-eval.csv')
    df = df.loc[df['family'] == 'synth_struct_chain']

    df = df[df['var_num'] > 1]  # skip first because times are zero (not well pictured in log scale)
    df = df[df['var_num'] <= 8]  # skip >= 8 because all timeouts
    # max_vars = df['var_num'].max()
    # max_vars = 8

    df.loc[df['vi_time'] == 'na', 'tot_time'] = 'na'
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

    df.loc[df['vi_time'] == 'na', 'tot_time'] = 'na'
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


def create_plot(out_filename=None):
    color_mapl_vi = "tab:blue"
    color_mapl_kcvi = "tab:green"
    color_mapl_gpu_vi = "tab:purple"
    color_mapl_gpu_tot = "tab:pink"
    label_mapl_vi = "VI"
    label_mapl_kcvi = "KC+VI"
    label_mapl_gpu_vi = "GPU, VI"
    label_mapl_gpu_tot = "GPU, tot"

    #
    # get data
    #
    data_chain = _get_synth_chain_data()
    data_cross_stitch = _get_synth_cross_stitch_data()
    data_gpu_chain = _get_synth_gpu_data(CHAIN_FAMILY)
    data_gpu_cross_stitch = _get_synth_gpu_data(CROSS_STITCH_FAMILY)

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
    data_mapl_vi = data_cross_stitch[data_cross_stitch["solver"] == "maple-cirup (vi)"]
    x = data_mapl_vi["var_num"]
    y = data_mapl_vi["time"]
    assert len(x) > 0
    axs[0].plot(x, y, color=color_mapl_vi, marker=".", label=label_mapl_vi)

    data_mapl_kcvi = data_cross_stitch[data_cross_stitch["solver"] == "maple-cirup (kc+vi)"]
    x = data_mapl_kcvi["var_num"]
    y = data_mapl_kcvi["time"]
    assert len(x) > 0
    axs[0].plot(x, y, color=color_mapl_kcvi, marker=".", label=label_mapl_kcvi)

    x = data_gpu_cross_stitch.loc[data_gpu_cross_stitch['var_num'] <= 7, 'var_num']
    y = data_gpu_cross_stitch.loc[data_gpu_cross_stitch['var_num'] <= 7, 'vi_time']
    assert len(x) > 0
    axs[0].plot(x, y, color=color_mapl_gpu_vi, marker=".", label=label_mapl_gpu_vi)

    x = data_gpu_cross_stitch["var_num"]
    y = data_gpu_cross_stitch["tot_time"]
    assert len(x) > 0
    axs[0].plot(x, y, color=color_mapl_gpu_tot, marker=".", label=label_mapl_gpu_tot)

    axs[0].axhline(y=TIMEOUT, color="gray", linestyle="dashed")
    #axs[0].text(2, TIMEOUT+100, "timeout (600s)", rotation=0)

    axs[0].set_yscale('log')
    axs[0].set_title("cross-stitch")
    axs[0].set_ylabel("time (s)")
    axs[0].set_xlabel("number of variables")
    axs[0].grid(axis="y")

    #
    # chain
    #
    data_mapl_vi = data_chain[data_chain["solver"] == "maple-cirup (vi)"]
    x = data_mapl_vi["var_num"]
    y = data_mapl_vi["time"]
    assert len(x) > 0
    axs[1].plot(x, y, color=color_mapl_vi, marker=".", label=label_mapl_vi)

    data_mapl_kcvi = data_chain[data_chain["solver"] == "maple-cirup (kc+vi)"]
    x = data_mapl_kcvi["var_num"]
    y = data_mapl_kcvi["time"]
    assert len(x) > 0
    axs[1].plot(x, y, color=color_mapl_kcvi, marker=".", label=label_mapl_kcvi)

    x = data_gpu_chain.loc[data_gpu_chain['var_num'] <= 6, 'var_num']
    y = data_gpu_chain.loc[data_gpu_chain['var_num'] <= 6, 'vi_time']
    assert len(x) > 0
    axs[1].plot(x, y, color=color_mapl_gpu_vi, marker=".", label=label_mapl_gpu_vi)

    x = data_gpu_chain["var_num"]
    y = data_gpu_chain["tot_time"]
    assert len(x) > 0
    axs[1].plot(x, y, color=color_mapl_gpu_tot, marker=".", label=label_mapl_gpu_tot)

    axs[1].axhline(y=TIMEOUT, color="gray", linestyle="dashed")
    #axs[0].text(2, TIMEOUT+100, "timeout (600s)", rotation=0)

    axs[1].set_yscale('log')
    axs[1].set_title("chain")
    # axs[1].set_ylabel("time (s)")
    axs[1].set_xlabel("number of variables")
    axs[1].grid(axis="y")
    axs[1].tick_params(left=False)
    # axs[1].legend()
    axs[1].minorticks_off()

    #
    # finish
    #
    handles, labels = axs[1].get_legend_handles_labels()
    fig.legend(handles, labels, bbox_to_anchor=(0.5, 0), loc='upper center', ncol=4, frameon=False, fancybox=False, shadow=False)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.03)
    if out_filename is None:
        plt.show()
    else:
        plt.savefig(out_filename, bbox_inches="tight")


if __name__ == "__main__":
    create_plot("./synth_struct_combined_plot_gpu.pdf")



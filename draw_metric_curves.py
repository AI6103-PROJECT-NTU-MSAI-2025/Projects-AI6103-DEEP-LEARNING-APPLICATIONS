import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import sys
import os
from cycler import cycler


def load_metric_data(file_path, selected_metrics):
    """Loads metrics from a single event file."""
    data_frames = {}
    last_values = {}

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"can't find file: {file_path}")

    acc = EventAccumulator(file_path, size_guidance={'scalars': 0})
    acc.Reload()
    scalar_tags = acc.Tags()['scalars']

    for tag in selected_metrics:
        if tag in scalar_tags:
            events = acc.Scalars(tag)
            df = pd.DataFrame(events, columns=['wall_time', 'step', 'value'])
            data_frames[tag] = df

            if not df.empty:
                last_values[tag] = df['value'].iloc[-1]
            else:
                last_values[tag] = None
        else:
            last_values[tag] = None

    return data_frames, last_values

def draw_curves_and_results(EXP_PATHS,OUTPUT_DIR,SMOOTHING_WINDOW,selected_metrics,metrics_title_dic,COLOR_LIST):
    all_exp_data = {}
    all_last_values = {}

    try:
        for name, path in EXP_PATHS.items():
            print(f"loading data from exps: {name}")
            dfs, lasts = load_metric_data(path, selected_metrics)
            all_exp_data[name] = dfs
            all_last_values[name] = lasts

    except FileNotFoundError as e:
        print(f"\nerror：{e}")
        print("please input the correct file path")
        sys.exit(1)
    except Exception as e:
        print(f"\nerror: {e}")
        sys.exit(1)

    print("\n--- generating metric curves ---")
    for metric in selected_metrics:
        title_name = metrics_title_dic.get(metric, metric)
        plt.figure(figsize=(10, 6))
        plt.title(f'Metric Comparison: {title_name}')
        plt.xlabel('Step')
        plt.ylabel('Value')

        is_loss_metric = (metric == 'losses/ar_loss')

        for i, (name, dfs) in enumerate(all_exp_data.items()):
            if metric in dfs and dfs[metric] is not None and not dfs[metric].empty:
                df = dfs[metric]
                current_color = COLOR_LIST[i % len(COLOR_LIST)]
                if is_loss_metric:
                    ''' #print original value
                    plt.plot(df['step'], df['value'],
                             label=f'{name} (Raw)',
                             color=current_color,
                             alpha=0.2,  # 透明度
                             linestyle='--')  # 虚线
                    '''
                    plot_values = df['value'].rolling(window=SMOOTHING_WINDOW).mean()
                    plot_df = pd.DataFrame({'step': df['step'], 'value': plot_values}).dropna()
                    plot_values = plot_df['value']
                    plot_steps = plot_df['step']

                    label_suffix = f" (Smoothed w={SMOOTHING_WINDOW})"
                    plt.plot(plot_steps, plot_values,
                             label=f'{name}{label_suffix}',
                             color=current_color,
                             linewidth=2)
                else:
                    plt.plot(df['step'], df['value'], label=f'{name}')

        plt.legend()
        plt.grid(True)

        filename = os.path.join(OUTPUT_DIR, f"{metric.replace('/', '_')}_comparison.png")
        plt.savefig(filename)
        print(f"Saved plot: {filename}")
        # plt.show()
    print("\n--- Final (last) values comparison ---")
    comparison_df = pd.DataFrame(all_last_values).T
    comparison_df_formatted = comparison_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else str(x))
    comparison_df_formatted.to_csv(OUTPUT_DIR+"/test_results.csv")
    print(comparison_df_formatted)


if __name__ == "__main__":


    EXP_PATHS_512b_wd_lr_scheduled = {
        "Sum": "exps/amzn23_office-l50-wd_lrd/HSTU-b4-h4-dqk16-dv16-lsilud0.5-ad0.0_blair_DotProduct_local_text-l2-eps1e-06_ssl-t0.05-n512-b512-lr0.001-cosmin1e-05-wu100-wd0.001-2025-11-23-fe5/events.out.tfevents.1763868888.x1000c0s4b0n1.2154623.0",
        "Weighted_sum":"exps/amzn23_office-l50-wd_lrd/amzn23_office-I50_weighted_sum_wd_lr/events.out.tfevents.1763833030.I252664a0bd0060187d.1980.0",
        "Gated": "exps/amzn23_office-l50-wd_lrd/HSTU-b4-h4-dqk16-dv16-lsilud0.5-ad0.0_gated_DotProduct_local_text-l2-eps1e-06_ssl-t0.05-n512-b512-lr0.001-cosmin1e-05-wu100-wd0.001-2025-11-23-fe5/events.out.tfevents.1763829228.x1000c0s4b0n1.1341390.0",
        "Resnet": "exps/amzn23_office-l50-wd_lrd/HSTU-b4-h4-dqk16-dv16-lsilud0.5-ad0.0_resnet_mlp_DotProduct_local_text-l2-eps1e-06_ssl-t0.05-n512-b512-lr0.001-cosmin1e-05-wu100-wd0.001-2025-11-22-fe5/events.out.tfevents.1763826890.x1000c0s2b0n1.1277715.0",
        "FFNN": "exps/amzn23_office-l50-wd_lrd/HSTU-b4-h4-dqk16-dv16-lsilud0.5-ad0.0_concat_DotProduct_local_text-l2-eps1e-06_ssl-t0.05-n512-b512-lr0.001-cosmin1e-05-wu100-wd0.001-2025-11-23-fe5/events.out.tfevents.1763873554.x1000c0s4b0n1.2256998.0",
        "Baseline": "exps/amzn23_office-l50-wd_lrd/HSTU-b4-h4-dqk16-dv16-lsilud0.5-ad0.0_baseline_DotProduct_local-l2-eps1e-06_ssl-t0.05-n512-b512-lr0.001-cosmin1e-05-wu100-wd0.001-2025-11-23-fe5/events.out.tfevents.1763828989.x1000c0s4b0n1.1336929.0",
    }

    EXP_PATHS_128b_wd = {
        "Sum": "exps/amzn23_office-l50-wd_only/amzn23_office-I50_sum_wd/events.out.tfevents.1763734644.autodl-container-828249b609-a2deb705.36302.0",
        "Weighted_sum":"exps/amzn23_office-l50-wd_only/amzn23_office-I50_weighted_sum_wd/events.out.tfevents.1763805572.I25292188ad00b01dab.2529.0",
        "Gated": "exps/amzn23_office-l50-wd_only/amzn_23_office-I50_gated_wd/events.out.tfevents.1763705763.autodl-container-828249b609-a2deb705.1392.0",
        "Resnet": "exps/amzn23_office-l50-wd_only/amzn_23_office-I50_resnet_wd/events.out.tfevents.1763711526.x1000c0s1b0n1.3606124.0",
        "FFNN": "exps/amzn23_office-l50-wd_only/amzn23_office-I50_concat_mlp_wd/events.out.tfevents.1763753549.I252664a0bd0060187d.13027.0",
        "Baseline": "exps/amzn23_office-l50-wd_only/amzn_23_office-I50_None_wd/events.out.tfevents.1763788179.x1000c0s2b0n1.684447.0",
    }

    OUTPUT_DIR_128b_wd = './RESULTS_128b_wd'
    OUTPUT_DIR_512b_wd_lrs = './RESULTS_512b_wd_lrs'

    SMOOTHING_WINDOW = 100
    selected_metrics = ['losses/ar_loss', 'eval_epoch_full/ndcg@10', 'eval_epoch_full/ndcg@50',
                        'eval_epoch_full/hr@10', 'eval_epoch_full/hr@50', 'eval_epoch_full/mrr']

    metrics_title_dic = {
    'losses/ar_loss': 'Autoregressive Loss',
    'eval_epoch_full/ndcg@10': 'NDCG@10',
    'eval_epoch_full/ndcg@50': 'NDCG@50',
    'eval_epoch_full/hr@10': 'HR@10',
    'eval_epoch_full/hr@50': 'HR@50',
    'eval_epoch_full/mrr': 'MRR',
    }

    my_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                 '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#4e79a7', '#f28e2b']

    draw_curves_and_results(EXP_PATHS_128b_wd, OUTPUT_DIR_128b_wd, SMOOTHING_WINDOW, selected_metrics, metrics_title_dic,my_colors)
    draw_curves_and_results(EXP_PATHS_512b_wd_lr_scheduled, OUTPUT_DIR_512b_wd_lrs, SMOOTHING_WINDOW, selected_metrics, metrics_title_dic,my_colors)
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
import sys
import os




EXP_PATHS = {
    "Sum": "exps/amzn23_office-I50_sum_wd/events.out.tfevents.1763734644.autodl-container-828249b609-a2deb705.36302.0",
    "Gated": "exps/amzn_23_office-I50_gated_wd/events.out.tfevents.1763705763.autodl-container-828249b609-a2deb705.1392.0",
    "Resnet": "exps/amzn_23_office-I50_resnet_wd/events.out.tfevents.1763711526.x1000c0s1b0n1.3606124.0",
    "FFN": "exps/amzn23_office-I50_concat_mlp_wd/events.out.tfevents.1763753549.I252664a0bd0060187d.13027.0",
    "baseline": "exps/amzn_23_office-I50_None_wd/events.out.tfevents.1763788179.x1000c0s2b0n1.684447.0",
}

OUTPUT_DIR='./RESULTS'
SMOOTHING_WINDOW = 200
selected_metrics = ['losses/ar_loss', 'eval_epoch_full/ndcg@10', 'eval_epoch_full/ndcg@50',
                    'eval_epoch_full/hr@10', 'eval_epoch_full/hr@50', 'eval_epoch_full/mrr']


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
    plt.figure(figsize=(10, 6))
    plt.title(f'Comparison: {metric}')
    plt.xlabel('Step')
    plt.ylabel('Value')

    is_loss_metric = (metric == 'losses/ar_loss')

    for name, dfs in all_exp_data.items():
        if metric in dfs and dfs[metric] is not None and not dfs[metric].empty:
            df = dfs[metric]

            current_color = plt.gca()._get_lines.get_next_color()
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

    #plt.show()

print("\n--- Final (last) values comparison ---")

comparison_df = pd.DataFrame(all_last_values).T
comparison_df_formatted = comparison_df.applymap(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else str(x))
comparison_df_formatted.to_csv("RESULTS/test_results.csv")
print(comparison_df_formatted)
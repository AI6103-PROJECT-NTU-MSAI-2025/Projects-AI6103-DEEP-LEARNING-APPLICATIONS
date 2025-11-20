import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# put event file here
event_file_path = "exps/amzn23_game-l50/HSTU-b4-h4-dqk16-dv16-lsilud0.5-ad0.0_blair_DotProduct_local_text-l2-eps1e-06_ssl-t0.05-n512-b128-lr0.001-wu0-wd0-2025-11-18-fe5/events.out.tfevents.1763435129.autodl-container-a0244cade5-e13101f2.1395.0"
# select metrics you want to draw
selected_metrics=['eval_epoch_full/ndcg@10','eval_epoch_full/ndcg@50','eval_epoch_full/hr@100','eval_epoch_full/hr@50','eval_epoch_full/mmr']

acc = EventAccumulator(event_file_path, size_guidance={'scalars': 0})
acc.Reload()
scalar_tags = acc.Tags()['scalars']
print(f"found metrics in training log: {scalar_tags}")

plt.figure(figsize=(15, 5 * len(scalar_tags)))
data_frames = {}

for i, tag in enumerate(scalar_tags):
    # draw loss curve for selected metrics
    if tag in selected_metrics:
        events = acc.Scalars(tag)
        # 'step' :x-axis
        # 'value' :y-axis
        # 'wall_time' :time-stamp
        df = pd.DataFrame(events, columns=['wall_time', 'step', 'value'])
        data_frames[tag] = df

        plt.plot(df['step'], df['value'])
        plt.title(f'metric: {tag}')
        plt.xlabel('Step')
        plt.ylabel('Value')
        plt.grid(True)
        plt.show()




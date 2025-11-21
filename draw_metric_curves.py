import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt

# put event file here
event_file_path = "exps/amzn_23_office-I50_gated/events.out.tfevents.1763637455.autodl-container-a0244cade5-e13101f2.3543.0"
# select metrics you want to draw
selected_metrics=['losses/ar_loss','eval_epoch_full/ndcg@10','eval_epoch_full/ndcg@50','eval_epoch_full/hr@100','eval_epoch_full/hr@50','eval_epoch_full/mrr']

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




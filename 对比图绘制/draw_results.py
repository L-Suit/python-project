import os
import pandas as pd
import matplotlib.pyplot as plt





def plot_metrics_and_loss(experiment_names, metrics_info, loss_info, metrics_subplot_layout, loss_subplot_layout,
                          base_directory,metrics_figure_size=(20, 10), loss_figure_size=(20, 10)):
    # Plot metrics
    plt.figure(figsize=metrics_figure_size)
    for i, (metric_name, title) in enumerate(metrics_info):
        plt.subplot(*metrics_subplot_layout, i + 1)
        for name in experiment_names:
            file_path = os.path.join(base_directory, name, 'results.csv')
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == metric_name][0]
            plt.plot(data[column_name], label=name)
        plt.xlabel('Epoch',fontsize=22)
        plt.title(title,fontsize=22)
        plt.legend(fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tight_layout()
    metrics_filename = 'metrics_curves.png'
    plt.savefig(metrics_filename,dpi=400)
    # plt.show()

    # Plot loss
    plt.figure(figsize=loss_figure_size)
    for i, (loss_name, title) in enumerate(loss_info):
        plt.subplot(*loss_subplot_layout, i + 1)
        for name in experiment_names:
            file_path = os.path.join(base_directory, name, 'results.csv')
            data = pd.read_csv(file_path)
            column_name = [col for col in data.columns if col.strip() == loss_name][0]
            plt.plot(data[column_name], label=name)
        plt.xlabel('Epoch',fontsize=22)
        plt.title(title,fontsize=22)
        plt.legend(fontsize=14)
        plt.tick_params(axis='both', which='major', labelsize=17)
    plt.tight_layout()
    loss_filename = 'loss_curves.png'
    plt.savefig(loss_filename,dpi=400)
    # plt.show()

    return metrics_filename, loss_filename



if __name__ == '__main__':
    base_directory = 'D:/实验室/小论文/实验数据'

    # Metrics to plot
    metrics_info = [
        # ('metrics/precision(B)', 'Precision'),
        # ('metrics/recall(B)', 'Recall'),
        ('metrics/mAP50(B)', 'mAP at IoU=0.5'),
        ('metrics/mAP50-95(B)', 'mAP for IoU Range 0.5-0.95')
    ]

    # Loss to plot
    loss_info = [
        # ('train/box_loss', 'Training Box Loss'),
        # ('train/cls_loss', 'Training Classification Loss'),
        # ('train/dfl_loss', 'Training DFL Loss'),
        ('val/box_loss', 'Validation Box Loss'),
        ('val/cls_loss', 'Validation Classification Loss'),
        # ('val/dfl_loss', 'Validation DFL Loss')
    ]

    experiment_names = ['PCSNet',
                        'yolov8n',
                        'yolov10n',
                        'yolov11n',
                         'rtdetr-l',
                        'faster-rcnn_r50',
                        'dynamic-rcnn_r50',
                        ]



    # Plot the metrics and loss from multiple experiments
    metrics_filename, loss_filename = plot_metrics_and_loss(
        experiment_names=experiment_names,
        metrics_info=metrics_info,
        loss_info=loss_info,
        metrics_subplot_layout=(1, 2),
        loss_subplot_layout=(1, 2),
        base_directory=base_directory
    )
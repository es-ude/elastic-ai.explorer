from elasticai.explorer.knowledge_repository import KnowledgeRepository, Metrics
import matplotlib.pyplot as plt
import numpy as np
from settings import ROOT_DIR
from scipy.stats import kendalltau
def compute_kandell(list_x: list, list_y: list) -> float:
    """Computes Kandell Correlation Coefficient.

    Args:
        list_x (list): _description_
        list_y (list): _description_

    Returns:
        float: _description_
    """
        

    # Taking values from the above example in Lists
    rank_x = [sorted(list_x).index(x) for x in list_x]
    rank_y = [sorted(list_y).index(x) for x in list_y]

    # Calculating Kendall Rank correlation
    corr, _ = kendalltau(rank_x, rank_y)
    print('Kendall Rank correlation: %.5f' % corr)

    return corr

CONTEXT_PATH = ROOT_DIR / "plots"
class Visualizer:

    def __init__(self, metrics: Metrics):
        self.data = metrics.structured_est_metrics
        self.labels = metrics.structured_samples
        self.metrics = metrics
    def plot_all_results(self, figure_size = [15, 20], filename = None):
        plt.figure()
        fig = plt.figure(num=1, clear=True)
        fig.set_size_inches(figure_size[0], figure_size[1], forward=True)
        axes = []
        axes.append(fig.add_subplot(311))
        axes.append(fig.add_subplot(312))
        axes.append(fig.add_subplot(313))
        bar_width = 0.2
        

        #get Kendall for estimated and measured Accuracies
        kendall_coef_accuracy = compute_kandell(self.metrics.structured_est_accuracies, self.metrics.raw_measured_accuracies)
        #get Kendall for estimated FLOPs and measured latencies
        kendall_coef_latencies = compute_kandell(self.metrics.structured_est_flops, self.metrics.raw_measured_latencies)
        #get Kendall for Combined Metric and measured Accuracies
        kendall_coef_combined = compute_kandell(self.metrics.structured_est_combined, self.metrics.raw_measured_latencies)

        indices = np.arange(0, len(self.data[0][0][:]), 1)

        #Accuracy Estimate vs Accuracy on Pi
        axes[0].bar(x=indices, height = self.data[0][0][:], width = bar_width,label= "Estimated Accuracy in %")
        axes[0].bar(x=indices+bar_width, height = self.data[0][1][:], width = bar_width,label= "Measured Accuracy in %")
        axes[0].set_xlabel("Sample")
        axes[0].set_ylabel("Accuracy in %")
        axes[0].set_title(f"Accuracy Estimation vs Measured Accuracy: Kendall's tau = {kendall_coef_accuracy}")

        #FLOPS Proxy vs Latency on Pi
        axes[1].bar(x=indices, height = self.data[1][0][:], width = bar_width,label= "FLOPs Estimation in Log10")
        axes[1].bar(x=indices+bar_width, height = self.data[1][1][:], width = bar_width,label= "Latency in Mircosec.")
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Number FlOPs log10 and Latency in ms")
        axes[1].set_title(f"Latency Proxy-Estimation vs Measured Latency: Kendall's tau = {kendall_coef_latencies}")

        #Combined Metric, accuracy-estimation and flops estimation
        axes[2].bar(x=indices, height = self.data[2][0][:], width = bar_width, label= "Combined Metric")
        axes[2].bar(x=indices+bar_width, height = self.data[2][1][:], width = bar_width,label= "Accuracy")
        axes[2].set_xlabel("Sample")
        axes[2].set_ylabel("Accuracy and FLOPs")
        axes[2].set_title(f"Combined Estimation vs Measured Accuracy: Kendall's tau = {kendall_coef_combined}")

        for ax in axes:
            ax.legend(loc = "upper left")
            ax.set_xticks(indices + bar_width / 2)
            ax.set_xticklabels(self.labels)


        if filename:
            plt.savefig(str(CONTEXT_PATH) + "/" + filename + ".png")
        else:
            plt.show()





            

        
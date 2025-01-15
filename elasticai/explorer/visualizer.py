from elasticai.explorer.knowledge_repository import KnowledgeRepository, Metrics
import matplotlib.pyplot as plt
import numpy as np
from settings import ROOT_DIR

CONTEXT_PATH = ROOT_DIR / "plots"
class Visualizer:

    def __init__(self, metrics: Metrics):
        self.data = metrics.structured_metrics
        self.labels = metrics.structured_samples

    def plot_all_results(self, figure_size = [15, 20], filename = None):
        plt.figure()
        fig = plt.figure(num=1, clear=True)
        fig.set_size_inches(figure_size[0], figure_size[1], forward=True)
        axes = []
        axes.append(fig.add_subplot(311))
        axes.append(fig.add_subplot(312))
        axes.append(fig.add_subplot(313))
        bar_width = 0.35
        
        indices = np.arange(0, len(self.data[0][0][:]), 1)

        #Accuracy Estimate vs Accuracy on Pi
        axes[0].bar(x=indices, height = self.data[0][0][:], width = bar_width,label= "Estimated Accuracy in %")
        axes[0].bar(x=indices+bar_width, height = self.data[0][1][:], width = bar_width,label= "Measured Accuracy in %")
        axes[0].set_xlabel("Sample")
        axes[0].set_ylabel("Accuracy in %")
        axes[0].set_title("Accuracy Estimation vs Measured Accuracy")

        #FLOPS Proxy vs Latency on Pi
        axes[1].bar(x=indices, height = self.data[1][0][:], width = bar_width,label= "FLOPs Estimation in Log10")
        axes[1].bar(x=indices+bar_width, height = self.data[1][1][:], width = bar_width,label= "Latency in Mircosec.")
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel("Number FlOPs log10 and Latency in ms")
        axes[1].set_title("Latency Proxy-Estimation vs Measured Latency")

        #Combined Metric, accuracy-estimation and flops estimation
        axes[2].bar(x=indices, height = self.data[2][0][:], width = bar_width, label= "Combined Metric")
        axes[2].bar(x=indices+bar_width, height = self.data[2][1][:], width = bar_width,label= "Accuracy")
        axes[2].set_xlabel("Sample")
        axes[2].set_ylabel("Accuracy and FLOPs")
        axes[2].set_title("Combined Estimation vs Measured Accuracy")

        for ax in axes:
            ax.legend(loc = "upper left")
            ax.set_xticks(indices + bar_width / 2)
            ax.set_xticklabels(self.labels)


        if filename:
            plt.savefig(str(CONTEXT_PATH) + "/" + filename + ".png")
        else:
            plt.show()





            

        
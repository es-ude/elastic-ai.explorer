from elasticai.explorer.knowledge_repository import KnowledgeRepository, SearchMetrics
import matplotlib.pyplot as plt
import numpy as np
from settings import ROOT_DIR

CONTEXT_PATH = ROOT_DIR / "plots"
class Visualizer:

    def __init__(self, metrics: SearchMetrics):
        self.data = metrics.structured_metrics
        self.index = metrics.sample_list

    def plot_all_results(self, figure_size = [15, 20], filename = None):
        plt.figure()
        fig = plt.figure(num=1, clear=True)
        fig.set_size_inches(figure_size[0], figure_size[1], forward=True)
    
        ax1=fig.add_subplot(311)
        ax2=fig.add_subplot(312)
        ax3=fig.add_subplot(313)
        bar_width = 0.35
        
        indices = np.arange(0, len(self.data[0][0][:]), 1)

        #Accuracy Estimate vs Accuracy on Pi
        ax1.bar(x=indices, height = self.data[0][0][:], width = bar_width,label= "Estimated Accuracy in %")
        ax1.bar(x=indices+bar_width, height = self.data[0][1][:], width = bar_width,label= "Measured Accuracy in %")

        #FLOPS Proxy vs Latency on Pi
        ax2.bar(x=indices, height = self.data[1][0][:], width = bar_width,label= "FLOPs Estimation in Log10")
        ax2.bar(x=indices+bar_width, height = self.data[1][1][:], width = bar_width,label= "Latency in Mircosec.")

        #Combined Metric, accuracy-estimation and flops estimation
        ax3.bar(x=indices, height = self.data[2][0][:], width = bar_width, label= "Combined Metric")
        ax3.bar(x=indices+bar_width, height = self.data[2][1][:], width = bar_width,label= "Accuracy")


        ax1.legend(loc = "upper left")
        ax2.legend(loc = "upper left")
        ax3.legend(loc = "upper left")
        
        if filename:
            plt.savefig(str(CONTEXT_PATH) + "/" + filename + ".png")
        else:
            plt.show()





            

        
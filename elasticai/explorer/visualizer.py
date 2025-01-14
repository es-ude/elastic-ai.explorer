from elasticai.explorer.knowledge_repository import KnowledgeRepository
import matplotlib.pyplot as plt
import numpy as np

class Visualizer:

    def __init__(self):
        pass

    def plot_all_results(self, data, figure_size = [15, 20]):
        plt.figure()
        fig = plt.figure(num=1, clear=True)
        fig.set_size_inches(figure_size[0], figure_size[1], forward=True)
    
        ax1=fig.add_subplot(111)
        ax2=fig.add_subplot(212)
        ax3=fig.add_subplot(313)

        np_data= np.array(data)
        indices = np.arange(0, len(data[:][0]), 1)

        ax1.scatter(x=indices, y = np_data[:][0], s=10, label= "Accuracy in %")
        ax2.scatter(x=indices, y = np_data[:][1], s=10, label= "Latenzy in Mircosec.")
        ax3.scatter(x=indices, y = np_data[:][2], s=10, label= "Combined Metric")
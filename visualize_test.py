from elasticai.explorer import visualizer
import numpy as np
if __name__ == "__main__":
    data = np.reshape(np.arange(0,30,1), [3,2,5])
    print(data)

    visu = visualizer.Visualizer(data)
    visu.plot_all_results(filename="plot")
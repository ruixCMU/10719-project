import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

if __name__ == "__main__":
    # Data 
    beta = 1000
    sizes = np.round(np.random.dirichlet([beta] * 5) * 100, 1)
    labels = ['A', 'B', 'C', 'D', 'E']
    colors = ['red', 'green', 'blue', 'yellow', 'purple']

    # Draw Pi Chart
    patches, _, autotexts = plt.pie(sizes, labels=labels, colors=colors,
                                    autopct='', startangle=90)

    # Customized Label, Proportioanlly 
    legend_labels = [f'{label} ({size}%)' for label, size in zip(labels, sizes)]
    legend_handles = [Patch(color=color) for color in colors]

    plt.title(fr"$\beta$ = {beta}")

    # Adding Diagram
    plt.legend(legend_handles, legend_labels, loc="upper right", bbox_to_anchor=(0.15, 1))

    # Set the axis ratio
    plt.axis('equal')

    # Show plot
    plt.show()

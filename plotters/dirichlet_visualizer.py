import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

if __name__ == "__main__":
    # 数据
    beta = 1000
    sizes = np.round(np.random.dirichlet([beta] * 5) * 100, 1)
    labels = ['A', 'B', 'C', 'D', 'E']
    colors = ['red', 'green', 'blue', 'yellow', 'purple']

    # 画扇形图
    patches, _, autotexts = plt.pie(sizes, labels=labels, colors=colors,
                                    autopct='', startangle=90)

    # 创建自定义标签，显示比例
    legend_labels = [f'{label} ({size}%)' for label, size in zip(labels, sizes)]
    legend_handles = [Patch(color=color) for color in colors]

    plt.title(fr"$\beta$ = {beta}")

    # 添加图例
    plt.legend(legend_handles, legend_labels, loc="upper right", bbox_to_anchor=(0.15, 1))

    # 设置图形的纵横比例相等
    plt.axis('equal')

    # 显示图形
    plt.show()
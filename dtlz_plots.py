from pymoo.visualization.scatter import Scatter
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
import matplotlib.pyplot as plt

def main():
    ref_dirs = get_reference_directions("das-dennis", 3, n_partitions=12)

    pf = get_problem("dtlz7").pareto_front()

    plot = Scatter(angle=(45,45)).add(pf).show()
    plot.add(pf)

    # Adjust layout before saving the plot
    plt.tight_layout()  # Adjusts the plot layout to prevent clipping of labels

    # Show the plot in SciView
    plot.show()

    # Save the plot with higher DPI to ensure better resolution
    plt.savefig(f"dtlz_plots/dtlz7_high_res.png", dpi=300)  # Save as PNG with high resolution

if __name__ == '__main__':
    main()

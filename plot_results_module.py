import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# Define data from the table
cities = ["City 1 (9 lmks)", "City 2 (34 lmks)", "City 3 (44 lmks)", "City 4 (175 lmks)",
        #   "Average"
]

# modules = [
#         "SRER",
#         "REG Top-1",
#         "REG Top-5",
#         "REG Top-10",
#         "SPG",
#         "LT Finetuned T5-base",
#         "LT RAG-10",
#         "LT RAG-50",
#         "LT RAG-100"
# ]

# accuracy_values = np.array([
#     [99.45, 99.43, 99.56, 99.39],  # SRER
#     [99.68, 97.98, 88.74, 78.35],  # REG Top-1
#     [100.00, 100.00, 99.56, 99.15], # REG Top-5
#     [100.00, 100.00, 99.70, 99.98], # REG Top-10
#     [100.00, 100.00, 99.53, 99.35], # SPG
#     [99.45, 99.45, 99.45, 99.45],  # LT Finetuned T5-base
#     [69.33, 70.34, 69.65, 70.39],  # LT RAG-10
#     [83.79, 83.93, 83.75, 83.93],  # LT RAG-50
#     [88.20, 88.25, 87.79, 87.70]   # LT RAG-100
# ])



def plot_module(module_id):
    modules = {
            "srer": ["SRER"],
            "reg": ["REG Top-1", "REG Top-5", "REG Top-10"],
            "spg": ["SPG"],
            "lt": ["LT Finetuned T5-base", "LT RAG-10", "LT RAG-50", "LT RAG-100"]
    }

    accuracy_values ={
        "srer": [[99.45, 99.43, 99.56, 99.39]],  # SRER
        "reg": [[99.68, 97.98, 88.74, 78.35],  # REG Top-1
                [100.00, 100.00, 99.56, 99.15], # REG Top-5
                [100.00, 100.00, 99.70, 99.98]], # REG Top-10
        "spg": [[100.00, 100.00, 99.53, 99.35]], # SPG
        "lt": [[99.45, 99.45, 99.45, 99.45],  # LT Finetuned T5-base
            [69.33, 70.34, 69.65, 70.39],  # LT RAG-10
            [83.79, 83.93, 83.75, 83.93],  # LT RAG-50
            [88.20, 88.25, 87.79, 87.70]]   # LT RAG-100
    }

    ylabels = {
        "srer": "Spatial Referring Expression Recognition (SRER)",
        "reg": "Referring Expression Grounding (REG)",
        "spg": "Spatial Predicate Grounding (SPG)",
        "lt": "Lifted Translation (LT)"
    }

    # Plot settings
    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")

    # Plot each module
    for i, module in enumerate(modules[module_id]):
        plt.plot(cities, accuracy_values[module_id][i], marker='o', linewidth=3, label=module)

    # Ensure y-axis covers the full range from 0 to 100
    plt.ylim(65, 101)

    # Title and Labels
    plt.title(ylabels[module_id], fontsize=20, fontweight='bold')
    # plt.xlabel("Cities", fontsize=16, fontweight='bold')
    plt.ylabel("Accuracy (%)", fontsize=15, fontweight='bold')
    # plt.title("Lifted Translation (LT)", fontsize=18, fontweight='bold')
    if module_id == "reg" or module_id == "lt":
        plt.legend(loc="lower left", bbox_to_anchor=(1, 0.5))
    plt.xticks(rotation=20, fontsize=15, fontweight='bold')

    # Show plot
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"{module_id}.png")

if __name__ == "__main__":
    for module_id in ["srer", "reg", "spg", "lt"]:
        plot_module(module_id)

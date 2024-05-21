import matplotlib.pyplot as plt
import numpy as np

x = np.array(["Proposed (TripleNet-S)", "Proposed (TripleNet-B)", "Yolo V5", "Yolo V3"])
y = np.array([96, 98, 80, 78])

plt.figure(figsize=(8, 6))  # Adjust figure size as needed
bar_width = 0.25  # Adjust bar width for clarity

# Add labels and title
plt.xlabel("Comparison Models")
plt.ylabel("Accuracy")
plt.title("Performance Comparison of Different Models")

plt.xticks(rotation=45, ha="right")
plt.bar(x, y, color = "hotpink")

plt.show()
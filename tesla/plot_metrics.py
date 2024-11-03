import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

column_names = [
    "Timestamp", "Rank", "Epoch", "Computation_Time", "GPU_ID", "GPU_Name",
    "GPU_Load", "GPU_Free_Mem_MB", "GPU_Used_Mem_MB", "GPU_Total_Mem_MB", "GPU_Temp_C"
]

df = pd.read_csv("metrics.csv", names=column_names)
# print("Columns in DataFrame:", df.columns)

# Convert percentage strings in 'GPU_Load' to numeric values
df['GPU_Load'] = df['GPU_Load'].str.replace('%', '').astype(float)
# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

output_dir = "output_fig"
os.makedirs(output_dir, exist_ok=True)
ranks = df['Rank'].unique()

# Set a seaborn theme for better aesthetics
sns.set_theme(style="whitegrid")


# =============================START PLOTTING===========================================
# 
#
# Plot GPU Load by Epoch for each Rank
plt.figure(figsize=(12, 6))
for rank in ranks:
    subset = df[df['Rank'] == rank]
    plt.plot(subset['Epoch'], subset['GPU_Load'], label=f'Rank {rank}')
plt.title("GPU Load by Epoch for Each Rank")
plt.xlabel("Epoch")
plt.ylabel("GPU Load (%)")
plt.legend(title="Rank")
plt.savefig(os.path.join(output_dir, "gpu_load_by_epoch.png"))
plt.close()

# Plot Free Memory over Epochs for each GPU
plt.figure(figsize=(12, 6))
for rank in ranks:
    subset = df[df['Rank'] == rank]
    plt.plot(subset['Epoch'], subset['GPU_Free_Mem_MB'], label=f'Rank {rank}')
plt.title("Free Memory by Epoch for Each Rank")
plt.xlabel("Epoch")
plt.ylabel("Free Memory (MB)")
plt.legend(title="Rank")
plt.savefig(os.path.join(output_dir, "free_memory_by_epoch.png"))
plt.close()

# Plot Used Memory over Epochs for each GPU
plt.figure(figsize=(12, 6))
for rank in ranks:
    subset = df[df['Rank'] == rank]
    plt.plot(subset['Epoch'], subset['GPU_Used_Mem_MB'], label=f'Rank {rank}')
plt.title("Used Memory by Epoch for Each Rank")
plt.xlabel("Epoch")
plt.ylabel("Used Memory (MB)")
plt.legend(title="Rank")
plt.savefig(os.path.join(output_dir, "used_memory_by_epoch.png"))
plt.close()

# Plot Temperature over Epochs for each GPU
plt.figure(figsize=(12, 6))
for rank in ranks:
    subset = df[df['Rank'] == rank]
    plt.plot(subset['Epoch'], subset['GPU_Temp_C'], label=f'Rank {rank}')
plt.title("GPU Temperature by Epoch for Each Rank")
plt.xlabel("Epoch")
plt.ylabel("Temperature (°C)")
plt.legend(title="Rank")
plt.savefig(os.path.join(output_dir, "temperature_by_epoch.png"))
plt.close()

# Plot Computation Time by Epoch for each GPU
plt.figure(figsize=(12, 6))
for rank in ranks:
    subset = df[df['Rank'] == rank]
    plt.plot(subset['Epoch'], subset['Computation_Time'], label=f'Rank {rank}')
plt.title("Computation Time by Epoch for Each Rank")
plt.xlabel("Epoch")
plt.ylabel("Computation Time (seconds)")
plt.legend(title="Rank")
plt.savefig(os.path.join(output_dir, "computation_time_by_epoch.png"))
plt.close()

# (THIS IS OPTIONAL) Plot correlation heatmap for numerical columns
plt.figure(figsize=(10, 8))
sns.heatmap(df[['GPU_Load', 'GPU_Free_Mem_MB', 'GPU_Used_Mem_MB', 'GPU_Total_Mem_MB', 'GPU_Temp_C', 'Computation_Time']].corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap of GPU Metrics")
plt.savefig(os.path.join(output_dir, "correlation_heatmap.png"))
plt.close()

# (THIS IS OPTIONAL) Plot pairplot to visualize distribution and relationships of metrics by Rank
sns.pairplot(df, hue='Rank', vars=['GPU_Load', 'GPU_Free_Mem_MB', 'GPU_Used_Mem_MB', 'GPU_Temp_C', 'Computation_Time'], palette="viridis")
plt.suptitle("Pairplot of GPU Metrics by Rank", y=1.02)
plt.savefig(os.path.join(output_dir, "pairplot_metrics_by_rank.png"))
plt.close()

print(f"All plots have been saved to the '{output_dir}' directory.")

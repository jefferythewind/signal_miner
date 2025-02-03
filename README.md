
<img width="1185" alt="snake_axe_pixel_banner" src="https://github.com/user-attachments/assets/e4616c8b-782c-477a-9d16-4d8fff95eaa9" />

# Signal Miner

> **Revolutionizing Staking:** Aligning users and the fund through unique models.

This repository houses code and notebooks to **mine** (or systematically search for) machine learning models that aim to beat benchmarks for [Numerai Classic Tournament](https://numer.ai). By automating the process of iteratively training, evaluating, and retaining high-performing models, **Signal Miner** is your quickstart into generating models that potentially produce better-than-benchmark performance on historical data.

---

## Table of Contents
1. [Background](#background)
2. [Installation & Setup](#installation--setup)
3. [Usage Overview](#usage-overview)
4. [Performance Plot & Randomness](#performance-plot--randomness)
5. [Contributing](#contributing)
6. [License](#license)

---

## Background

This notebook addresses the **Numerai Classic** data science tournament and aims to **align incentives for generic staking** on the tournament. Ideally, when more people stake, the hedge fund’s meta model improves because it can incorporate a diversity of unique signals. However, under the current setup, generic stakers often rely on **pre-existing models**—either Numerai’s example models or paid models from NumerBay—which limits the potential for fresh, **unique alpha**.

**Make Staking Great Again:**  
The core idea of this project is that **every staker** should be able to contribute unique alpha to Numerai Classic. Why? Because unique alpha:
- Has a better chance of producing **positive MMC (Meta Model Contribution)**.
- Potentially earns higher payouts than staking on widely used example models.  
- Doesn’t compromise on performance (all generated models exceed specified benchmark metrics like correlation and Sharpe).

By automatically **searching for and refining** these distinct models, we **increase** the variety of signals feeding into Numerai’s meta model, benefiting both stakers (via higher potential rewards) and Numerai (via more robust, diversified signals). 

**Signal Miner** extends this idea by creating a pipeline to **search** for robust models in an automated fashion, focusing on:
- **Unique**: Emphasizing uncommon or orthogonal predictions that add new information.
- **Transparent**: Offering clear performance metrics at each mining iteration.
- **Efficient**: Letting your machine handle the computational tasks while you focus on analysis.

The result is a **win-win**:  
- Stakers are happy because they can generate new signals and potentially earn more.  
- Numerai’s hedge fund is happy because it gains new, non-redundant alpha from the community.  


---

## Installation & Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/jefferythewind/signal_miner.git
    cd signal_miner

2. **Create (and activate) a virtual environment** (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # on Linux or macOS
    # or
    venv\Scripts\activate     # on Windows

3. **Install required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
4. **Install Jupyter (if you want to use the notebook)**:
    ```bash
    pip install jupyter
    ```
**That’s it!** Once done, you’re ready to either run the code directly (e.g., via Python scripts) or explore the iPython notebooks.

## Usage Overview

> **Recommended**: See [`Model Miner.ipynb`](Model%20Miner.ipynb) for a complete end-to-end example. It’s best run from top to bottom using **Python 3.10**.

Below is a high-level summary of how you might use **Signal Miner** in practice:

1. **Load your data** as usual (e.g., reading a CSV or Parquet file into a Pandas DataFrame).
2. **Define a benchmark configuration** to compare against (e.g., a standard LightGBM model).
3. **Create a parameter dictionary** (hyperparameters to be sampled or searched).
4. **Set up time-series cross-validation** with an embargo or gap (important to avoid leakage in financial data).
5. **Launch the asynchronous mining process** (which iterates through parameter combinations and evaluates them across the defined cross-validation folds).
6. **Check progress** periodically and see how many configurations have run.
7. **Evaluate results** relative to your benchmark on the validation and test folds (e.g., correlation, Sharpe).
8. **Export or ensemble** any configuration(s) that exceed your benchmark.

### Step-by-Step Example

Below are excerpts from the notebook demonstrating these steps:

**1. Define the benchmark configuration:**
```python
benchmark_cfg = {
    "colsample_bytree": 0.1,
    "max_bin": 5,
    "max_depth": 5,
    "num_leaves": 2**4 - 1,
    "min_child_samples": 20,
    "n_estimators": 2000,
    "reg_lambda": 0.0,
    "learning_rate": 0.01,
    "target": 'target'  # Using the first target for simplicity
}
```

**2. Create the parameter dictionary to search:**
```python
param_dict = {
    'colsample_bytree': list(np.linspace(0.001, 1, 100)),
    'reg_lambda': list(np.linspace(0, 100_000, 10000)),
    'learning_rate': list(np.linspace(0.00001, 1.0, 1000, dtype='float')),
    'max_bin': list(np.linspace(2, 5, 4, dtype='int')),
    'max_depth': list(np.linspace(2, 12, 11, dtype='int')),
    'num_leaves': list(np.linspace(2, 24, 15, dtype='int')),
    'min_child_samples': list(np.linspace(1, 250, 250, dtype='int')),
    'n_estimators': list(np.linspace(10, 2000, 1990, dtype='int')),
    'target': targets
}
```

**3. Set up time-series cross-validation** (with a gap/embargo to avoid leakage across eras):
```python
ns = 2  # number of splits
all_splits = list(TimeSeriesSplit(n_splits=ns, max_train_size=100_000_000, gap=12).split(eras))
```
Here, we use two folds. The first fold acts as “validation” and the second as a “test” set, ensuring no overlap.

![output](https://github.com/user-attachments/assets/d12e2f2d-f8da-4f2e-9e50-b03d413e2161)

**4. Launch the mining process** (asynchronous job pool) to train multiple configurations:
```python
start_mining()
```
This begins training across the folds for each parameter combination. The process runs in the background, so you can continue using the notebook.

**5. Periodically check progress:**
```python
check_progress()
# Example Output:
# Progress: 122.0/2002 (6.09%)
```
This lets you know how many configurations have completed.

**6. Evaluate results** once you’ve accumulated sufficient runs:
```python
res_df = evaluate_completed_configs(
    data, configurations, mmapped_array, done_splits, all_splits, ns
)
# Label any benchmark configuration
res_df['is_benchmark'] = (res_df.index == BENCHMARK_ID)

print("Benchmark Results:")
res_df[res_df['is_benchmark']]
```
You’ll see metrics such as `validation_corr`, `test_corr`, `whole_corr`, `validation_shp`, etc., alongside your benchmark.

**7. Compare models to the benchmark** to find superior configurations:
```python
print("Better Than Benchmark Results:")
compare_to_benchmark(res_df)
```

**8. Export any top-performing models** for deployment:
```python
to_export = [res_df.sort_values('whole_shp').iloc[-1].name]  # pick the best by Sharpe
evaluate_and_ensemble(
    to_export, configurations, mmapped_array, data,
    all_splits, feature_cols, get_model, save_name="model"
)
# Example output:
# Predict function saved as predict_model_full.pkl
```
The above snippet creates an ensemble (even if it’s a single model) and saves a `.pkl` file suitable for future inference or Numerai submission.

---

That’s the **overall usage flow** of **Signal Miner**. For the most up-to-date code and additional detail, please refer to the [**Model Miner** notebook](Model%20Miner.ipynb).


## Performance Plot & Randomness

Below is a scatter plot, which illustrates the relationship between **past performance** (cross-validation / in-sample Sharpe) and **future performance** (test fold or out-of-sample Sharpe):

<img width="883" alt="sharpe_scatter" src="https://github.com/user-attachments/assets/514da2e4-3630-475d-80b6-cb7d1776690a" />


> **Key Takeaway**: The best model on historical (validation) data is **not necessarily** the best model for unseen data. There’s inherent randomness in the modeling process, and no amount of backtesting can completely guarantee out-of-sample success.

In our example plot, each dot represents a model configuration:
- The **x-axis** is the validation Sharpe (past fold).
- The **y-axis** is the test Sharpe (future fold).
- The **benchmark** model is shown as a star, and we fit a best-fit line showing a strong linear relationship.

Some observations:
1. **Not Perfect**: The top-performing validation model isn’t the top performer on the test set, confirming that overfitting or luck can play a role in “winning” the validation stage.
2. **Benchmark Surprises**: The benchmark ranks near the top in validation, yet multiple models outperformed it on the test set.
3. **Encouraging Correlation**: Despite the inevitable randomness, there is a strong positive correlation between past and future performance—**meaning high validation Sharpe often translates to high test Sharpe.**  
4. **What If the Plot Looked Random?**: If, instead, you saw a circular or completely random distribution, that would mean your model selection is mostly noise. In such cases, “chasing” the top validation model yields little to no real out-of-sample edge.

This dynamic mirrors the transition from **training** to **live deployment**: even the best backtested model might not be the best performer going forward. But a solid positive correlation provides some confidence that better in-sample results can lead to better out-of-sample performance.

## Hardware & Resource Considerations  

This project was developed using **Python 3.10** on **Ubuntu Linux** running on an **AMD chipset** with **128 GB of RAM**.  

### Swap Space: The Secret to Avoiding Memory Errors  

One of the **most crucial** optimizations for running large-scale model mining is **ensuring you have enough swap space**. By default, Linux systems often allocate **far too little swap**, leading to **memory errors** when working with large datasets.  

**Recommendation:** Set your **swap space** to **2X your RAM**.  

In my case, that meant **expanding swap to 256 GB**—a full **1/4 of my 1 TB hard drive**!  
Since making this change, **99.99% of my memory errors have disappeared**.  

#### Linux Makes This Easy  
Ubuntu allows full control over **swap size**, unlike macOS (which doesn’t let you modify it) or Windows (which, well, let’s not even talk about Windows).  

### Expanding Swap on Ubuntu  

Run the following commands to **increase swap space** to any desired size (example: **256 GB**).  

```
# Step 1: Turn off existing swap  
sudo swapoff -a  

# Step 2: Create a new swap file of desired size (256 GB in this case)  
sudo fallocate -l 256G /swapfile  

# Step 3: Set proper permissions  
sudo chmod 600 /swapfile  

# Step 4: Format the swap space  
sudo mkswap /swapfile  

# Step 5: Enable swap  
sudo swapon /swapfile  

# Step 6: Make it permanent (add this line to /etc/fstab)  
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab  

# Verify that swap is active  
swapon --show
```


## Contributing

We welcome contributions! Whether it’s:
- Bug fixes or clarifications
- Additional model-mining techniques
- Expanded plotting and diagnostic tools

Feel free to open a Pull Request or Issue.

## License

This project is licensed under the [MIT License](LICENSE). You’re free to use and modify this code for your own modeling adventures.

**Namaste, and happy mining!**

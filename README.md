## 📚 Project Overview

**Catastrophic forgetting** remains a major challenge in continual learning systems, where neural networks forget previously learned tasks when adapting to new ones.  
This project proposes a novel **locally learnable activation function** based on **B-spline basis functions** to mitigate forgetting without sacrificing plasticity.

**Key Contributions:**
- Developed a **learnable B-spline activation function** module.
- Integrated B-spline activations into a feedforward neural network architecture.
- Conducted experiments on:
  - **Permuted MNIST continual learning** tasks.
  - **Toy regression tasks**
- Compared models using traditional activations (**ReLU**) versus **learnable B-spline activations**.

---

## 📁 Repository Structure

```bash
.
├── bspline_activation.py      # Learnable B-spline activation module
├── model.py                   # Neural network models with/without learnable activations
├── permuted_mnist.py           # Permuted MNIST dataset creation
├── train_eval_mnist.py         # Training and evaluation on continual MNIST tasks
├── plot_mnist_results.ipynb       # Visualization of forgetting curves (MNIST)
├── toy_regression.ipynb           # Regression task with Gaussian peaks (Toy dataset)
└── README.md                   
```

---

## 🧠 Learnable B-Spline Activation Function

Traditional activation functions (e.g., ReLU, Tanh) are **global** and affect the entire input space. In contrast, B-splines are **locally supported**: a change in one region can leave other regions intact, helping to mitigate forgetting of previously learned tasks.

In this implementation:
- The activation is a **trainable weighted sum of B-spline basis functions**.
- We optimize the **control points** during network training.
- Enables **local adaptation** to new tasks with **minimal interference** to old tasks.
---

## 🔥 Experiments

### 1. Permuted MNIST

- **Task**: Sequential learning of different permutations of MNIST digits.
- **Comparison**:  
  - **Baseline**: MLP with ReLU activations  
  - **Proposed**: MLP with Learnable B-Spline activations
- **Metric**: Classification Accuracy across tasks after each training stage.

**Results**:
  - These plots shows how accuracy on previous tasks drops as new tasks are learned.
  - Model with ReLU activations show sharp accuracy drops on old tasks after learning new ones, a clear sign of catastrophic forgetting.
  - Modes with B-Spline activations demonstrate much flatter forgetting curves, indicating stronger resistance to forgetting.

<p align="center">
  <img src="results/forgetting1_mnist.png" alt="Forgetting curves" width="45%"/>
  <img src="results/forgetting2_mnist.png" alt="Forgetting curves 2" width="45%"/>
</p>


  - The second visualization tracks accuracy on each task throughout all training epochs.
  - It highlights that B-Spline activations enable the network to learn new tasks while maintaining more stable performance on earlier ones, compared to the sharp performance drops observed in ReLU models.
    

<div align="center">
  <img src="results/training_timeline.png" alt="Training timeline">
</div>


### 2. Peaks Regression (Toy Task)

- **Task**: Sequential modeling of separated peaks.
- **Comparison**:
  - **Baseline**: MLP with ReLU activations
  - **Proposed**: MLP with Learnable B-Spline activations
- **Metric**: Regression accuracy on earlier peaks after training new ones.

**Important Note**:  
The toy regression task was adapted and modified from an existing repository:  
👉 [Original Source Repository](https://github.com/KindXiaoming/pykan)

**Resultes**:
  - These plots show the model’s prediction curves after sequentially learning each peak.
  - ReLU networks show strong overfitting to the most recently learned peaks, almost entirely forgetting previously fitted peaks.
  - B-Spline networks retain better fitting of earlier peaks even after multiple new peaks are learned, demonstrating **significantly reduced catastrophic forgetting**.

  ![Regression fitting](results/regression_fitting2.png)
  ![Regression fitting](results/regression_fitting.png)

---

## 🚀 Running the Code

For running and visualizing the experiments:
- Simply run the cells in `plot_mnist_results.ipynb` for the **Permuted MNIST** forgetting analysis.
- Simply run the cells in `toy_regression.ipynb` for the **Peaks Regression** fitting experiment.


---

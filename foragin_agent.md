
---

### **GUIDE FOR ASSISTANT: Implementing a Foundational Hierarchical Reasoning Model**

**Objective:**
Your task is to implement the Hierarchical Reasoning Model (HRM) to solve a simple "Foraging" task. An agent in a 2D grid world must learn to navigate to a piece of food, collect it, and return to its nest. This guide will walk you through setting up the environment, generating training data, adapting the HRM architecture, and training the model.

This project will solidify your understanding of:
1.  Applying HRM to a dynamic, non-puzzle-based problem.
2.  The interplay between the high-level (strategic) and low-level (tactical) modules.
3.  Generating "expert" data to bootstrap model learning via imitation.

---

### **Step 1: The "Foraging" Environment (`ForagingEnv`)**

First, we need a playground. You will create a simple 2D grid environment. This can be a Python class using NumPy. No need for complex libraries like `gymnasium` yet, unless you prefer it.

**Environment Specifications:**

*   **Grid Size:** 16x16.
*   **Grid Cells:** Represented by integers.
    *   `0`: Empty space
    *   `1`: Wall (impassable)
    *   `2`: Agent's current position
    *   `3`: Food location
    *   `4`: Nest location
*   **State:** The full 16x16 NumPy array is the state observed by the agent.
*   **Actions:** A discrete set of 4 actions:
    *   `0`: Up
    *   `1`: Down
    *   `2`: Left
    *   `3`: Right
*   **Episode Logic:**
    1.  An episode begins. Randomly place the nest, a single piece of food, and 5-10 small wall segments. Place the agent at the nest location.
    2.  The agent's internal "goal state" is initially **Find Food**.
    3.  When the agent moves onto the food's coordinates, the food is "collected." The agent's internal goal state switches to **Return to Nest**. The food cell `3` is now treated as empty space `0`.
    4.  The episode successfully ends when the agent, carrying food, returns to the nest's coordinates.

**Your `ForagingEnv` class should have these methods:**
*   `__init__(self, size=16)`: Initializes the grid.
*   `reset(self)`: Creates a new random puzzle (walls, food, nest) and returns the initial state.
*   `step(self, action)`: Takes an action, updates the agent's position, checks for goal completion, and returns the `(new_state, reward, done)` tuple. For now, `reward` can be a simple `+1` for finding food and `+10` for finishing the episode.

---

### **Step 2: Generating Expert Training Data**

The HRM needs examples of correct behavior. We will generate these examples by creating an "expert" solver that always finds the optimal path. A simple A* search algorithm is perfect for this.

**Data Generation Process:**

1.  Create a script `generate_foraging_data.py`.
2.  In a loop (e.g., to generate 10,000 episodes):
    a.  Initialize a new `ForagingEnv` by calling `env.reset()`.
    b.  **Path 1 (Nest to Food):** Use your A* implementation to find the shortest path from the agent's starting position (the nest) to the food.
    c.  **Path 2 (Food to Nest):** Use A* again to find the shortest path from the food's location back to the nest.
    d.  **Combine Paths:** Concatenate the two paths to form a single, complete, optimal trajectory.
    e.  **Store the Trajectory:** For each step in the trajectory, save the `(state, action)` pair. The `state` is the 16x16 grid *before* the action was taken, and the `action` is the optimal move (0-3) determined by A*.
3.  Save the entire dataset of `(state, action)` pairs to files (e.g., `train_states.npy` and `train_actions.npy`). This will be your training data.

---

### **Step 3: Adapting the HRM Architecture**

We will adapt the `HierarchicalReasoningModel_ACTV1` from the repository. The core logic of H/L modules remains, but the input/output heads need to change.

**File to Modify:** `models/hrm/hrm_act_v1.py` (or a copy, e.g., `hrm_foraging.py`)

**Key Architectural Changes:**

*   **Input Embedding (`_input_embeddings` method):**
    *   The input is a `(B, 256)` tensor, where `B` is batch size and `256 = 16x16`. Each value is 0-4.
    *   This is very similar to the token-based inputs in the repo. You can treat each grid cell as a "token."
    *   In the model's `__init__`, change the `embed_tokens` layer. The `vocab_size` is now **5** (for empty, wall, agent, food, nest).
    *   The `puzzle_emb` part is not needed for this simple task. You can remove it or design it to be a no-op.

*   **Output Head (`lm_head`):**
    *   The original model predicts a token for each position in a sequence. Our goal is to predict a *single action* for the *entire state*.
    *   The `lm_head` should be changed to project the final hidden state of a special token (e.g., the agent's token, or a global average) to the number of actions.
    *   Modify the `lm_head` to be a `CastedLinear(config.hidden_size, 4)`.
    *   The output of the H-module, `z_H`, is `(B, 257, D)`. We can take the hidden state corresponding to the agent's position (or simply the first token if we add a `[CLS]` token) and pass that through the `lm_head`. A simpler first pass is to average `z_H` across the sequence dimension: `z_H.mean(dim=1)`.
    *   The final output of the model should be a logit vector of shape `(B, 4)`.

*   **Loss Function (`ACTLossHead`):**
    *   The task is no longer sequence-to-sequence prediction. It's a simple classification task (predict the correct action).
    *   The loss will be `F.cross_entropy()` between the model's output logits `(B, 4)` and the expert action from your generated data `(B,)`.
    *   You will need to create a new, simpler loss head (`ActionPredictionLossHead`) that computes this cross-entropy loss. The Q-learning and halting logic (ACT) can be kept exactly as is, as it's independent of the primary loss function.

---

### **Step 4: The Training Loop**

Now, tie it all together.

1.  **Dataset Loader (`puzzle_dataset.py`):** Create a new `ForagingDataset` class that inherits from `torch.utils.data.Dataset`. It should read the `train_states.npy` and `train_actions.npy` files you generated. Its `__getitem__` method should return one `(state, action)` pair.
2.  **Training Script (`pretrain.py`):**
    *   Point the script to use your new `ForagingDataset`.
    *   Update the configuration (`cfg_pretrain.yaml`) to reference your new model (`hrm_foraging`) and loss head (`ActionPredictionLossHead`).
    *   The "deep supervision" mechanism will still work. At each segment, the model will run a forward pass, predict an action, and get a loss signal based on how close its prediction was to the A* expert's action.
    *   The ACT Q-learning will teach the model *how many reasoning steps* are needed to make a good decision. It might learn to "think" longer when it's in a complex part of the maze far from the food, and "think" faster when the path is a straight line.

### **First Milestone: Successful Imitation**

Your initial goal is to train the model until it can reliably imitate the A* expert. You can evaluate this by feeding it states from a test set and checking if its predicted action matches the A* action >95% of the time. You should see the training loss decrease steadily.

### **Next Steps (Beyond this Guide):**

Once the model can imitate perfectly, you have a strong foundation. The exciting next step would be to introduce an "enemy" that chases the agent, and use this pre-trained model as a starting point for full Reinforcement Learning. The model would then have to learn to deviate from the "optimal" path to avoid the enemy, developing a more robust and intelligent strategy than the simple A* solver.

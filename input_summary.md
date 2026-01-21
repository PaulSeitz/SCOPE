# Agent Data Flow & Reward Structure

## 1. Data Flow: Who Gets What?
A breakdown of the specific features fed into each network module.

| Module | Input Sources | Specific Feature Details |
| :--- | :--- | :--- |
| **Meta-Policy** | **Full Context** | • **Global Features:** 10 dims (Oxygen, Lives, Player Pos, Densities)<br>• **Objects:** All 34 objects (Flattened history)<br>• **Labels:** Multi-hot encoding of object types<br>• **Masks:** Existence flags |
| **Instance Selector** | **Full Context + Command** | • **Global Features:** Broadcasted to all objects<br>• **Objects:** All 34 objects<br>• **Command:** One-hot encoding of the `Selected Concept` (from Meta)<br>• **Masks:** Existence + Semantic Validity (Concept-based) |
| **Sub-Policy** | **Local / Relative** | • **Global Features:** 10 dims (Context awareness)<br>• **Relative Vector:** 2 dims ($dx, dy$) pointing to the *selected* target.<br>• **Capabilities:** 4 dims (Can Fire, Facing X, Facing Y) + **One-hot Concept Context** (Agent knows *why* it is moving) |

### Input Variations by Active Concept
How the Orchestrator modifies inputs before sending them to the Sub-Policy:

* **`COLLECT` / `DESTROY`:**
    * *Relative Vector:* Standard vector to `Target_Index`.
    * *Capabilities:* Standard + One-hot Concept ID.
* **`DEFAULT` (Monitor):**
    * *Relative Vector:* **Hard-masked to Zeros** (Forces agent to rely purely on Global Features/Oxygen level).
* **`AVOID`:**
    * *Relative Vector:* The Sub-Policy is executed $N$ times (once for every threat). The Orchestrator aggregates the Q-values (summation).
    * *Capabilities:* Standard + One-hot `AVOID` ID.

---



## 2. Mathematical Formulation (Rewards)

The reward vector $R \in \mathbb{R}^{|C|}$ is calculated per step. The structure ensures that each concept is trained on signals relevant only to its specific role, preventing "credit assignment confusion" where a sub-policy gets rewarded for an action it didn't intend to take.

### A. The Core Equation (Gate + Shape)
The total reward for any concept $c$ is the sum of its local "dense" progress (shaping) and its share of the global success (sparse reward).

$$R_{total}(c) = R_{shaped}(c) + \text{Gate}(c, e_t) \cdot R_{global}$$

### B. Global Reward ($R_{global}$)
This represents the "true" objective of the game (points and survival). It is the only signal that matters for the Meta-Policy, as the Meta-Policy's job is to maximize the game score, not just move closer to objects.

$$R_{global} = \Delta S_t + \beta_{fail} \cdot \mathbb{I}_{death}$$

* $\Delta S_t$: **Score Increase**. The raw points obtained in this frame.
* $\beta_{fail}$: **Death Penalty** ($-5.0$). A moderate punishment for losing a life. It is not set excessively high (like -50) to prevent the agent from becoming too timid.
* $\mathbb{I}_{death}$: Indicator function ($1$ if agent died, $0$ otherwise).

### C. The Gating Function (Credit Assignment)
To prevent "free-riding," the global reward is only assigned to the concept responsible for the event. For example, if the `DEFAULT` concept is active but a shark swims into the player's bullet, `DEFAULT` should not be rewarded for that "accidental" kill.

$$
\text{Gate}(c, e_t) =
\begin{cases}
    1 & \text{if } e_t = \text{COLLECTION} \text{ and } c = \text{COLLECT} \\
    1 & \text{if } e_t = \text{COMBAT} \text{ and } c = \text{DESTROY} \\
    1 & \text{if } e_t = \text{DEATH} \text{ (Shared Penalty)} \\
    0 & \text{otherwise}
\end{cases}
$$

* **Logic:** The `COLLECT` concept only cares about divers. The `DESTROY` concept only cares about enemies.
* **Exception:** Death is a shared failure. If the agent dies, *all* concepts receive the penalty, teaching the Meta-Policy that no concept is safe if the agent is dead.

### D. Specific Shaping Logics ($R_{shaped}$)
These are "dense" signals calculated every frame to guide the Sub-Policy towards its goal even when no scoring event occurs.

#### 1. Collect ($c=0$)
Encourages closing the physical gap to divers.

$$R_{collect} = (d_t - d_{t+1}) + \gamma \cdot \mathbb{I}_{closing} + \rho \cdot \mathbb{I}_{success}$$

* $(d_t - d_{t+1})$: **Distance Reduction**. Positive if the player is closer to the diver than in the previous frame.
* $\gamma \cdot \mathbb{I}_{closing}$: **Inertia Bonus**. A small binary bonus just for moving in the right direction. This helps overcome friction/inertia where the raw distance delta might be small.
* $\rho \cdot \mathbb{I}_{success}$: **Success Bridge**. A specific bonus ($+5.0$) for collecting the diver. This "bridges" the gap between the small shaping rewards and the potentially delayed global score.

#### 2. Avoid ($c=1$)
Penalizes getting close to threats, aggregated over specific targets and the nearest global threat.

$$R_{avoid} = - (d_t - d_{t+1}) \cdot \mathbb{I}_{panic} - \Delta d_{min\_threat} + \beta_{fail} \cdot \mathbb{I}_{collision}$$

* $\mathbb{I}_{panic}$: **Panic Radius Mask**. This is crucial. The penalty for moving closer to an enemy is *only* applied if the distance $d_t < 0.2$ (screen units). This allows the agent to swim freely in "open water" without constant negative feedback, preventing it from freezing in corners.
* $\Delta d_{min\_threat}$: **Tunnel Vision Fix**. Regardless of the target, this term penalizes getting closer to the *absolute nearest* threat on the screen, ensuring the agent doesn't dodge one shark only to back into another.

#### 3. Destroy ($c=2$)
Encourages aligning the shooting vector $\vec{v}_{fire}$ with the target vector $\vec{v}_{target}$. It relies on "alignment potential" rather than distance.

Let alignment $\phi = \max(0, \vec{v}_{fire} \cdot \hat{v}_{target})$.

$$R_{destroy} = (\phi_{t+1} - \phi_t) + \epsilon \cdot \mathbb{I}_{holding} + \rho \cdot \mathbb{I}_{success}$$

* $(\phi_{t+1} - \phi_t)$: **Aim Potential**. Rewards the *change* in alignment. Positive if the player's facing vector is rotating towards the enemy.
* $\epsilon \cdot \mathbb{I}_{holding}$: **Holding Bonus**. A small reward for *staying* aligned ($\phi > 0.9$). This prevents "aim jitter" where the agent constantly corrects its aim back and forth.

#### 4. Monitor / Default ($c=3$)
Encourages surfacing behavior when no object is targeted.

$$R_{monitor} = 0.5 \cdot (y_t - y_{t+1}) + \alpha \cdot \mathbb{I}_{refill}$$

* $0.5 \cdot (y_t - y_{t+1})$: **Surface Shaping**. Since $y=0$ is the top of the screen in Seaquest, decreasing $y$ (moving up) yields a positive reward. This weak shaping signal gently pushes the agent towards the surface.
* $\alpha \cdot \mathbb{I}_{refill}$: **Oxygen Bonus**. A reward given while the oxygen bar is actively refilling. This teaches the agent that being at the surface is valuable specifically when it needs oxygen.
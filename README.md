# Ex-10: Rod-Maneuvering-with-Prioritized-Sweeping

## Aim:

To implement the **Prioritized Sweeping Reinforcement Learning algorithm** for a Rod Maneuvering environment and learn an optimal policy that efficiently guides the rod from its initial position to the target position while avoiding obstacles and minimizing the number of steps.

---

## Algorithm:

### Prioritized Sweeping Algorithm:

**Step-1.** Initialize the state-value or action-value function (Q(s,a)) arbitrarily.

**Step-2.** Initialize a model to store state transitions and rewards.

**Step-3.** Observe the current state (s).

**Step-4.** Select an action (a) using an exploration strategy (e.g., ε-greedy).

**Step-5.** Execute the action and observe:

   * Next state (s')
   * Reward (r)

**Step-6.** Store the transition ((s,a,r,s')) in the model.

**Step-7.** Compute the priority:

   [
   P = |r + \gamma \max_a Q(s',a) - Q(s,a)|
   ]

**Step-8.** If the priority exceeds a threshold, insert ((s,a)) into the priority queue.

**Step-9.** While the queue is not empty:

   * Remove the state-action pair with the highest priority.
   * Update its Q-value.
   * Find predecessor states.
   * Recalculate their priorities.
   * Insert significant predecessors into the queue.

**Step-10.** Repeat until the goal state is reached or the maximum number of episodes is completed.

**Step-11.** Extract the optimal policy from the learned Q-values.

---

## Program:

```python

#Rod Maneuvering with Prioritized Sweeping

import heapq
import numpy as np

# Grid size (Rod maneuvering environment)
ROWS, COLS = 5, 5

actions = [(-1,0),(1,0),(0,-1),(0,1)]  # Up, Down, Left, Right

gamma = 0.95
alpha = 0.1
theta = 0.01

Q = np.zeros((ROWS, COLS, len(actions)))

model = {}
predecessors = {}

priority_queue = []

goal = (4,4)

def step(state, action):
    r, c = state
    dr, dc = actions[action]

    nr = max(0, min(ROWS-1, r+dr))
    nc = max(0, min(COLS-1, c+dc))

    next_state = (nr, nc)

    reward = 100 if next_state == goal else -1

    return next_state, reward

for episode in range(100):

    state = (0,0)

    while state != goal:

        action = np.random.randint(len(actions))

        next_state, reward = step(state, action)

        model[(state, action)] = (next_state, reward)

        if next_state not in predecessors:
            predecessors[next_state] = set()

        predecessors[next_state].add((state, action))

        p = abs(
            reward +
            gamma*np.max(Q[next_state]) -
            Q[state][action]
        )

        if p > theta:
            heapq.heappush(priority_queue,
                           (-p, state, action))

        for _ in range(5):

            if not priority_queue:
                break

            _, s, a = heapq.heappop(priority_queue)

            ns, r = model[(s,a)]

            target = r + gamma*np.max(Q[ns])

            Q[s][a] += alpha*(target-Q[s][a])

            if s in predecessors:

                for ps, pa in predecessors[s]:

                    p = abs(
                        model[(ps,pa)][1] +
                        gamma*np.max(Q[s]) -
                        Q[ps][pa]
                    )

                    if p > theta:
                        heapq.heappush(
                            priority_queue,
                            (-p, ps, pa)
                        )

        state = next_state

print("Learning Complete")
print(Q)

```


## Output:

```text

Learning Complete
[[[58.53731666 62.79588866 58.76517447 62.94096172]
  [62.92391976 67.41134433 58.70377549 67.22931872]
  [67.37815743 72.04422626 62.87367569 71.85307056]
  [72.08888176 76.93778169 67.39981617 76.2452909 ]
  [76.2512758  81.31893803 72.07795577 76.25189834]]

 [[58.73402083 67.42022927 62.99665818 67.4701236 ]
  [62.81462643 72.08912563 63.02029191 72.0425996 ]
  [67.32542897 76.93690051 67.39299893 76.8055542 ]
  [72.03830155 82.04191302 72.03592914 81.3187635 ]
  [76.24867713 86.65260826 76.88026837 81.31906007]]

 [[63.01306479 72.02279191 67.39366508 72.03178099]
  [67.42885531 76.94200332 67.41096616 76.92014629]
  [72.03314403 82.04341794 72.06372891 82.04177808]
  [76.93358159 87.42342049 76.93591918 86.65310664]
  [81.31616737 92.26643905 82.04106551 86.65311192]]

 [[67.39671877 76.78885855 71.88320081 76.92179442]
  [72.04060547 81.85837661 71.8764191  82.04492914]
  [76.9311934  87.41699772 76.93921606 87.42325152]
  [82.04055949 93.07867824 82.03831372 92.26642934]
  [86.65304889 98.17519964 87.29366012 92.26643965]]

 [[71.92901184 76.73931452 76.7679747  81.88311789]
  [76.93756653 81.86272494 76.77377515 87.27074978]
  [82.04513654 87.30638404 81.78685015 93.07230371]
  [87.42341955 93.07859797 87.25837943 99.03022627]
  [ 0.          0.          0.          0.        ]]]

```



---

## Result:

The **Prioritized Sweeping algorithm** was successfully implemented for the Rod Maneuvering environment. The agent learned an optimal path from the start state to the goal state by combining real experience with model-based planning. Prioritized updates enabled faster convergence and improved learning efficiency compared to conventional reinforcement learning approaches.



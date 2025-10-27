
# Assignment 1 DRL For Automated Testing

This project is designed to train DRL agents in 2 games, Pacman and Snake, using two seperate reward configurations, Explorer and Survival.


The training implementation has been run through both PPO (Proximal Policy Optimization) and A2C (Action-Actor-Critique), allowing multiple agents to be trained and reviewed.





## Trained Games Overview


## Installation

In order to run the following application, enter this within the terminal while in the root directory of the folder.

```bash
  pip install -r requirements.txt
```
    
## Running Tests

To start the program, run the following command

```bash
  python main.py
```

You will be greeted by the following message:
```bash
Main Menu
  [1] Watch (play current model)
  [2] Train
  [3] Plot single persona (Rewards)
  [4] Compare personas (Data Analytics)
  [5] Quit
Choose #:
```



## Watching Models


Choosing to Watch will continue onto this selection, choose which program you would like to view:
```bash
App
  [1] pacman
  [2] snake 
Choose #:   
```

After Selecting the Application, Select the training process you would like to proceed with:

```bash
 Algorithm 
  [1] ppo 
  [2] a2c 
Choose #: 
```

The final option allows you to choose a specific Persona.
```
Explorer -> Focuses on moving around, collection of points, etc.

Survivor -> Focuses on surviving, ignoring points, extending the survival time as much as possible.
```
Choose the persona which aligns with your original application choice (If you chose pacman then choose pacman_explorer or pacman_survivor)
```
Persona (from configs/models)
  [1] pacman_explorer        
  [2] pacman_survivor        
  [3] snake_explorer
  [4] snake_survivor
Choose #: 
```

The Program will now Generate a window in order for you to view the Models


## Training


Choosing Training will continue onto this selection, choose which program you would like to train:
```bash
App
  [1] pacman
  [2] snake 
Choose #:   
```

After Selecting the Application, Select the training process you would like to proceed with:

```bash
 Algorithm 
  [1] ppo 
  [2] a2c 
Choose #: 
```

The final option allows you to choose a specific reward.
```
Explorer -> Focuses on moving around, collection of points, etc.

Survivor -> Focuses on surviving, ignoring points, extending the survival time as much as possible.
```
```
Persona (from configs/models)
  [1] pacman_explorer        
  [2] pacman_survivor        
  [3] snake_explorer
  [4] snake_survivor
Choose #: 
```

Proceed to choose a number for the total timesteps looking to be trained (Default = 3,000,000)
```
Total timesteps [3000000]: 
```
Proceed to choose a number for the number of envs you want to run (Default = 8)

```
Vectorized envs [8]: 
```


## Comparing Personas (Metrics)

When choosing the Comparing Personas option:
```
Program will read all the data created while training, and create multiple different plots based on csv files 
located in logs -> {Program} -> {Persona} -> {Training Type}. The program will then create multiple data charts
analyzing the following:

pacman_survivor_ppo_vs_a2c_reward
pacman_explorer_ppo_vs_a2c_reward
snake_survivor_ppo_vs_a2c_reward
snake_explorer_ppo_vs_a2c_reward
pacman_overview_reward
snake_overview_reward
```


## Analysis

The Analysis is Stored within RootFolder/notebooks/analysis.ipynb

It has a markdown file of all the charts, explaining similarities, differences, and key arguments
## Structure

The Structure of this file is organized in the following manner:

```
Decouple app/env code, training, and testing/eval so the framework is reusable.
```
## Authors

- Alexander Simeon(from gta5) Lowe [@crazycrash115](https://www.github.com/crazycrash115)
- Ata-US-Samad Khan [@SamadAKU](https://www.github.com/SamadAKU)


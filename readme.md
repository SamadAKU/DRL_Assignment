
# Assignment 1 DRL For Automated Testing

This project is designed to train DRL agents in 2 games, Pacman and Snake, using two seperate reward configurations, Explorer and Survival.


The training implementation has been run through both PPO (Proximal Policy Optimization) and A2C (Action-Actor-Critique), allowing multiple agents to be trained and reviewed.

Please Note The Following: 


Data for Auxillary Data (pellets, deaths, tiles, apples eaten) was implemented, but not defined in analytics due to not seeing a big change while looking through the data. Files located in Logs->game->persona->algo
Able to create by using plot single persona within menu



## Trained Games Overview
The following gifs are evidence of trained DRL agents using their respective rewards and training methods.

### PACMAN EXPLORER PPO
![Pacman Training Demo](https://cdn.discordapp.com/attachments/1154497948424609792/1432180652098195456/2025-10-26_21-23-09.gif?ex=69001dac&is=68fecc2c&hm=7cdd74f24c7bb250d712a329cf6f2680d785e13dca57ff164c56d067d795de42&)

### PACMAN EXPLORER A2C
![Pacman Training Demo](https://cdn.discordapp.com/attachments/1154497948424609792/1432181131503210626/2025-10-26_21-24-44.gif?ex=69001e1e&is=68fecc9e&hm=0e40a84ab789be058065ff786036180f9dd6fca5571d94e21759e5db0505a17d&)

### PACMAN SURVIVAL PPO
![Pacman Training Demo](https://cdn.discordapp.com/attachments/1154497948424609792/1432181473103843428/2025-10-26_21-25-32.gif?ex=69001e6f&is=68feccef&hm=f9277154f2d6d151777dc8c6c553dc06a24716c0a3476a78c32854de0e94eab7&)

### PACMAN SURVIVOR A2C
![Pacman Training Demo](https://cdn.discordapp.com/attachments/1154497948424609792/1432181717736886365/2025-10-26_21-26-39.gif?ex=69001eaa&is=68fecd2a&hm=5aa4f8981045469db243b594bfdb057cec8f484f6da5a0e46cb9ea2283ed2142&)

### SNAKE EXPLORER PPO
![Pacman Training Demo](https://cdn.discordapp.com/attachments/1154497948424609792/1432181979062997032/2025-10-26_21-27-22.gif?ex=69001ee8&is=68fecd68&hm=30073067b4e99f744f7159622bbabcf20bdd748853622d6394ea96d0305826a6&)

### SNAKE EXPLORER A2C
![Pacman Training Demo](https://cdn.discordapp.com/attachments/1154497948424609792/1432182215785054289/2025-10-26_21-28-00.gif?ex=69001f20&is=68fecda0&hm=9e0cc6d7fa98aea61f696dc1e7ff14ceb08f692d5f7635993b7faf43eec8838a&)

### SNAKE SURVIVOR PPO
![Pacman Training Demo](https://cdn.discordapp.com/attachments/1154497948424609792/1432182477807685632/2025-10-26_21-28-46.gif?ex=69001f5f&is=68fecddf&hm=3f69e3a8f876180aa4c46f7ea92f77ae627b3252c26276e8b0a8d73fa9ef4472&)

### SNAKE SURVIVOR A2C
![Pacman Training Demo](https://cdn.discordapp.com/attachments/1154497948424609792/1432182683307475126/2025-10-26_21-30-19.gif?ex=69001f90&is=68fece10&hm=e7fe4a9c93e8f7f792ab749917f7c8f85c3e2af09db96f8dccb1c355be2d16e9&)

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
- Ata-US-Samad Khan (Not because he Khan-t but because he Khan) [@SamadAKU](https://www.github.com/SamadAKU)


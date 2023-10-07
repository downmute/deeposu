# deeposu
## Setup:
---
Install __Streamcompanion__

Go to _Settings_ -> _Output Patterns_ 

Clear all the default patterns and replace them with these:

(Format: Command Name - Formatting)

* finished - !status!

* misses - !miss!

* combo - !combo!

* 300 - !c300!

* 100 - !c100!

* 50 - !c50!

* name !username!


Go to _General_ -> _Miscellaneous_

Ensure that "Output live tokens to text files on disk" is checked. 

## Running:
---
Ensure StreamCompanion and osu! are open, or the program will not work properly!

Run the main.py file after installing all required libraries.

Run osu!

Ensure your monitor is on 
 



## Settings
---

Epochs - The number of times the policy is trained each training period

Iterations - The total amount of maps played before stopping

Learning Rate - The rate at which policies are changed 

Load Pretrained - Uses the latest saved model (use if training again from a checkpoint or during inference inference)

No Train - Performs inference only (just play the map normally)

## Credit
---
Credit to Edan Meyer for the base PPO implementation
https://colab.research.google.com/drive/1MsRlEWRAk712AQPmoM9X9E6bNeHULRDb?usp=sharing#scrollTo=RZOgKa5nzG5Y





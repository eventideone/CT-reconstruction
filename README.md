# Instruction of Our Code 

This readme document describes how our code runs.

Our code can process both clinical dose and low dose CT images simultaneously by only changing some parameters. 



## Prepare the environment
Our program runs on a Linux system and uses Pytorch version 1.10.
Please refer to `environment.yml` for a list of conda environments that can be used to run the code. 

Please download our [pre-trained model](https://drive.google.com/file/d/17nY62hSJ6rBGiQvs6NqOs4IcsA-PCiOe/view?usp=sharing) and place it in the folder `model`.

## Running the code
According to the competition’s template file, we provided script.py, and you can run it directly after modifying the file path `folder` and variables `iflow` that indicate the current dose type in `script.py`.
```
python script.py
```
And in each loop, the function my_recon returns a result of numpy array type as our competition’s result.
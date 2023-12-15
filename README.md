# Instruction of Our Code 

This readme document describes how our code runs.

Our code can process both clinical dose and low dose CT images simultaneously by only changing some parameters. Please make sure when reconstructing different doses CT images, there are only files of the specified dose in the folder. For example, if you want to process low dose CT images, please make sure there are only files like `*low_dose*.npy` and target files in the folder.



## Prepare the environment
Our program runs on a Linux system and uses Pytorch version 1.10.
Please refer to `environment.yml` for a list of conda environments that can be used to run the code. 

Please download our pre-trained model and place it in the folder `model`.

## Running the code
According to the competition’s template file, we provided script.py, and you can run it directly after modifying the file path.
```
python script.py
```
And in each loop, the function my_recon returns a result of numpy array type as our competition’s result.
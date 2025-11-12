# MLops_Lab5

## Steps to run the lab
### Step 1
'''bash
cd src
'''

### Step 2 
'''bash
python train.py
'''

### Step 3 
'''bash
cd ..
'''

Run this as the model directory is being saved in the root directory

### Step 4
'''bash
uvicorn src.main:app --reload
'''
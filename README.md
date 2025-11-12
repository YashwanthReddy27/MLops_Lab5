# MLops_Lab5
Steps to run the lab
Step 1
'''
cd src
'''

Step 2
'''
python train.py
'''

Step 3
'''
cd ..
'''

Run this as the model directory is being saved in the root directory
Step 4
'''
uvicorn src.main:app --reload
'''
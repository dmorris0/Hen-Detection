# DeepLabCut on HPCC

## DeepLabCut Usage **TODO**

### HPCC Access **TODO**

### Starting Up DeepLabCut Software **TODO**

## DeepLabCut Installation on HPCC

1. Install Anaconda **TODO**

2. Create virtual environment

	`conda create -n DLC python=3.8`<br>
	`conda activate DLC`
	
3. Install DeepLabCut

	`pip install deeplabcut`

4. Load HPCC modules

	`module load cuDNN/7.6.4.38-CUDA-10.1.105`<br>
	`module load FFmpeg`
	
5. Install correct versions of Tensorflow and Keras **TODO**
	
6. Start Ipython & import DeepLabCut

	`ipython`<br>
	`import deeplabcut`
	
7. Clone DeepLabCut Repository & Run Test Script

	1. Navigate to directory you would like to save the DeepLabCut 
	   source code
	   
	2. Clone the repo & navigate to test script directory
	
		`git clone https://github.com/DeepLabCut/DeepLabCut.git`<br>
		`cd DeepLabCut/examples`
		
	3. Run test script - **the test script will produce a lot of output, but at the end you should see "ALL DONE!!! - default cases are functional." near the bottom**
	
		`python testscript.py`


## Using DeepLabCut on the HPCC

### Setup

1. Upload your videos to the HPCC

2. To start (anytime you use DeepLabCut), enter the following in the terminal
	
	`module load cuDNN/7.6.4.38-CUDA-10.1.105`<br>
	`module load FFmpeg`<br>
	`ipython`<br>
	`import deeplabcut`
	
2. Create a new project
	
	- You will need the config_path for most DeepLabCut commands, so it is useful to
	  save it at the  beginning of a work session. It is a string containing the path to the config.yaml
	  file in your DeepLabCut project. The create_new_project (you only run this once to create a project)
	  function will output the path to the config file. Reference the "DeepLabCut Usage" section of this
	  document to see how to save the config_path in normal use cases.
	
	    `config_path = deeplabcut.create_new_project('HenTracks','YourName', multianimal=True)`
              
3. Configure the project **TODO: link to google drive**
    
    1. Download configuration file from Google Drive: Hens_Collab/DeepLabCutDocumentation/config.yaml
     
    2. Replace the config.yaml in your project folder with the new file
     
    3. Open the config.yaml file in a text editor
     
    	1. Change "scorer" to your initials
    	2. Change "project_path" to your DeepLabCut project path
    	3. Change the paths for each video to the correct path for your HPCC account
    	4. Optionally: Add a skeleton definition, see docs for details. This skeleton will be used when
    	   plotting analysis, however a data-driven skeleton will still be used for tracking & assembly
    	   
   4. See docs for other configuration options
   	 
### Export Label Studio Labels **TODO: link to google drive**
	
1. From your hens project main page in label studio, click Export -> JSON -> Export
2. Save the JSON file
3. Download the python script from Google Drive: Hens_Collab/DeepLabCutDocumentation/LabelStudio/cvt-label-studio-to-DLC.py
4. Upload the JSON file and the python script to the HPCC
5. If you are still in your ipython terminal, type "exit()" and hit the Enter key
6. Enter the following in the terminal in the directory with the python script
	
    `python cvt-label-studio-to-DLC.py PATH-TO-YOUR-JSON-FILE PATH-TO-YOUR-DEEPLABCUT-PROJECT`
	
7. This will place a csv file with the labels for each frame in the appropriate folder in your DeepLabCut project
8. Register labels with DeepLabCut
		
	- In your HPCC terminal
		
		`cd YOUR-DEEPLABCUT-PROJECT-DIRECTORY`<br>
		`module load cuDNN/7.6.4.38-CUDA-10.1.105`<br>
		`module load FFmpeg`<br>
		`ipython`<br>
		`import deeplabcut`<br>
		`import os`<br>
		`config_path = os.path.join(os.getcwd(), 'config.yaml')`<br>
		`deeplabcut.convertcsv2h5(config_path, userfeedback=False)`
		
9. From here, you should be able to follow the [User Guide](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/maDLC_UserGuide.md) starting from the heading "Create Training Dataset"


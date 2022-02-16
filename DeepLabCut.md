# DeepLabCut on HPCC

## HPCC Access

 1. Open PowerShell
 2. Enter the following into the shell

	`ssh <your MSU netID>@hpcc.msu.edu`<br>
	`<your MSU password>`<br>
	`ssh dev-amd20-v100`

## Starting Up DeepLabCut Software on the HPCC

1. Change to your DeepLabCut project directory
2. Activate virtual environment<br><br>
	`conda activate DLC`
	
	(You may have to load Conda before this command using `module load Conda/3`)
	
4. Load CUDA and FFmpeg HPCC Modules<br><br>
 	`module load cuDNN/7.6.4.38-CUDA-10.1.105`<br>
	`module load FFmpeg`
	
	(Before you load FFmpreg, you may have to use `module load GCCcore/10.2.0`
	
4. Start IPython shell<br><br>
	`ipython`
	
5. Within the IPython Shell<br><br>
	`import os`<br>
	`import deeplabcut`<br>
	`config_path = os.path.join(os.getcwd(), 'config.yaml')`

## DeepLabCut Installation on HPCC

1. Install Anaconda
	1. Open this [link](https://www.anaconda.com/products/individual#linux) in your web browser
	2. Scroll down to the heading "Anaconda Installers"
	3. Under "Linux" right click the 64-bit x86 installer and copy the link to the download
	4. SSH to the HPCC
	5. Enter the following commands

		`curl -O <paste the link you copied here>`<br>
		`bash Anaconda-<version>-x86_64.sh`
	6. Type "yes" and hit enter to accept the license agreement
	7. Use default installation location unless you already have an Anaconda installation, in that case, 
	choose your path
	8. Type "yes" and hit enter to run conda init
	9. Type "no" and hit enter to decline to install VSCode
	10. **If you have a previously created Anaconda installation, modify the 4 file paths in ~/.bashrc to 
	be your new installation path**
	11. Refresh contents of .bashrc file
		
		`source ~/.bashrc`

2. Create virtual environment

	`conda create -n DLC python=3.8`<br>
	`conda activate DLC`
	
3. Install DeepLabCut

	`pip install deeplabcut`

4. Load HPCC modules

	`module load cuDNN/7.6.4.38-CUDA-10.1.105`<br>
	`module load FFmpeg`
	
5. Ensure you have the correct versions of Tensorflow and Keras<br>

	1. Enter the following and note the version number next to "Version: " in the output

		`pip show tensorflow`<br>
		`pip show keras`
		
	2. If you do not have version 2.2 of TensorFlow and version 2.3.1 of Keras, complete the following steps

		`pip uninstall tensorflow`<br>
		`pip install tensorflow==2.2`<br>
		`pip uninstall keras`<br>
		`pip install keras==2.3.1`
	
6. Start Ipython & import DeepLabCut

	`ipython`<br>
	`import deeplabcut`

7. You will see a warning indicating that the GUI is unavailable, this is fine. If the above executes 
without any other errors (make sure you don't see an error related to tensorflow/cudart), exit from the IPython shell

	`exit()`
	
8. Clone DeepLabCut Repository & Run Test Script

	1. Navigate to directory you would like to save the DeepLabCut 
	   source code
	   
	2. Clone the repo & navigate to test script directory
	
		`git clone https://github.com/DeepLabCut/DeepLabCut.git`<br>
		`cd DeepLabCut/examples`
		
	3. Run test script - **the test script will produce a lot of output, but at the end you should see 
	"ALL DONE!!! - default cases are functional." near the bottom**
	
		`python testscript.py`


## Using DeepLabCut on the HPCC

### Setup

1. See "Starting Up DeepLabCut Software on the HPCC"

2. Download the zip file (3.5 GB)[here](https://drive.google.com/file/d/1RImf2z4xWjYluUKZ92aeW9WekkqwxvPo/view?usp=sharing).
This zip file contains a folder with three subdirectories.

	1. labeled-data: this folder will be copied directly into your DeepLabCut project (once you create it). It contains
	the images and their labels in DeepLabCut's CSV and H5 formats
	
	2. videos: this folder contains all the videos from which labeled frames were extracted. These should be added to
	your DeepLabCut project so you can perform analysis on them, however, it is possible to create a "dummy" folder in
	the "labeled-data" directory in your DeepLabCut project folder with the same name as the video file, excluding the 
	file extension

	3. scripts: **TODO**

3. Open a command prompt and complete the following steps

	1. Create a folder for your DeepLabCut project
	
		`mkdir C:\<your path>`
		`cd C:\<your path>`
		`mkdir videos`
		
	2. Extract the contents of the zip file containing the videos to 'C:\<your path>\videos'

4. Download the labels [here](https://drive.google.com/drive/folders/1X6Ekwn6_y4cCfZdJM1VoT10zhmMjlG5a?usp=sharing)

5. Create a new project
	
	- You will need the config_path for most DeepLabCut commands, so it is useful to
	  save it at the  beginning of a work session. It is a string containing the full path to the config.yaml
	  file in your DeepLabCut project. The create_new_project function (you only run this once to create a project)
	  will output the path to the config file. Reference the "DeepLabCut Usage" section of this
	  document to see how to save the config_path in normal use cases.
	
	    `config_path = deeplabcut.create_new_project('HenTracks','YourName', multianimal=True)`

6. Configure the project
    
    1. Download configuration file [here](https://drive.google.com/file/d/1rS0mkF33flUWivtDJS4UYNaz4yGjyj4f/view?usp=sharing)
     
    2. Replace the config.yaml in your project folder with the new file
     
    3. Open the config.yaml file in a text editor
     
    	1. Change "scorer" to your initials, or use "CZ" as your scorer to avoid having to change this throughout the project files
    	2. Change "project_path" to your DeepLabCut project path
    	3. Change the paths for each video to the correct path for your HPCC account
    	4. Optionally: Add a skeleton definition, see docs for details. This skeleton will be used when
    	   plotting analysis, however a data-driven skeleton will still be used for tracking & assembly
    	   
   4. See docs for other configuration options
   5. Download the folder [here](https://drive.google.com/drive/folders/1ftkCbHyo9JHMCj1kyE_3FLYNQCTW8oK2?usp=sharing)
   6. Copy each subdirectory of that folder into your "labeled-data" folder in your DeepLabCut project
   7. If you did not use "CZ" as your scorer in your config.yaml file, you will need to complete the following steps
   	- **Within each of the folders you copied over**
   		1. Change the name of the CSV file from CollectedData_CZ.csv to CollectedData_{your scorer variable}.csv
		2. Delete the .h5 file
		3. Open the CollectedData_{your scorer variable}.csv in Excel
		4. Change all values in the "scorer" row to your scorer variable
		5. Save the file
	- Follow the steps in "Starting Up DeepLabCut Software on the HPCC"
	- Run the following command in the IPython shell
		`deeplabcut.convertcsv2h5(config_path, user_feedback=False)`
              
	
   	 
### Export Label Studio Labels

1. From your hens project main page in label studio, click Export -> JSON -> Export
2. Save the JSON file
3. Download the python script [here](https://drive.google.com/file/d/1e3vFUEQjowPxxVXSOJvRZug1C9eCeZ5D/view?usp=sharing)
4. Enter the following in PowerShell from the directory with the python script
	
    `python cvt-label-studio-to-DLC.py PATH-TO-YOUR-JSON-FILE PATH-TO-YOUR-DEEPLABCUT-PROJECT`
	
5. This will create a CSV file in the same directory as the script called "CollectedData_{your scorer variable}.csv". Each row
	in this file corresponds to the labels for a particular frame. Some frames are taken from the same video, **all rows for frames
	from the same video need to be copied into separate CSV files all named "CollectedData_{your scorer variable}.csv". Each of
	these files need to be copied into {your deeplabcut project directory}/labeled-data/{folder for video which frames were taken from}
	and they all need to have the same header rows as the original CSV file.**
6. Register labels with DeepLabCut
		
	1. In your HPCC terminal, follow the steps under "Starting Up DeepLabCut Software on the HPCC"
	2. Enter the following command
	
		`deeplabcut.convertcsv2h5(config_path, userfeedback=False)`
		
7. From here, you should be able to follow the [User Guide](https://github.com/DeepLabCut/DeepLabCut/blob/master/docs/maDLC_UserGuide.md) starting from the heading "Create Training Dataset"


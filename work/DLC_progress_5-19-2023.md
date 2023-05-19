# DeepLabCut - How to Use

To open DeepLabCut with GUI in Terminal:

 1. Type 'conda activate DEEPLABCUT_M1'
 2. Type 'pythonw -m deeplabcut'

## Step 0: Before Opening DeepLabCut

Prepare all data necessary for annotations before opening DeepLabCut. Organize photos or videos into a separate folder or arranged in one's files for ease of access. 

## Step 1: Create a DeepLabCut Project

Open DLC and create a new project, with title and scorer variable. When accessing the file in the GUI, add new data by "Load New Videos" and "Add New Videos". This will add another file according to the path of the file on the computer.

Additionally, check each of the 'Optional Attributes' if necessary:
- 'Select the directory where project will be created': Browse for the file path where the file can be easily located.
- 'Copy the videos': Instead of dragging the image files directly from the file path, DLC makes a new folder that stores all of the images instead. 
- 'Is it a multi-animal project?': Single-animal and multi-animal projects have different requisites, dependent on the number of individual items intended from the images.
  - Single-animal: one item of interest in frame at any time. 
  - **Multi-animal**: multiple individuals of interest in each frame. Requires addition of individuals, see next step. **Multi-animal related changes will be denoted in bold**.


## Step 2: Configure the config file

Click on "Edit Config File" to open the file on the computer in a file editor (often Visual Studio Code). Scroll through to the end of the file, just under the 'bodyparts' list. From here:
 1. Add each item of the intended annotation parts to the list of 'bodyparts'.
 2. Go to "skeleton"; define connections using the '-' character (to mimic the bullet points below).
    - - Item 1-Start
      - Item 1-A, connected to Item 1-Start
      - Item 1-B, connected to Item 1-A
    - - Item 2-Start
      - Item 2-A, connected to Item 2-Start
      - Item 2-B, connected to Item 2-A
      - Item 2-C, connected to Item 2-B
    - - Item 3-Start
 3. Optionally, define the color of the skeleton, next to 'skeleton-color'. 
 4. **If in a multi-animal project, add a number of 'individual#' (replace # with an increasing number, 20 is a good start for small projects, expect up to 100 or more for bigger projects); for example**: 
    - individual1
    - individual2
    - individual3
    - individual4

Save file and exit, move to next tab to begin labeling.

## Step 3: Extract and Label Frames

If the input files were images, this step may not be necessary. Otherwise, move to the 'Extract Frames' tab on the GUI, select the config file, then run the file to extract all frames from videos. 

Then, begin labeling frames through the next tab, 'Label Frames', and begin labeling each frame. The order of the annotations are based on the list of 'bodyparts' outlined in Step 2. Due to model accuracy being dependent on the accuracy of the annotations in this stage, take as much time as necessary to be as accurate as possible.

## Step 4: Begin Training

Once all images have been labeled, move to the next tabs, 'Create Training Network' and 'Train Network'. Create a training network through the options in the former tab, and begin waiting. This takes a long time, so prepare to wait around and do something else, but be sure that the computer remains on and running to prevent accidents and errors.

After it is done, use 'Evaluate Network' to see the accuracy of the model. 

In order to analyze new data, begin by analyzing new video data in the 'Analyze Videos' tabs, then creating videos in the 'Create Videos'. It is required to analyze before creating, as otherwise DLC runs an error of 'No unfiltered data'. 

<br>

# DeepLabCut - Progress on Testing


## (4/10 - 4/14/2023) - 1: Single Animal Testing

By recommendations from Professor Morris, I revised the type of data and how it would be tracked.

Instead of testing how to track all of the parts of an object (still using clocks to test a simple set), I should instead focus on one part of the object and track that part's motion.

For example, track only the minute hand on a clock as it moves. In this instance, it would best mimic the data process of tracking hens, of which the real data forms a line down the general spine of a hen. 

### Part 2: Creating a Project

See the 4/3-4/7 time stamp for Part 1 in finding data. It is exactly the same step in this case. 

Instead of having separate markers for separate points, I used these label markers and skeleton.

bodyparts:
- center
- minute1 
- minute2
- minute3
- minute4

Each minute section represents one (1) quarter (1/4) more down the minute hand. I am using the minute hand, as it is longer than the hour hand, for ease of annotations. 

skeleton:
- - center
  - minute1
  - minute2
  - minute3
  - minute4

Because this annotation style relies only on one subject within the object, there is no need for multiple skeleton frames. Instead, it is one skeleton framework that encompasses the entire area of the minute hand; a method to replicate that of the annotation framework down the spine of a hen.

### Part 3: Getting Training Data

I put too many total iterations into the program, and ended up waiting over two hours to get 5000 iterations of training done. Do not use this many for simple training, use a reduced amount. On the flip side, a high number of iterations does make a model more accurate. 

After training, run evaluations in the next tab. Afterwards, analyze videos in the next tab afterwards.

Many stock videos are completely free, often barring a small watermark someplace. As such, I used free stock data of clocks moving to test the model.


## (4/10 - 4/14) - 2: Multi-Animal Testing

After getting a functional model of 'single-animal' testing, I began working on trying a test 'multi-animal model'. However, instead of heading straight into large batches of data, which would take far too much time for a simple trial, I searched for small data of multiple animals and opted for simple annotation skeletons, rather than trying complex frames.

### Part 1: Finding Suitable Data

For some data, I tried to find some data online, but ended up using a few small images from the hens document, ones that have very few numbers. Many of the images in the 'AllExtractedFrames' folder have very few hen in the main camera and the rest are cropped out, so I used those as test data first.

### Part 2: Creating a Project

With multi-animal projects, a list of individuals must be created; this number should be more than the total number of individuals in any image, though images can have less. I added 60 in total, expecting only a few from selectively picking out smaller sets.


## (4/3 - 4/7/2023)

### Part 1: Finding Suitable Data

To make an easy trial for simplistic testing, I used one that was quick to find en masse and in enough quantity. An easy one to track is one I found from [a similar tutorial](https://guillermohidalgogadea.com/openlabnotebook/training-your-first-dlc-model-/): they used clocks for single-animal tracking. 

Clocks have two hands, the minute and hour hands respectively, though the tutorial above adds the center, the hand denoting seconds, and 'twelve' (the point at 12 on any clock, likely for reference). 

Data is taken from photos of clocks viewed as various angles, to ensure that the program can read the data from a whole variety of angles; this type of data is to be mimicked for the actual project subject of hens. There were 10 photos taken for a simple starting data set. A few of the images didn't work, so I left it at 6.

### Part 2: Creating a Project

This project is defined as a single-target project: there is typically one clock for marking, and no others present in view for this experiment. Thus, there is no need to add individuals in the project files.

Labels are defined as follows:
- center
- hour
- minute
- second
- twelve

The skeleton is altered to individually connect each point to center:
- - center
  - hour
- - center
  - minute
- - center
  - twelve
- - center
  - second

After extracting frames from each input file, labeling begins. Labels run in order of creation. 

### Part 3: Getting Training Data

I used a smaller number of training iterations in the 'Train' tab, dropping each number of iterations 100 times smaller.

Then, I used a video I found online to test all the evaluation and see what they look like. 
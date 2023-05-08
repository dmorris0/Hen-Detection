# DeepLabCut - How to Use

To open DeepLabCut with GUI in MacOS Terminal:

 1. Type 'conda activate DEEPLABCUT_M1'
 2. Type 'pythonw -m deeplabcut'

![Terminal Commands](./Screen%20Shot%202023-04-28%20at%202.17.22%20PM.png)

## Step 0: Before Opening DeepLabCut

Prepare all data necessary for annotations before opening DeepLabCut. Organize photos or videos into a separate folder or arranged in one's files for ease of access. 

## Step 1: Create a DeepLabCut Project

Open DLC and create a new project, with title and scorer variable. When accessing the file in the GUI, add new data by "Load New Videos" and "Add New Videos". This will add another file according to the path of the file on the computer.

Additionally, check each of the 'Optional Attributes' if necessary:
- 'Select the directory where project will be created': Browse for the file path where the file can be easily located.
- 'Copy the videos': Instead of dragging the image files directly from the file path, DLC makes a new folder that stores all of the images instead. 
- 'Is it a multi-animal project?': Single-animal and multi-animal projects have different requisites, dependent on the number of individual items intended from the images.
  - Single-animal: one item of interest in frame at any time. 
  - Multi-animal: multiple individuals of interest in each frame. Requires addition of individuals, see next step.

![Optional Attributes](./Screen%20Shot%202023-04-28%20at%202.20.16%20PM.png)

## Step 2: Configure the config file

Click on "Edit Config File" to open the file on the computer in a file editor (often Visual Studio Code). Scroll through to the end of the file, just under the 'bodyparts' list. From here:
 1. Add each item of the intended annotation parts to the list of 'bodyparts'.

![Body Parts Example](./Screen%20Shot%202023-04-28%20at%202.26.06%20PM.png)

 2. Go to "skeleton"; define connections using the '-' character (to mimic the bullet points below).
    - - Item 1-Start
      - Item 1-A, connected to Item 1-Start
      - Item 1-B, connected to Item 1-A
    - - Item 2-Start
      - Item 2-A, connected to Item 2-Start
      - Item 2-B, connected to Item 2-A
      - Item 2-C, connected to Item 2-B
    - - Item 3-Start

![Skeleton Example](./Screen%20Shot%202023-04-28%20at%202.26.14%20PM.png)

 3. Optionally, define the color of the skeleton, next to 'skeleton-color'. 
 4. If in a multi-animal project, add a number of 'individual#' (replace # with an increasing number, 20 is a good start for small projects, expect up to 100 or more for bigger projects); for example: 

![Individual Example](./Screen%20Shot%202023-04-28%20at%202.25.17%20PM.png)

Save file and exit, move to next tab to begin labeling.

## Step 3: Extract and Label Frames

If the input files were images, this step may not be necessary. Otherwise, move to the 'Extract Frames' tab on the GUI, select the config file, then run the file to extract all frames from videos. 

Then, begin labeling frames through the next tab, 'Label Frames', and begin labeling each frame. The order of the annotations are based on the list of 'bodyparts' outlined in Step 2. Due to model accuracy being dependent on the accuracy of the annotations in this stage, take as much time as necessary to be as accurate as possible.

## Step 4: Begin Training

Once all images have been labeled, move to the next tabs, 'Create Training Network' and 'Train Network'. Create a training network through the options in the former tab, and begin waiting. This takes a long time, so prepare to wait around and do something else, but be sure that the computer remains on and running to prevent accidents and errors.

After it is done, use 'Evaluate Network' to see the accuracy of the model. 

In order to analyze new data, begin by analyzing new video data in the 'Analyze Videos' tabs, then creating videos in the 'Create Videos'. It is required to analyze before creating, as otherwise DLC runs an error of 'No unfiltered data'. 

<br>


# DeepLabCut - Creating and Testing a Training Model for Analyzing Hens

## (4/24 - 4/28/2023) - Pre-Annotated Data, Theory

This week's focus is on grabbing data from external data: config files and pre-annotated images.

However, this week was primarily spent on the theory of making this type of DLC project (mostly due to exams). By the next report, I will begin to make a real project using external data.

## How Annotation Data Works

Annotation data comes in the form of coordinates in a H5 file in DeepLabCut, but comes as a CSV file in other annotation formats. DLC supports a file converted in H5, but not in CSV, but will automatically convert data in CSV format (created when annotating images in GUI) into H5 format.

For a CSV file, annotation data is formatted by the X and Y pixel coordinates, body part names for each annotated part, the path to the image and scorer name (which must match the name in the config file).

![Example CSV Data](./Screen%20Shot%202023-04-26%20at%203.19.16%20PM.png)

To convert a file from CSV to H5, type: 'deeplabcut.convertcsv2h5' in a command window, along with the requisite CSV file.


External example files from this project were saved in JSON format, which can be converted to CSV. There are many methods for converting JSON to CSV, or JSON directly to H5, find one that works. (Note from me: I don't currently have an answer for this yet.)

## Using Config Files and Pre-Annotated Data

The key difference of using other config files is the paths of files; different operating system uses different types of file paths, making it difficult even if every video and/or image is the exact same, in the same place and folder.

However, while this can be remedied by manually copying each file path based on the current user's path, the better solution may be simply not to use external config files. Config files have a scorer name variable tied to it, alongside the other differences above, and the time spent altering each section of the file can be better spent by creating a new file to do the same.

![Other Config File](./Screen%20Shot%202023-04-27%20at%2011.44.05%20AM.png)

As seen above, there is the file path for the videos, which is different than below, which came from my computer (a MacBook Pro, for reference).

![My Config File](./Screen%20Shot%202023-04-27%20at%2011.58.59%20AM.png)

### Step 1: Creating a Config File for Pre-Annotated Data

Create a config file with all of the necessary images. Follow instructions below for instructions on how to make Single-Animal or Multi-Animal data sets. 

Individuals, bodyparts, and skeletons may be duplicated straight from other config files, and try to use pre-annotated images, but in a file path and format that the current computer uses.

### Step 2: Uploading Annotation Data

Extract frames, but instead of moving directly into annotating images, hop into the files and go to the 'labeled data' folder, then into each image folder. From there, upload the H5 and/or CSV file with its corresponding image. (The below images were taken from my previous training set as an example.)

![All Documents](./Screen%20Shot%202023-04-25%20at%203.06.55%20PM.png)

![Labeled Data](./Screen%20Shot%202023-04-25%20at%203.07.01%20PM.png)

![Specific Image](./Screen%20Shot%202023-04-25%20at%203.07.05%20PM.png)

Upload the completed annotation data into the folder of each image (see third image), then move forward. It may be necessary to reload DeepLabCut and the config file to ensure all annotations were properly uploaded.

### Step 3: Training the Model

Train and evaluate the model in the same method as the other models: train a number of iterations, then analyze with test data.

---
<br>

## (4/17 - 4/21/2023) - Multi-Animal Testing

This is a proper attempt, last week's attempt was stuck bouncing around between a few types of data.

Mushrooms were an idea, but upon realization that finding moving data for later would be impossible, it was quickly scrapped. I also tried to use schools of fish (particularly salmon), but it was difficult to try and annotate spots on all of them, and undersea footage is often foggy and unclear.

Eventually, I settled on annotating a small number of images taken as frames from the 'AllExtractedFrames' folder in the Google Drive. Specifically searching for images with little hens in the main image, I can gather data that could be used as a start for testing.

### Part 1: Finding Suitable Data

Image data is taken directly from the 'AllExtractedFrames' folder in the Google Drive. As stated above, data was more selective, taking smaller scaled images to avoid spending too much time annotating.

Unlike the real data sets made for the project, my test data only includes the head, tail, and small parts down the spines of each hen. This is also to reduce time spent annotating, especially given data already exists for extensive annotations throughout the hens.

### Part 2: Creating a Project

Multi-animal projects require a list of individiuals, of which the skeleton will be replicated for each individual in the list. I made a list of around 60, but don't expect to use the full amount as the images are intentionally smaller.

individuals:
- individual1
- individual2
... (ad nauseam)

I did not individually type these all out. I copied the full list of individuals from the other config file present in 'DLCProjectFiles'.

Body parts and skeleton are as follows below. The body parts list is a shortened version of the list outlined in the GitHub [here](https://github.com/dmorris0/Hen-Detection/blob/main/Annotations.md), or see below. Each of these body parts are shared among every animal, so there is no need to fill the 'uniquebodyparts' tag.

![EX_Annotation](./ex_annotation.png)

bodyparts:
- head
- spine
- tail

In place of the three 'beak' and 'hackle' sections is the 'head', and in place of all four 'spine' and 'tail' sections is just the one 'spine' and 'tail' respectively. Comb and blade are also omitted for this trial. Follow the descriptions in the GitHub source linked above for more details on each part of the hen that is being tracked.

(It was initially a 'beak' and 'neck' instead of just a 'head', but after having difficulties finding the beak and neck on hens when zoomed in, I figured it would be easier to track in this manner, especially for a small test.)

skeleton:
- - head
  - spine
  - tail

Once again, just like the clock trial in (4/10-4/14), there is no need for multiple skeletons each pointing to separate objects, just one that marks down the line from beak to tail on a hen. 

### Part 3: Getting Training Data

Initially, the images from the Google Drive are PNG files. However, while online sources state that OpenCV (and by extension, DeepLabCut) supports this file format, I was having trouble getting DeepLabCut to begin extracting data from these files, so I converted these into JPEG files, a file type that I had used previously for the single-animal test data that functioned. Indeed, after running a new test file, it worked with JPEG files.

Annotating does get a little more tedious, but that is par for the course for multi-animal projects: it's expected to take long period of times to accurately annotate everything necessary, even if the test data in this scenario is smaller. If parts of the hen were indistinguishable (mostly between head and tail), it was ignored to avoid false data.

This time, I used a properly small number of iterations (500 total), to avoid waiting too long for a data set, though in exchange for one that is less accurate.

---

<br>

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

---

<br>

## (4/10 - 4/14) - 2: Multi-Animal Testing

After getting a functional model of 'single-animal' testing, I began working on trying a test 'multi-animal model'. However, instead of heading straight into large batches of data, which would take far too much time for a simple trial, I searched for small data of multiple animals and opted for simple annotation skeletons, rather than trying complex frames.

### Part 1: Finding Suitable Data

For some data, I tried to find some data online, but ended up using a few small images from the hens document, ones that have very few numbers. Many of the images in the 'AllExtractedFrames' folder have very few hen in the main camera and the rest are cropped out, so I used those as test data first.

### Part 2: Creating a Project

With multi-animal projects, a list of individuals must be created; this number should be more than the total number of individuals in any image, though images can have less. I added 60 in total, expecting only a few from selectively picking out smaller sets.

---

<br>

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
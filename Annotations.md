# Installing & Configuring Annotation Software on Windows

1. Download Python 3.9 from the Windows Store

2. Open PowerShell

3. Create a directory for your project

	`mkdir my_project`<br>
	`cd my_project`
	
4. Install Label Studio - **This may take a long time**

	`pip install label-studio`
	
5. Add Label Studio executable ptath to user environment variables

	1. Search for "env" in Windows search
	2. Click "Edit the system environment variables"
	3. Under the "Advanced" tab, click on "Environment Variables"
	4. Under "User variables for <your account name>", click on "New"
	5. ???

6. Create a Label Studio Account

7. Create and Configure Project **TODO: link to google drive**

	- Download the [XML configuration file](https://drive.google.com/file/d/1zpVENMFebEQhxsQVjsIAEjBeD-Rf3LxO/view?usp=sharing)

	- Go to the settings for your new project
	
	- Go to "Labeling Interface"
	
	- Click on "Code"
	
	- Copy and paste the contents of the label-studio-config.xml file into the box
	
	- Click save
	
	- Now, once you import your data, you can begin labeling. When labeling
	  the first hen, for example, type "hen1-" in the filter box at the top 
	  and it will hide the labels you don't care about.
	  

## Bodypoints **TODO: link to google drive**

I recommend reviewing my examples [here](https://drive.google.com/drive/folders/1J7VcchOxZpjfc-51n-_e0IHKwFe8onw5?usp=sharing)

| Bodypart  | Description |
|-----------|-------------------------------------------------------------- |
| Beak 		| Tip of beak, if it's not visible, make your best guess |
| Comb		| In the middle of the comb, close to the hen's head |
| Blade		| Back of the comb, close to the hen's head |
| Hackle 1	| Just behind the blade, as close as you can get it to the blade without touching it |
| Hackle 2	| Halfway between Hackle 1 & 2, account for the orientation of the hen's neck |
| Hackle 3	| Where the neck anchors to the body |
| Spine 1 	| Just behind hackle 3, as close as you can get it to hackle 3 without touching it |
| Spine 2	| One third of the distance between Spine 1 & 4 |
| Spine 3	| Two thirds of the distance between Spine 1 & 4 |
| Spine 4	| The final point on the hen's back that is not the tail |
| Tail 1	| First point on tail, this should be very close to spine 4 |
| Tail 2	| Halfway between tail 1 & 3, account for tail  orientation |
| Tail 3	| Last point on tail, if feathers split apart a	t the end, use last visible point in center |


# Mini-Project--Application-of-NN
## Project Title: SPEECH EMOTION RECOGNITION

## Project Description 
The act of attempting to understand human emotion and affective states from speech is known as Speech Emotion Recognition, or SER. This takes use of the fact that tone and pitch in the voice often indicate underlying emotion. Because emotions are subjective and annotating audio is difficult, SER is difficult. It basically aids the user in determining what type of feeling they are experiencing at the moment.


## Algorithm:
1. Get the data from libraries.
2. Run the program in sublime tool.
3. It detect the emotions from the played audio .
4. plot the graph.
5. Study the final output.

## Program:
```
Import  webbrowser
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import wave	
import random
def extract_feature(file_name, mfcc, chroma, mel):
with soundfile.SoundFile(file_name) as sound_file:
X = sound_file.read(dtype="float32")
sample_rate=sound_file.samplerate
if chroma:
	stft=np.abs(librosa.stft(X))
if mfcc:
mfccs=np.mean(librosa.feature.mfcc(y=X,sr=sample_rate, n_mfcc=40).T, axis=0)
result=np.hstack((result, mfccs))
accuracy=accuracy_score(y_true=y_test, y_pred=y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
```
```
<title>recommendation</title>

<style type="text/css">
	#button,#button1{
		background-color: green;
		color: yellow;
		padding: 5px 10px;
		border-radius: 10px;
	}
	#button:hover{
		background-color: darkred;
	}

	#button1 :hover{
		background-color: darkred;
	}
	body{
		border: solid 2px black; 
		padding-bottom: 10px;
	}
	h1{
		text-decoration: underline;	
	}
</style>
<center><h1><p style="color:red">HAPPY IS A DIRECTION NOT A PLACE.</p></h1></center>

	
</body>
<h1>
	ANGRY SONG 
</h1>
<form>
<a href="https://youtu.be/TMKb8An49mk" id="link">

<input type="button" value="click me.." id="button">

</a>
</form>


<h1>
	HAPPY MOTIVATIONAL SPEECH
	
</h1>
<form>
<a href="https://youtube.com/shorts/iuQ3zyKq-JY?feature=share">
<input type="button" value="click me.." id="button1">
</a>
</form>
</center>	
```
## Dataset :

![image](https://user-images.githubusercontent.com/96875495/205837996-86f49ade-88f7-4c59-9834-2da993647c7d.png)

## Output:

![image](https://user-images.githubusercontent.com/96875495/205838041-5f292abf-d928-4ef0-898f-7026e5ffd801.png)

![image](https://user-images.githubusercontent.com/96875495/205838096-f8436124-a26a-4269-b83a-4494be1c0c09.png)

## Graph:

![image](https://user-images.githubusercontent.com/96875495/205838237-7aa0363f-ec6b-4614-87be-34c6d8e095e2.png)


## Advantage :

Emotion recognition provides benefits to many institutions and aspects of life.

It is useful and important for security and healthcare purposes.

Also, it is crucial for easy and simple detection of human feelings at a specific moment without actually asking them.


## Result:

Thus the speech emotion recognition using MLP classifier is implemented successfully.



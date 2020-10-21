# document_identifier
It is a Deep Learning based application using intel OpenVino Toolkit
###### Pre-requisite :
```bash
install OpenVino
```
## Get started :
```bash
$ git clone https://github.com/halimasoulb/document_identifier
```
```bash
$ cd document_identifier
```
## How to run :
Run the script document_identifier.sh with one input argument, which is the path to documents gallery.
>Just Execute :
```bash
$ ./document_identifier.sh [<documents gallery path>]
```
###### ON CAMERA:
If you want to run with camera video, set the input to camera number or 0
```bash
$ ./document_identifier.sh 0
```
######  ON IMAGE: 
If you want to run with image, set path to the input stream
```bash
$ ./document_identifier.sh document.jpg
```
# How to label the identified Data
- Labeled text \
  Every text section is labeled with a single boxe and the text detected is written above thanks to the pre-trained OpenVino model text-spotting, to extract the relevant data detected with this models we used a regular expression filter placed in the config.js file all these treatments are done in the textDetector class
- Labeled face \
  The initial frame is given to DocumentAligner class to align the image when rotated thanks to face-detection-adas-0001 and facial-landmarks-35-adas-0002 OpenVino pre-trained models and other treatments.
- Final result \
  The final result is shown in the web browser  as data table using flask framework this treatment is carried out in WebServer class \
>Before loading data to the browser \
  ![WhatsApp Image 2020-10-21 at 12 24 05](https://user-images.githubusercontent.com/47951591/96720260-75fbd480-13a2-11eb-98d9-0a41c6c992bf.jpeg)
>After loading data \
  ![hihi](https://user-images.githubusercontent.com/47951591/96720812-279b0580-13a3-11eb-8b70-f14e7726075e.png)


  
  
# Identity Document identification Pipeline
![pipeline](https://user-images.githubusercontent.com/47951591/96708202-ccf8ae00-1390-11eb-8aab-51fd9bef8042.PNG)
  
  








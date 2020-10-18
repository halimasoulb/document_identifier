# document_identifier
It is a Deep Learning based application using intel OpenVino Toolkit
###### Pre-requisite:
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
ON CAMERA
If you want to run with camera video, set the input to camera number or 0
```bash
$ ./document_identifier.sh 0
```
ON IMAGE
If you want to run with image, set path to the input stream
```bash
$ ./document_identifier.sh document.jpg
```




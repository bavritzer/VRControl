# VRControl
This repository provides tools for interacting with StableDiffusion in VR using controlnet in realtime. Inspired by [this software by house of secrets](https://github.com/houseofsecrets/SdPaint/tree/main). I made this software and have tested it myself, as you can see in [the included video](vrcontroldemo.mp4) and I hope it works for you too!     
  
## IMPORTANT DISCLAIMERS  
This repository uses Generative AI models which are fundamentally stochastic. As such, they may generate explicit or disturbing imagery, although there are measures in place to mitigate this (see usage). Also, this repository uses http communication functionality for TCP/IP communication and is not secure for a production environment. It is intended only for personal use, and the producers of this software assume no liability nor warranty for any damages that may ensue. Please use this software responsibly.  
## Licensing  
[StableDiffusion](https://huggingface.co/runwayml/stable-diffusion-v1-5) and [ControlNet](https://huggingface.co/lllyasviel/sd-controlnet-scribble) are each subject, respectively, to a different version of the [openrail](https://huggingface.co/spaces/CompVis/stable-diffusion-license)[license](https://huggingface.co/blog/open_rail), which is not a license that Github currently supports. I have included a link to the licenses in this repository. If you use the included functionality to download and communicate with StableDiffusion, ControlNet, and other included properties, you do so subject to that licensing agreement and its included conditions. Furthermore, all output of these programs, except where in violation of any applicable licenses, belongs to the user, and with it any liability or warranty for the use of this software. However, certain assets or intellectual property used to create this software may belong to Meta or Unity Technologies, and are subject to their terms of licensing and use. Please use AI models responsibly.   
## Installation  
This repository includes an APK file for installation onto a VR Headset (tested on Meta Quest 2, should also work on Meta Quest 3) and software for communicating with your headset using your computer.    
1. To load this APK file onto a Meta Quest headset, you will need [Sidequest](https://sidequestvr.com/). A guide on installing and loading an APK file onto your headset for use can be found [here](https://www.youtube.com/watch?v=zzizceAOW-w).

2. This repository also includes a server file built in python that processes requests from the headset and returns the augmented drawing. A text file is included in this repository which specifies the necessary dependencies. To use these, first you should install [Anaconda](https://www.anaconda.com/download), a freely-available dependency management software. You may also need to install NVIDIA's [cudatoolkit](https://developer.nvidia.com/cuda-12-1-0-download-archive) depending on your computer platform. I have tested this software on a 3070Ti and it generates in real time as seen in [this video](vrcontroldemo.mp4). On CPU, it is significantly slower, but I have not tested it on a Mac so it may run faster on e.g. an M2 Mac.   
3. Currently, you must use Anaconda to create the environment and import the dependencies from the vrcreq.txt file using `conda create -n vrcenv --file vrcreq.txt` in the terminal. You should then activate the environment using `conda activate vrcenv` or by referencing the associated python.exe file when running the script `potatoserver.py`. I will try to condense this into a shell script in the future.       
## Usage
1. First, boot up the server on your computer and agree to the terms. You may be prompted to allow python to access your computer's networking functionality. When you do this, you should see a 2 digit code. You will need to enter this code on your headset.   
2. Once you have entered and confirmed the code on the headset using the sliders, just draw using the black marker on the white canvas, within the semi-transparent area. To move around, use the joysticks. To change the marker to white, press A. To clear your drawing, press B. The right (or left) trigger must be pressed to draw and the secondary trigger must be pressed to hold the marker.   
3. Once you start drawing, the program will process your drawings in realtime (depending on your hardware) and generate the output automatically. It will then be displayed on the golden easel and also on the ceiling. If you don't add anything to your drawing, it will not generate any new output. It is set to impressionism by default, but this can be changed. Press X to open the style menu and change how the drawing is rendered using the slider. You can also adjust the settings for the drawings in the `potatoserver.py` file manually.  
4. The first time you run this software on a computer, python will by default download the necessary AI models as soon as you communicate with the server. You do this subject to the terms and conditions of licensing of these models as discussed above. If you do agree to the terms, this downloading process will take a considerable amount of time and space, and a progress bar is included. Once that process is done, you should be good to go as soon as you enter the code!    
5. The server will run until it is stopped by pressing the "OK" button in the End Session dialog. You can minimize the dialog while you are using the program. DO NOT attempt to kill the python script manually. 
  
## Notes  
The music included with this program was written by me and is made freely available. The skybox used is open source and freely available. The other assets are generated by Unity and Meta and subject to applicable licensing and use conditions.  
## Contribution  
Contributions to this project are welcome! Forks and issue tickets are also welcome. The APK file was built using Unity but the python code is exposed for anyone to modify subject to the terms of the applicable licensing. This project is made freely available and will be maintained as permits my schedule. If you want to contribute financially to my work you can [buy me a coffee](https://ko-fi.com/brunoavritzer). Also, if you would like an alpha key to test the project, you may [contact me](mailto:bavritzer@yahoo.com) to request one. Requests from Mac users are especially appreciated. 

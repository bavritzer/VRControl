from http.server import CGIHTTPRequestHandler, HTTPServer
import os
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import socket
import datetime
import tkinter as tk
from tkinter import messagebox
import threading
Image.LOAD_TRUNCATED_IMAGES = True

inputcount = 0
outputcount = 0
keymag = 1
prompts = ["a painting in the style of Rembrandt, oil on canvas, Masterpiece", "a painting in the style of Monet, Masterpiece", "a painting in the style of Picasso, Masterpiece", "a painting in the style of Kandinsky, Masterpiece", "a painting in the style of Andy Warhol, Masterpiece", ""]
promptindex = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pipe = None
inputpath = ""
outputpath = ""
modinputpath = ""
date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(' ', '').replace('-', '').replace(':', '')

class createBox(threading.Thread):
    def run(self):
        root = tk.Tk()
        resp = messagebox.showinfo(title = 'End Session', message= "Server is now running. Press OK to end session.", type = messagebox.OK, parent = root)
        if resp == 'ok':
            os._exit(0)

class PotatoHTTPServer(CGIHTTPRequestHandler):

    def do_POST(self):
        global promptindex, prompts
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        try:
            promptindex = int(post_data)
        except ValueError:
            prompts[5] = str(post_data)[1:]
            promptindex = 5
        self.send_response(200)
        self.end_headers()

        
    def do_PUT(self):
        global inputcount, outputcount, prompts, promptindex, device, pipe, inputpath, modinputpath, outputpath, date
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        # Get the PNG file from the request data and save it to disk
        png_data = self.rfile.read(int(self.headers['Content-Length']))
        fnamein = 'input'+date+str(inputcount)+'.png'
        fnameout = 'output'+date+str(outputcount)+'.png'
        inputcount = inputcount+1
        outputcount = outputcount+1
        os.chdir(inputpath)
        with open(fnamein, 'wb') as f:
            f.write(png_data)

        
        
        image = Image.open(fnamein)
        thresh = 130
        fn = lambda x: 255 if x> thresh else 0
        image = image.convert('L').point(fn, mode = '1')
        os.chdir(modinputpath)
        image.save('modinput'+date+str(inputcount-1)+'.png')
        generator = torch.manual_seed(0)
        image_output = pipe(prompts[promptindex], image, num_inference_steps=10, width=512, height=512, generator = generator, guidance_scale=6).images[0]
        os.chdir(outputpath)
        image_output.save(fnameout, format='png')
        
        # Serve the saved output file
        
        with open(fnameout, 'rb') as f:
            self.wfile.write(f.read())

if __name__ == "__main__":
    resp = messagebox.showinfo(message = "By clicking OK, you agree to the terms and conditions of the AI model licenses laid out in the ReadMe included in this repository at https://github.com/bavritzer/VRControl.", title = "Initializing", type = messagebox.OKCANCEL)
    if resp=='cancel':
        exit(0)
    if not os.path.exists("inputs"):
        os.mkdir("inputs")
    if not os.path.exists("modinputs"):
        os.mkdir("modinputs")
    if not os.path.exists("outputs"):
        os.mkdir("outputs")
    resp = messagebox.showinfo(message = "Would you like to disable the safety checker? This will speed up image generation and eliminate any blank images but may produce content unsuitable for certain audiences such as minors. By disabling this feature, you acknowledge that you have read the terms and conditions of usage as described at https://github.com/bavritzer/VRControl and certify that your usage complies with those terms.", title = "Disable Safety Checker?", type = messagebox.YESNO, default=messagebox.NO)
    inputpath = os.path.abspath("inputs")
    modinputpath = os.path.abspath("modinputs")
    outputpath = os.path.abspath("outputs")
    controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
    if resp == 'yes':
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
    else: 
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"), torch_dtype=torch.float16)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    if(device=='cpu'):
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    idn = s.getsockname()[0]
    s.close()
    idm = idn[-3:]
    idm = idm.replace('.', '')
    idm = str(int(idm)*keymag)
    messagebox.showinfo(title="Access Code", message = "Please Enter This Code When Prompted on VRControl: \n\n"+idm)
    server_address = (idn, 8080)
    httpd = HTTPServer(server_address, PotatoHTTPServer)
    print('Serving at '+str(idn)+':8080')
    createBox().start()
    threading.Thread(target = httpd.serve_forever(), daemon = True).start()
    

   
    
    
from http.server import CGIHTTPRequestHandler, HTTPServer
import os
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
from diffusers.utils import load_image
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
import socket
import ctypes
Image.LOAD_TRUNCATED_IMAGES = True

inputcount = 0
outputcount = 0
keymag = 1
prompts = ["a painting in the style of Rembrandt, oil on canvas, Masterpiece", "a painting in the style of Monet, Masterpiece", "a painting in the style of Picasso, Masterpiece", "a painting in the style of Kandinsky, Masterpiece", "a painting in the style of Andy Warhol, Masterpiece"]
promptindex = 1

class PotatoHTTPServer(CGIHTTPRequestHandler):
    def do_GET(self):
        global inputcount, outputcount
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        # Get the PNG file from the request data and save it to disk
        png_data = self.rfile.read(int(self.headers['Content-Length']))
        fnamein = './input'+str(inputcount)+'.png'
        fnameout = './output'+str(outputcount)+'.png'
        
        inputcount = inputcount+1
        outputcount = outputcount+1
        with open(fnamein, 'wb') as f:
            f.write(png_data)

        
        # Process the PNG file and save
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"), torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        image = Image.open(fnamein)
        thresh = 130 #threshold to send to white
        fn = lambda x: 255 if x> thresh else 0
        image.convert('L').point(fn, mode = '1')
        generator = torch.manual_seed(0)
        image_output = pipe("an illustration in the style of Moebius", image, num_inference_steps=20, width=512, height=512, guidance_scale=7.5).images[0]
        image_output.save(fnameout, format='png')
        
        # Serve the saved output file
        with open(fnameout, 'rb') as f:
            self.wfile.write(f.read())

    def do_POST(self):
        global promptindex
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        promptindex = int(post_data)
        print(promptindex)
        self.send_response(200)
        self.end_headers()

        
    def do_PUT(self):
        global inputcount, outputcount, prompts, promptindex
        self.send_response(200)
        self.send_header('Content-type', 'image/png')
        self.end_headers()
        # Get the PNG file from the request data and save it to disk
        png_data = self.rfile.read(int(self.headers['Content-Length']))
        fnamein = './input'+str(inputcount)+'.png'
        fnameout = './output'+str(outputcount)+'.png'
        inputcount = inputcount+1
        outputcount = outputcount+1
        with open(fnamein, 'wb') as f:
            f.write(png_data)

        
        # Process the PNG file and save
        controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-scribble", torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=StableDiffusionSafetyChecker.from_pretrained("CompVis/stable-diffusion-safety-checker"), torch_dtype=torch.float16)
        #pipe = StableDiffusionControlNetPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", controlnet=controlnet, safety_checker=None, torch_dtype=torch.float16)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.enable_model_cpu_offload()
        image = Image.open(fnamein)
        thresh = 130
        fn = lambda x: 255 if x> thresh else 0
        image = image.convert('L').point(fn, mode = '1')
        image.save('input'+str(inputcount-1)+'mod.png')
        generator = torch.manual_seed(0)
        image_output = pipe(prompts[promptindex], image, num_inference_steps=10, width=512, height=512, generator = generator, guidance_scale=6).images[0]
        image_output.save(fnameout, format='png')
        
        # Serve the saved output file
        with open(fnameout, 'rb') as f:
            self.wfile.write(f.read())

if __name__ == "__main__":
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    idn = s.getsockname()[0]
    s.close()
    idm = idn[-3:]
    idm = idm.replace('.', '')
    idm = str(int(idm)*keymag)
    ctypes.windll.user32.MessageBoxW(0, idm, "Please Enter This Code When Prompted on VRPaint")
    server_address = (idn, 8080)
    httpd = HTTPServer(server_address, PotatoHTTPServer)
    print('Serving at '+str(idn)+':8080')
    httpd.serve_forever()
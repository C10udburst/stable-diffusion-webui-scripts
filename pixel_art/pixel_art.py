import modules.scripts as scripts
import gradio as gr

from modules import images
from modules.processing import process_images, Processed
from modules.processing import Processed
from modules.shared import opts, cmd_opts, state


class Script(scripts.Script):  
    def title(self):
        return "Pixel Art"
    def show(self, is_img2img):
        return True

# How the script's is displayed in the UI.
# The returned values are passed to the run method as parameters.
    def ui(self, is_img2img):
        from PIL import features
        
        downscale = gr.Slider(minimum=1, maximum=64, step=1, value=8, label="Downscale multiplier")
        rescale = gr.Checkbox(True, label="Rescale image back to original size")
        color_palette = gr.Slider(minimum=0, maximum=256, step=1, value=16, label="Color palette size (set to 0 to keep all colors)")
        palette_methods = ['Median cut', 'Maximum coverage', 'Fast octree']
        if features.check_feature("libimagequant"):
            palette_methods.insert(0, "libimagequant")
        palette_method = gr.Radio(choices=palette_methods, value=palette_methods[0], label='Palette extraction method')
        return [downscale, rescale, color_palette, palette_method]

  

# This is where the additional processing is implemented. The parameters include
# self, the model object "p" (a StableDiffusionProcessing class, see
# processing.py), and the parameters returned by the ui method.
    def run(self, p, downscale, rescale, color_palette, palette_method):
        from PIL import Image

        palette_methods = {
            "libimagequant": Image.Quantize.LIBIMAGEQUANT,
            "Median cut": Image.Quantize.MEDIANCUT,
            "Maximum coverage": Image.Quantize.MAXCOVERAGE,
            "Fast octree": Image.Quantize.FASTOCTREE
        }
        quantize = palette_methods.get(palette_method, None)

        # function which takes an image from the Processed object
        def process(im):
            # calculate sizes
            o_width, o_height = im.size
            s_width = int(o_width / downscale)
            s_height = int(o_height / downscale)

            raf = im
            raf = raf.resize((s_width, s_height), Image.NEAREST)
            
            if color_palette > 0:
                raf = raf.convert("RGB")
                raf = raf.quantize(colors=int(color_palette), method=quantize, dither=Image.Dither.FLOYDSTEINBERG)
            if rescale:
                raf = raf.resize((o_width, o_height), Image.NEAREST)
            
            return raf


        proc = process_images(p)
        # use the save_images method from images.py to save
        # them.
        for i in range(len(proc.images)):

            proc.images[i] = process(proc.images[i])

            images.save_image(proc.images[i], p.outpath_samples, "",
            proc.seed + i, proc.prompt, opts.samples_format, info= proc.info, p=p)

        return proc

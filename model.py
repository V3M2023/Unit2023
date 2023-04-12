from jetson_inference import imageNet, detectNet, segNet, poseNet, actionNet, backgroundNet
from jetson_utils import cudaFont, cudaAllocMapped, Log
from enum import Enum


class Model:
    """
    Represents DNN models for classification, detection, pose, ect.
    """
    def __init__(self,type, model, labels='', colors='', input_layer='', output_layer='', **kwargs):
        """
        Load the model, either from a built-in pre-trained model or from a user-provided model.
        
        Parameters:
        
            type (string) -- the type of the model (classification, detection, ect)
            model (string) -- either a path to the model or name of the built-in model
            labels (string) -- path to the model's labels.txt file (optional)
            input_layer (string or dict) -- the model's input layer(s)
            output_layer (string or dict) -- the model's output layers()
        """
        self.model = model
        self.enabled = True
        self.results = None
        self.frames = 0
        
        if not output_layer:
            output_layer = {'scores': '', 'bbox': ''}
        elif isinstance(output_layer, str):
            output_layer = output_layer.split(',')
            output_layer = {'scores': output_layer[0], 'bbox': output_layer[1]}
        elif not isinstance(output_layer, dict) or output_layer.keys() < {'scores', 'bbox'}:
            raise ValueError("for detection models, output_layer should be a dict with keys 'scores' and 'bbox'")
         
        print(input_layer)
        print(output_layer)
        
        self.net = detectNet(model=model, labels=labels, colors=colors,
                             input_blob=input_layer,
                             output_cvg=output_layer['scores'],
                             output_bbox=output_layer['bbox'])

        self.net.SetTrackingEnabled(True)
        self.net.SetTrackingParams(minFrames=3, dropFrames=10, overlapThreshold=0.5)
            
    def Process(self, img):
        """
        Process an image with the model and return the results.
        """
        if not self.enabled:
            return
            
        self.results = self.net.Detect(img, overlay='none')
        
        self.frames += 1
        return self.results

    def Visualize(self, img, results=None):
        """
        Visualize the results on an image.
        """
        if not self.enabled:
            return img
        
        
            
        if results is None:
            results = self.results
            results = [result for result in results if result.ClassID == 1]

        self.net.Overlay(img, results)
            
        return img
        
    def IsEnabled(self):
        """
        Returns true if the model is enabled for processing, false otherwise.
        """
        return self.enabled
        
    def SetEnabled(self, enabled):
        """
        Enable/disable processing of the model.
        """
        self.enabled = enabled
        
    @staticmethod
    def Usage():
        """
        Return help text for when the app is started with -h or --help
        """
        return imageNet.Usage() + detectNet.Usage() + segNet.Usage() + actionNet.Usage() + poseNet.Usage() + backgroundNet.Usage() 
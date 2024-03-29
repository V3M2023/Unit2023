#!/usr/bin/env python3
#
# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the 'Software'),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#
import sys
import threading
import traceback
from datetime import datetime
from dataclasses import dataclass

from model import Model
from jetson_utils import videoSource, videoOutput

@dataclass
class Person:
    track_id: int
    time_in: int


class Stream(threading.Thread):
    """
    Thread for streaming video and applying DNN inference
    """
    def __init__(self, args):
        """
        Create a stream from input/output video sources, along with DNN models.
        """
        super().__init__()
        
        self.args = args
        self.input = videoSource(args.input, argv=sys.argv)
        self.output = videoOutput(args.output, argv=sys.argv)
        self.frames = 0
        self.models = {}
        self.time_ins = {}
        self.duration_history = []
        self.count_history = []
        
        # these are in the order that the overlays should be composited
        model_types = {
            'detection': args.detection,
        }
        
        for key, model in model_types.items():
            if model:
                self.models[key] = Model(key, model=model, labels=args.labels, colors=args.colors, input_layer=args.input_layer, output_layer=args.output_layer)

        if args.log:
            self.write_file_header()
            
        """
        # these are in the order that the overlays should be composited
        if args.background:
            self.models['background'] = Model('background', model=args.background)
            
        if args.segmentation:
            self.models['segmentation'] = Model('segmentation', model=args.segmentation)
            
        if args.classification:
            self.models['classification'] = Model('classification', model=args.classification)
        
        if args.detection:
            self.models['detection'] = Model('detection', model=args.detection)
           
        if args.pose:
            self.models['pose'] = Model('pose', model=args.pose)
            
        if args.action:
            self.models['action'] = Model('action', model=args.action)
            
            if args.classification:
                self.models['action'].fontLine = 1
        """

    def process(self):
        """
        Capture one image from the stream, process it, and output it.
        """
        img = self.input.Capture()
        
        if img is None:  # timeout
            return
            
        for model in self.models.values():
            results = model.Process(img)
            timestamp = datetime.now()#.strftime("%Y-%m-%d %H:%M:%S")

            objects_count = self.get_count(results)
            self.count_history.append((timestamp, objects_count))
            if (self.args.log):
                self.write_to_file(timestamp, objects_count)

            people_results = self.get_people_results(results)
            # Register new people
            for result in people_results:
                if result.TrackID not in self.time_ins:
                    self.time_ins[result.TrackID] = timestamp
            # Remove people that are not in the frame anymore
            to_remove = []
            for track_id in self.time_ins.keys():
                if track_id not in [result.TrackID for result in people_results]:
                    duration = (timestamp - self.time_ins[track_id]).total_seconds() * 1000
                    if duration > 1000:
                        self.duration_history.append(duration)
                    to_remove.append(track_id)
            for track_id in to_remove:
                del self.time_ins[track_id]

            print(f"count: {objects_count}, len(time_ins): {len(self.time_ins)}, len(duration_history): {len(self.duration_history)}")

        for model in self.models.values():
            img = model.Visualize(img)
            
        self.output.Render(img)

        if self.frames % 25 == 0 or self.frames < 15:
            print(f"captured {self.frames} frames from {self.args.input} => {self.args.output} ({img.width} x {img.height})")
   
        self.frames += 1

    def get_count(self, results):
        if self.args.detection == "peoplenet":
            people_results = [result for result in results if result.ClassID == 0]
            faces_results = [result for result in results if result.ClassID == 2]
            return max(len(people_results), len(faces_results))
        elif self.args.detection == "ssd-mobilenet-v2":
            people_results = [result for result in results if result.ClassID == 1]
            return len(people_results)
        else:
            raise Exception()

    def get_people_results(self, results):
        if self.args.detection == "peoplenet":
            return [result for result in results if result.ClassID == 0]
        elif self.args.detection == "ssd-mobilenet-v2":
            return [result for result in results if result.ClassID == 1]
        else:
            raise Exception()

    def get_duration_history(self):
        timestamp = datetime.now()
        current_durations = self.duration_history + [(timestamp - t_in).total_seconds() * 1000 for t_in in self.time_ins.values()]
        return current_durations

    def write_to_file(self, timestamp, objects_count):
        with open(self.args.log, "a") as f:
            f.write(f"{timestamp},{objects_count}\n")

    def write_file_header(self):
        with open(self.args.log, "w") as f:
            f.write("timestamp,people_count\n")

        
    def run(self):
        """
        Run the stream processing thread's main loop.
        """
        while True:
            try:
                self.process()
            except:
                traceback.print_exc()
                
    @staticmethod
    def usage():
        """
        Return help text for when the app is started with -h or --help
        """
        return videoSource.Usage() + videoOutput.Usage() + Model.Usage()
        
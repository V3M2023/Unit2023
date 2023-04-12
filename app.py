import argparse
import os

import flask

from stream import Stream

parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter, epilog=Stream.usage())

parser.add_argument("--host", default='0.0.0.0', type=str, help="interface for the webserver to use (default is all interfaces, 0.0.0.0)")
parser.add_argument("--port", default=8050, type=int, help="port used for webserver (default is 8050)")
parser.add_argument("--ssl-key", default=os.getenv('SSL_KEY'), type=str, help="path to PEM-encoded SSL/TLS key file for enabling HTTPS")
parser.add_argument("--ssl-cert", default=os.getenv('SSL_CERT'), type=str, help="path to PEM-encoded SSL/TLS certificate file for enabling HTTPS")
parser.add_argument("--title", default='V3M Cam', type=str, help="the title of the webpage as shown in the browser")
parser.add_argument("--input", default='/dev/video0', type=str, help="input camera stream or video file")
parser.add_argument("--output", default='webrtc://@:8554/output', type=str, help="WebRTC output stream to serve from --input")
# parser.add_argument("--detection", default='peoplenet', type=str, help="load object detection model (see detectNet arguments)")
parser.add_argument("--detection", default='ssd-mobilenet-v2', type=str, help="load object detection model (see detectNet arguments)")

parser.add_argument("--classification", default='', type=str, help="load classification model (see imageNet arguments)")
parser.add_argument("--segmentation", default='', type=str, help="load semantic segmentation model (see segNet arguments)")
parser.add_argument("--background", default='', type=str, help="load background removal model (see backgroundNet arguments)")
parser.add_argument("--action", default='', type=str, help="load action recognition model (see actionNet arguments)")
parser.add_argument("--pose", default='', type=str, help="load action recognition model (see actionNet arguments)")
parser.add_argument("--labels", default='', type=str, help="path to labels.txt for loading a custom model")
parser.add_argument("--colors", default='', type=str, help="path to colors.txt for loading a custom model")
parser.add_argument("--input-layer", default='', type=str, help="name of input layer for loading a custom model")
parser.add_argument("--output-layer", default='', type=str, help="name of output layer(s) for loading a custom model (comma-separated if multiple)")

args = parser.parse_known_args()[0]

app = flask.Flask(__name__)
stream = Stream(args)


@app.route('/')
def index():
    return flask.render_template(
        'index.html',
        send_webrtc=False,
        input_stream=args.input,
        output_stream=args.output,
        classification=os.path.basename(args.classification),
        detection=os.path.basename(args.detection),
        segmentation=os.path.basename(args.segmentation),
        pose=os.path.basename(args.pose),
        action=os.path.basename(args.action),
        background=os.path.basename(args.background)
    )

@app.route('/data', methods=['GET'])
def data():
    return flask.jsonify(history=stream.count_history)

@app.route('/durations', methods=['GET'])
def durations():
    return flask.jsonify(history=stream.get_duration_history())

# start stream thread
stream.start()

# check if HTTPS/SSL requested
ssl_context = None

if args.ssl_cert and args.ssl_key:
    ssl_context = (args.ssl_cert, args.ssl_key)

app.run(host=args.host, port=args.port, ssl_context=ssl_context, debug=True, use_reloader=False)

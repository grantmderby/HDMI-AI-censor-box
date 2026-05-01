from flask import Flask, request, render_template_string
import json, os

CONFIG_FILE = 'config.json'
LEVELS = ['KIDS', 'STRICT', 'STANDARD', 'LIGHT']
app = Flask(__name__)

HTML = '''
<!DOCTYPE html>
<html>
<head>
 <title>AI Censor Box Control</title>
 <meta name="viewport" content="width=device-width, initial-scale=1">
 <style>
    body { font-family: Arial, sans-serif; padding: 40px;
            max-width: 500px; margin: auto; background: #f5f5f5; }
    h2 { color: #1F4E79; }
    .lvl { display: inline-block; margin: 8px; }
    button {
        padding: 14px 28px; font-size: 16px; border: none;
        border-radius: 6px; cursor: pointer; background: #2E75B6;
        color: white; transition: background 0.2s;
    }
 button:hover { background: #1F4E79; }
 .active-level { background: #1F4E79; font-weight: bold;
                box-shadow: 0 0 0 3px #FFF2CC; }
 .status {  margin-top: 20px; padding: 12px 16px;
            background: #E2EFDA; border-radius: 6px;
            color: #375623; font-weight: bold; }
 </style>
</head>
<body>
 <h2>AI Censor Box Control</h2>
 <p>Select a censorship level. Changes take effect within ~1 second.</p>
 <div class="status">Current Level: {{ level }}</div>
 <br>
 {% for l in levels %}
 <form method="post" action="/set_level" class="lvl">
    <input type="hidden" name="level" value="{{ l }}">
    <button class="{% if l == level %}active-level{% endif %}">
        {{ l }}
    </button>
</form>
{% endfor %}
<br><br>
 <small style="color:#888">
    Simulation: webcam + CPU YOLOv8 &nbsp;|&nbsp;
    Hardware: HDMI capture + Hailo-8L NPU
 </small>
</body>
</html>
'''
def read_level():
    try:
        with open(CONFIG_FILE) as f:
            return json.load(f).get('level', 'STRICT')
    except:
        return 'STRICT'
@app.route('/')
def index():
    return render_template_string(HTML, level=read_level(), levels=LEVELS)

@app.route('/set_level', methods=['POST'])
def set_level():
    new_level = request.form.get('level')
    if new_level in LEVELS:
        # Atomic write — pipeline always reads a complete JSON
        tmp = CONFIG_FILE + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'level': new_level}, f)
        os.replace(tmp, CONFIG_FILE)
    return index()
if __name__ == '__main__':
 app.run(host='0.0.0.0', port=5000, debug=False)
 #Right now, the webcam is certainty interesting. Is it possible to make the videostream on my pc censored? For example, watching youtube?
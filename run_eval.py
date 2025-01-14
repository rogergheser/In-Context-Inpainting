import os
import argparse
import signal
import subprocess
parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--strength', type=float, default=0.8)
parser.add_argument('--batch_size', type=int, default=4)
args = parser.parse_args()

files = {}
for dir in os.listdir('inputs'):
    if os.path.isdir('inputs/' + dir):
        files[dir] = []
        for image, mask in zip(
            sorted(os.listdir('inputs/' + dir + '/images/')),
            sorted(os.listdir('inputs/' + dir + '/alphas/'))
        ):
            files[dir].append((
                'inputs/' + dir + '/images/' + image,
                'inputs/' + dir + '/alphas/' + mask
                ))

# For DEBUG
for key in files.keys():
    print(key)
    for image, mask in files[key]:
        print(image, mask)

RES_DIR = 'outputs_guid12.5_blurMask/a{}_s{}'.format(args.alpha, args.strength)
os.makedirs(RES_DIR, exist_ok=True)
for key in files.keys():
    os.makedirs(os.path.join(RES_DIR, key), exist_ok=True)
    init_image, init_mask = files[key][0] if "dog" not in key else files[key][2]
    
    for idx, (image, mask) in enumerate(files[key]):
        if idx == 0 and "dog" not in key:
            continue
        if idx == 2 and "dog" in key:
            continue
        os.makedirs(os.path.join(RES_DIR, key), exist_ok=True)
        command = [
            "python", "image_blending.py",
            "--prompt", f"An image of a {key}",
            "--init_image", init_image,
            "--guiding_image", image,
            "--mask", init_mask,
            "--device", str(args.device),
            "--image_guided_prompt_gen",
            "--alpha", str(args.alpha),
            "--strength", str(args.strength),
            "--batch_size", str(args.batch_size),
            "--output_path", os.path.join(RES_DIR, key, str(idx)+'_')
        ]
        process = subprocess.Popen(command)
        try:
            process.wait()
        except KeyboardInterrupt:
            process.send_signal(signal.SIGINT)
            process.wait()
            raise

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='mps')
parser.add_argument('--alpha', type=float, default=0.2)
parser.add_argument('--strength', type=float, default=0.9)
parser.add_argument('--batch_size', type=int, default=2)
args = parser.parse_args()

files = {}
for dir in os.listdir('inputs'):
    if os.path.isdir('inputs/' + dir):
        files[dir] = []
        for image, mask in zip(
            os.listdir('inputs/' + dir + '/images/'),
            os.listdir('inputs/' + dir + '/alphas/')
        ):
            files[dir].append((
                'inputs/' + dir + '/images/' + image,
                'inputs/' + dir + '/alphas/' + mask
                ))
        files[dir] = sorted(files[dir], key=lambda x: str(x[0]).split('_')[-1])

# For DEBUG
for key in files.keys():
    print(key)
    for image, mask in files[key]:
        print(image, mask)

RES_DIR = 'outputs/a{}_s{}'.format(args.alpha, args.strength)
os.makedirs(RES_DIR, exist_ok=True)
for key in files.keys():
    os.makedirs(os.path.join(RES_DIR, key), exist_ok=True)
    init_image = files[key][0][0]
    for idx, (image, mask) in enumerate(files[key]):
        if idx == 0:
            continue
        os.makedirs(os.path.join(RES_DIR, key, str(idx)), exist_ok=True)
        os.system(f"python image_blending.py" +
                    f" --prompt 'An image of a {key}'" +
                    f" --init_image {init_image} " +
                    f" --guiding_image {image}" +
                    f" --mask {mask}" +
                    f" --device {args.device}" +
                    " --image_guided_prompt_gen" +
                    f" --alpha {args.alpha}" +
                    f" --strength {args.strength}" +
                    f" --batch_size {args.batch_size}" +
                    f" --output_path {RES_DIR}/{key}/{str(idx)}/")
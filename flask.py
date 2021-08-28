from flask import Flask, request, jsonify, send_file
import werkzeug

from absl import app, flags, logging
from absl.flags import FLAGS

import cv2
import os
import pathlib
import numpy as np
import tensorflow as tf

from modules.models import RRDB_Model
from modules.utils import (load_yaml, set_memory_growth, imresize_np,
                           tensor2img, rgb2ycbcr, create_lr_hr_pair,
                           calculate_psnr, calculate_ssim)

app = Flask(__name__)



@app.route('/upload', methods=['POST'])
def updload():
    if(request.method == 'POST'):
        imagefile = request.files['image']
        filename = werkzeug.utils.secure_filename(imagefile.filename)
        imagefile.save("./data/" + filename+".jpg")

        inputImage = "./data/"+filename+".jpg"

        flags.DEFINE_string('cfg_path', './configs/esrgan.yaml', 'config file path')
        flags.DEFINE_string('gpu', '0', 'which gpu to use')
        flags.DEFINE_string('img_path', inputImage, 'path to input image')

        # init
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

        logger = tf.get_logger()
        logger.disabled = True
        logger.setLevel(logging.FATAL)
        set_memory_growth()

        cfg = load_yaml(FLAGS.cfg_path)

        # evaluation
        if FLAGS.img_path:
            print("[*] Processing on single image {}".format(FLAGS.img_path))
            raw_img = cv2.imread(FLAGS.img_path)
            lr_img, hr_img = create_lr_hr_pair(raw_img, cfg['scale'])

            sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
            bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)

            str_format = "[{}] PSNR/SSIM: Bic={:.2f}db/{:.2f}, SR={:.2f}db/{:.2f}"
            print(str_format.format(
                os.path.basename(FLAGS.img_path),
                calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
                calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))))
            result_img_path = './Bic_SR_HR_' + os.path.basename(FLAGS.img_path)
            print("[*] write the result image {}".format(result_img_path))
            results_img = hr_image #np.concatenate((bic_img, sr_img, hr_img), 1)
            cv2.imwrite(result_img_path, results_img)
        else:
            print("[*] Processing on Set5 and Set14, and write results")
            results_path = './results/' + cfg['sub_name'] + '/'

            for key, path in cfg['test_dataset'].items():
                print("'{}' form {}\n  PSNR/SSIM".format(key, path))
                dataset_name = key.replace('_path', '')
                pathlib.Path(results_path + dataset_name).mkdir(
                    parents=True, exist_ok=True)

                for img_name in os.listdir(path):
                    raw_img = cv2.imread(os.path.join(path, img_name))
                    lr_img, hr_img = create_lr_hr_pair(raw_img, cfg['scale'])

                    sr_img = tensor2img(model(lr_img[np.newaxis, :] / 255))
                    bic_img = imresize_np(lr_img, cfg['scale']).astype(np.uint8)

                    str_format = "  [{}] Bic={:.2f}db/{:.2f}, SR={:.2f}db/{:.2f}"
                    print(str_format.format(
                        img_name + ' ' * max(0, 20 - len(img_name)),
                        calculate_psnr(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                        calculate_ssim(rgb2ycbcr(bic_img), rgb2ycbcr(hr_img)),
                        calculate_psnr(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img)),
                        calculate_ssim(rgb2ycbcr(sr_img), rgb2ycbcr(hr_img))))
                    result_img_path = os.path.join(
                        results_path + dataset_name, 'Bic_SR_HR_' + img_name)
                    results_img = np.concatenate((bic_img, sr_img, hr_img), 1)
                    cv2.imwrite(result_img_path, results_img)
            print("[*] write the visual results in {}".format(results_path))

        #outputImage = "Upscaled_%s" % filename+".jpg"
        outputImage =filename+".jpg"
        
        return jsonify({
            "result": outputImage
        })
        
@app.route('/download/<fname>', methods=['GET'])
def download(fname):
    return send_file(fname, as_attachment=True)

#app.run(host="0.0.0.0", port="8080")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8080")
